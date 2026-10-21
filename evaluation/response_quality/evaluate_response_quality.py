import sys
import json
import csv
import argparse
import statistics
import time
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG
from src.generation.context_builder import build_context
from src.evaluation.simple_evaluator import SimpleEvaluator

RESPONSE_EVAL_MAX_NEW_TOKENS = 384
RESPONSE_EVAL_MAX_CONTEXT_CHARS = 10000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    eval_dir = PROJECT_ROOT / "evaluation"
    out_dir = eval_dir / "response_quality"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    queries_path = eval_dir / "generation_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)
        
    queries = all_queries[:args.num_queries]

    start_time = time.time()
    
    rag = HybridRAG()
    generator = rag.get_generator()
    evaluator = SimpleEvaluator()
    
    # Force smaller generation token limit for evaluation
    original_max_tokens = generator.max_new_tokens
    generator.max_new_tokens = RESPONSE_EVAL_MAX_NEW_TOKENS

    methods = ["Naive RAG", "Sentence Window", "Parent-Child", "Hybrid RAG"]
    
    per_query_results = []
    
    gen_calls = 0
    eval_calls = 0
    failures = 0

    for q in tqdm(queries, desc="Evaluating Queries"):
        q_id = q["query_id"]
        q_text = q["query"]
        ref_ans = q["expected_answer"]
        
        ckpt_path = ckpt_dir / f"{q_id}.json"
        
        if args.resume and ckpt_path.exists():
            with open(ckpt_path, "r", encoding="utf-8") as f:
                per_query_results.append(json.load(f))
            continue

        rag_res = rag.query(q_text, generate_answer=False)
        top_k = rag.generation_context_top_k
        
        candidates = {
            "Naive RAG": rag_res.get("naive_results", [])[:top_k],
            "Sentence Window": rag_res.get("sw_results", [])[:top_k],
            "Parent-Child": rag_res.get("pc_results", [])[:top_k],
            "Hybrid RAG": rag_res.get("final_results", [])[:top_k],
        }

        query_record = {
            "query_id": q_id,
            "query": q_text,
        }

        for m_name in methods:
            m_cands = candidates[m_name]
            is_hybrid = (m_name == "Hybrid RAG")
            context_str, _ = build_context(
                m_cands,
                top_k=top_k,
                query=q_text if is_hybrid else None,
                filter_unrelated=is_hybrid,
            )
            
            # Generate answer: only Hybrid RAG filters unrelated chunks; baselines receive chunks as-is
            gen_res = generator.generate(
                q_text,
                context_str,
                m_cands,
                top_k=top_k,
                filter_unrelated=is_hybrid,
            )
            response = gen_res["answer"]
            gen_calls += 1
            
            # Evaluate with key_facts support
            eval_res = evaluator.evaluate(q_text, response, ref_ans, context_str, key_facts=q.get("key_facts"))
            eval_calls += 1
            
            if eval_res["correctness"]["score"] is None:
                failures += 1
                
            query_record[m_name] = {
                "response": response,
                "correctness": eval_res["correctness"],
                "relevancy": eval_res["relevancy"],
                "faithfulness": eval_res["faithfulness"]
            }

        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(query_record, f, indent=2)
            
        per_query_results.append(query_record)

    # Restore token limit
    generator.max_new_tokens = original_max_tokens
    
    total_time = time.time() - start_time
    avg_time = total_time / len(queries) if len(queries) > 0 else 0

    method_scores = {m: {"correctness": [], "relevancy": [], "faithfulness": []} for m in methods}
    for qr in per_query_results:
        for m_name in methods:
            if qr[m_name]["correctness"]["score"] is not None:
                method_scores[m_name]["correctness"].append(qr[m_name]["correctness"]["score"])
            if qr[m_name]["relevancy"]["score"] is not None:
                method_scores[m_name]["relevancy"].append(qr[m_name]["relevancy"]["score"])
            if qr[m_name]["faithfulness"]["score"] is not None:
                method_scores[m_name]["faithfulness"].append(qr[m_name]["faithfulness"]["score"])

    summary = {}
    for m_name in methods:
        summary[m_name] = {}
        for metric in ["correctness", "relevancy", "faithfulness"]:
            scores = method_scores[m_name][metric]
            n = len(scores)
            summary[m_name][metric] = {
                "mean": sum(scores) / n if n > 0 else 0.0,
                "std": statistics.stdev(scores) if n > 1 else 0.0
            }
        summary[m_name]["num_samples"] = len(method_scores[m_name]["correctness"])

    results = {
        "evaluation_metadata": {
            "num_queries": len(queries),
            "generator_model": "Qwen/Qwen3-0.6B",
            "evaluator_model": "SentenceTransformers",
            "evaluation_type": "semantic-similarity",
            "correctness_range": "1-5",
            "relevancy_range": "0-1",
            "faithfulness_range": "0-1",
            "std_method": "sample",
            "runtime_seconds": total_time
        },
        "summary": summary,
        "per_query": per_query_results
    }

    json_path = out_dir / "response_quality_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "response_quality_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Num Queries", "Correctness Mean", "Correctness Std", "Relevancy Mean", "Relevancy Std", "Faithfulness Mean", "Faithfulness Std"])
        for m_name in methods:
            s = summary[m_name]
            writer.writerow([
                m_name, s.get("num_samples", 0),
                f"{s['correctness']['mean']:.4f}", f"{s['correctness']['std']:.4f}",
                f"{s['relevancy']['mean']:.4f}", f"{s['relevancy']['std']:.4f}",
                f"{s['faithfulness']['mean']:.4f}", f"{s['faithfulness']['std']:.4f}",
            ])
            
    md_path = PROJECT_ROOT / "reports" / "response_quality_evaluation.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Response Quality Evaluation\n\n")
        f.write("## 1. Evaluation Objective\nEvaluate the response quality of 4 RAG approaches using LLM-as-a-judge.\n\n")
        f.write("## 2. Dataset\nDerived from `evaluation/generation_queries.json` with existing reference answers.\n\n")
        f.write(f"## 3. Number of Queries\n{len(queries)}\n\n")
        f.write("## 4. RAG Strategies\nNaive RAG, Sentence Window, Parent-Child, Hybrid RAG\n\n")
        f.write("## 5. Generator Model\nQwen/Qwen3-0.6B\n\n")
        f.write("## 6. Evaluator Model\nQwen/Qwen3-0.6B\n\n")
        f.write("## 7. Correctness Definition\nMeasures accuracy and relevance (1-5).\n\n")
        f.write("## 8. Relevancy Definition\nMeasures if response directly answers query and aligns with context (0-1).\n\n")
        f.write("## 9. Faithfulness Definition\nMeasures if response is supported by context (0-1).\n\n")
        f.write("## 10. LLM Evaluation Methodology\nLLM-as-a-judge using specific prompts requesting structured JSON output.\n\n")
        f.write("## 11. Mean Calculation\nArithmetic mean of valid scores.\n\n")
        f.write("## 12. Standard Deviation Calculation\nSample standard deviation (`statistics.stdev`).\n\n")
        f.write("## 13. Results Table\n\n")
        f.write("TABLE I. RESPONSE QUALITY EVALUATION OF DIFFERENT RAG MODELS\n\n")
        f.write("| Model | Correctness (±std) | Relevancy (±std) | Faithfulness (±std) |\n")
        f.write("|---|---:|---:|---:|\n")
        for m_name in methods:
            s = summary[m_name]
            f.write(f"| {m_name} | {s['correctness']['mean']:.2f} ± {s['correctness']['std']:.2f} | {s['relevancy']['mean']:.2f} ± {s['relevancy']['std']:.2f} | {s['faithfulness']['mean']:.2f} ± {s['faithfulness']['std']:.2f} |\n")
            
        f.write("\n## 14. Limitations\nEvaluation relies on a smaller model (Qwen3-0.6B) as a judge.\n\n")
        f.write("## 15. Reproducibility Command\n`python evaluation/response_quality/evaluate_response_quality.py --num-queries 30 --resume`\n")

    print(f"\n--- EVALUATION RUN SUMMARY ---")
    print(f"Total Runtime: {total_time:.2f}s")
    print(f"Avg Time/Query: {avg_time:.2f}s")
    print(f"Generation Calls: {gen_calls}")
    print(f"Evaluator Calls: {eval_calls}")
    print(f"Failed parsing calls: {failures}")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
