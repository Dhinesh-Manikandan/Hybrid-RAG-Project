"""
run_standardized_benchmark.py -- Standardized Production Retrieval Benchmark.

Applies the standardized timing methodology:
1. Loads HybridRAG.
2. Performs 1 pre-warm query to eliminate model load & threadpool initialization cold-start.
3. Times end-to-end rag.query() across all 82 benchmark queries for:
   A. Baseline: K=15, V=3, BS=32
   B. Proposed Optimization: K=10, V=3, BS=32
4. Reports exact comparative metrics: Hit@1, Hit@3, Hit@5, MRR, Mean, Median, P95 latency.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from retrieve import HybridRAG


def compute_mrr(retrieved_docs: list[str], expected_docs: set[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected_docs:
            return 1.0 / rank
    return 0.0


def run_benchmark_for_config(rag: HybridRAG, queries: list[dict], fusion_top_k: int, config_label: str) -> dict:
    latencies = []
    hits_1 = []
    hits_3 = []
    hits_5 = []
    mrrs = []

    print(f"\n--- Running Standardized Benchmark: {config_label} (K={fusion_top_k}) ---")

    for idx, item in enumerate(queries, start=1):
        q = item["query"]
        expected_docs = set(item["expected_documents"])

        t0 = time.perf_counter()
        res = rag.query(q, fusion_top_k=fusion_top_k)
        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed)

        final_docs = [r.get("source_file") for r in res["final_results"]]
        h1 = bool(final_docs and final_docs[0] in expected_docs)
        h3 = any(d in expected_docs for d in final_docs[:3])
        h5 = any(d in expected_docs for d in final_docs[:5])
        mrr_val = compute_mrr(final_docs, expected_docs)

        hits_1.append(h1)
        hits_3.append(h3)
        hits_5.append(h5)
        mrrs.append(mrr_val)

        if idx % 20 == 0 or idx == len(queries):
            print(f"  [{idx:02d}/{len(queries)}] Last: {t_elapsed:.2f}s | Running Mean: {np.mean(latencies):.2f}s")

    n_total = len(queries)
    return {
        "config": config_label,
        "fusion_top_k": fusion_top_k,
        "hit_at_1": round(sum(hits_1) / n_total * 100, 2),
        "hit_at_3": round(sum(hits_3) / n_total * 100, 2),
        "hit_at_5": round(sum(hits_5) / n_total * 100, 2),
        "mrr": round(float(np.mean(mrrs)), 4),
        "mean_latency": round(float(np.mean(latencies)), 2),
        "median_latency": round(float(np.median(latencies)), 2),
        "p95_latency": round(float(np.percentile(latencies, 95)), 2),
    }


def main():
    benchmark_file = PROJECT_ROOT / "evaluation" / "multi_document_queries.json"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} benchmark queries.")
    rag = HybridRAG()

    # Pre-warm
    print("Executing single pre-warm query to eliminate cold-start...")
    t_warm = time.perf_counter()
    _ = rag.query("Warmup query to initialize PyTorch threadpool and CrossEncoder weights")
    print(f"Warmup complete ({time.perf_counter() - t_warm:.2f} s). Ready for standardized timing.")

    # 1. Configuration A: Baseline (K=15, V=3, BS=32)
    res_a = run_benchmark_for_config(rag, queries, fusion_top_k=15, config_label="Configuration A: Baseline (K=15, V=3)")

    # 2. Configuration B: Optimized (K=10, V=3, BS=32)
    res_b = run_benchmark_for_config(rag, queries, fusion_top_k=10, config_label="Configuration B: Optimized (K=10, V=3)")

    print("\n" + "=" * 95)
    print("STANDARDIZED BENCHMARK RECONCILIATION COMPARISON TABLE")
    print("=" * 95)
    header = f"{'Configuration':<35} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<7} | {'Mean(s)':<8} | {'Med(s)':<8} | {'P95(s)':<8}"
    print(header)
    print("-" * len(header))
    for r in [res_a, res_b]:
        print(f"{r['config']:<35} | {r['hit_at_1']:<7.1f}% | {r['hit_at_3']:<7.1f}% | {r['hit_at_5']:<7.1f}% | {r['mrr']:<7.4f} | {r['mean_latency']:<8.2f} | {r['median_latency']:<8.2f} | {r['p95_latency']:<8.2f}")
    print("=" * 95)

    # Save to JSON
    out_path = PROJECT_ROOT / "evaluation" / "results" / "standardized_reconciliation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"configuration_a": res_a, "configuration_b": res_b}, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
