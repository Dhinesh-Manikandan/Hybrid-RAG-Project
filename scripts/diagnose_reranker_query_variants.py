"""
diagnose_reranker_query_variants.py -- Controlled CrossEncoder Query-Variant Reranking Ablation.

Controlled diagnostic to evaluate whether CrossEncoder reranking should evaluate:
A. Original query only (baseline)
B. Each reformulated query individually
C. Score aggregations (Mean, Max, Weighted Mean)
D. Rank aggregations (Mean Rank, Best Rank, Reciprocal Rank / RRF)

Candidate pools are retrieved ONCE per query using the existing pipeline, ensuring
strict experimental control across all reranking strategies.

DO NOT MODIFY PRODUCTION CODE.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG
from src.reranking.cross_encoder import get_reranker
from src.retrieval.query_reformulation import reformulate_query
from src.config import NUM_REFORMULATIONS


def evaluate_query_reranking(
    rag: HybridRAG,
    reranker: Any,
    query: str,
    target_section: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the controlled reranking ablation for a single query.
    
    1. Retrieves candidate pool ONCE using rag.query().
    2. Uses existing reformulator to obtain query variants.
    3. Evaluates all candidates across all query variants and aggregation strategies.
    """
    # 1. Retrieve candidates using existing pipeline
    # We use res['fused_results'] or res['diverse_results'] (which is the candidate pool passed to reranker)
    res = rag.query(query)
    query_variants = res["query_variants"]
    # Candidate pool entering reranking in production is res["diverse_results"]
    candidate_pool = res["diverse_results"]

    # 2. Extract texts for reranking (canonical retrieval_text or text)
    candidate_texts = [c.get("retrieval_text", c["text"]) for c in candidate_pool]
    n_candidates = len(candidate_pool)

    # 3. Compute raw CrossEncoder scores for each query variant
    # variant_scores: list of length len(query_variants), each is list of scores for all candidates
    variant_scores: list[list[float]] = []
    for q_var in query_variants:
        pairs = [(q_var, text) for text in candidate_texts]
        raw_scores = reranker.predict(pairs)
        variant_scores.append([float(s) for s in raw_scores])

    # Ensure we have at least 3 variants (Original, Ref1, Ref2)
    orig_scores = variant_scores[0]
    ref1_scores = variant_scores[1] if len(variant_scores) > 1 else orig_scores
    ref2_scores = variant_scores[2] if len(variant_scores) > 2 else orig_scores

    # 4. Compute Score Aggregations
    mean_scores = []
    max_scores = []
    weighted_scores = []  # 0.5 * orig + 0.25 * ref1 + 0.25 * ref2

    for i in range(n_candidates):
        s_orig = orig_scores[i]
        s_ref1 = ref1_scores[i]
        s_ref2 = ref2_scores[i]

        all_s = [variant_scores[v][i] for v in range(len(query_variants))]
        mean_scores.append(sum(all_s) / len(all_s))
        max_scores.append(max(all_s))
        weighted_scores.append(0.50 * s_orig + 0.25 * s_ref1 + 0.25 * s_ref2)

    # 5. Compute Per-Variant Rankings (1-based: 1 = highest score)
    def compute_ranks(scores: list[float]) -> list[int]:
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranks = [0] * len(scores)
        for rank, (orig_idx, _) in enumerate(indexed, start=1):
            ranks[orig_idx] = rank
        return ranks

    orig_ranks = compute_ranks(orig_scores)
    ref1_ranks = compute_ranks(ref1_scores)
    ref2_ranks = compute_ranks(ref2_scores)

    # Rank Aggregations
    mean_ranks = []
    best_ranks = []
    reciprocal_ranks = []  # RRF-style: sum(1 / (60 + r))

    for i in range(n_candidates):
        r_list = [orig_ranks[i], ref1_ranks[i], ref2_ranks[i]]
        mean_ranks.append(sum(r_list) / len(r_list))
        best_ranks.append(min(r_list))
        reciprocal_ranks.append(sum(1.0 / (60.0 + r) for r in r_list))

    # Package candidate information with all scores and ranks
    annotated_candidates = []
    for i, c in enumerate(candidate_pool):
        item = {
            "index": i,
            "chunk_id": c.get("chunk_id", ""),
            "source_chunk_id": c.get("source_chunk_id", ""),
            "section": c.get("section", ""),
            "rrf_score": c.get("rrf_score", 0.0),
            "retrieval_sources": c.get("retrieval_sources", []),
            "retrieval_text": candidate_texts[i],
            "score_orig": orig_scores[i],
            "score_ref1": ref1_scores[i],
            "score_ref2": ref2_scores[i],
            "score_mean": mean_scores[i],
            "score_max": max_scores[i],
            "score_weighted": weighted_scores[i],
            "rank_orig": orig_ranks[i],
            "rank_ref1": ref1_ranks[i],
            "rank_ref2": ref2_ranks[i],
            "rank_mean": mean_ranks[i],
            "rank_best": best_ranks[i],
            "score_rrf_rank": reciprocal_ranks[i],
        }
        annotated_candidates.append(item)

    # Ranking functions for each strategy
    strategies = {
        "Original Query (Baseline)": sorted(annotated_candidates, key=lambda x: x["score_orig"], reverse=True),
        "Reformulation 1":           sorted(annotated_candidates, key=lambda x: x["score_ref1"], reverse=True),
        "Reformulation 2":           sorted(annotated_candidates, key=lambda x: x["score_ref2"], reverse=True),
        "Mean Score":                sorted(annotated_candidates, key=lambda x: x["score_mean"], reverse=True),
        "Max Score":                 sorted(annotated_candidates, key=lambda x: x["score_max"], reverse=True),
        "Weighted Mean (0.5/0.25/0.25)": sorted(annotated_candidates, key=lambda x: x["score_weighted"], reverse=True),
        "Mean Rank":                 sorted(annotated_candidates, key=lambda x: (x["rank_mean"], -x["score_orig"])),
        "Best Rank":                 sorted(annotated_candidates, key=lambda x: (x["rank_best"], -x["score_orig"])),
        "Reciprocal Rank (RRF)":     sorted(annotated_candidates, key=lambda x: x["score_rrf_rank"], reverse=True),
    }

    if verbose:
        print("\n" + "=" * 90)
        print(f"QUERY: \"{query}\"")
        print("=" * 90)
        print("QUERY VARIANTS GENERATED:")
        for idx, v in enumerate(query_variants):
            label = "Original" if idx == 0 else f"Reformulation {idx}"
            print(f"  [{label}]: {v}")

        print(f"\nSHARED CANDIDATE POOL (retrieved once, {n_candidates} candidates):")
        print(f"{'Pool#':<6} | {'Section':<25} | {'Source Chunk ID':<28} | {'RRF Score':<10} | {'Sources'}")
        print("-" * 90)
        for i, c in enumerate(candidate_pool, start=1):
            sec = c.get("section", "")[:24]
            scid = c.get("source_chunk_id", "")[:27]
            rrf_s = f"{c.get('rrf_score', 0.0):.6f}"
            srcs = ",".join(c.get("retrieval_sources", []))
            print(f"{i:<6} | {sec:<25} | {scid:<28} | {rrf_s:<10} | {srcs}")

        print("\n" + "=" * 90)
        print("EXPERIMENT A & B: RAW SCORES AND PER-VARIANT RANKINGS")
        print("=" * 90)
        for name in ["Original Query (Baseline)", "Reformulation 1", "Reformulation 2"]:
            ranked = strategies[name]
            score_key = (
                "score_orig" if "Original" in name
                else ("score_ref1" if "1" in name else "score_ref2")
            )
            print(f"\n--- Strategy: {name} ---")
            print(f"{'Rank':<5} | {'Score':<9} | {'Section':<25} | {'Source Chunk ID':<28}")
            print("-" * 75)
            for r_idx, item in enumerate(ranked[:7], start=1):
                sec = item["section"][:24]
                scid = item["source_chunk_id"][:27]
                score = f"{item[score_key]:+.4f}"
                print(f"{r_idx:<5} | {score:<9} | {sec:<25} | {scid:<28}")

        print("\n" + "=" * 90)
        print("EXPERIMENT C: SCORE AGGREGATION RANKINGS")
        print("=" * 90)
        for name in ["Mean Score", "Max Score", "Weighted Mean (0.5/0.25/0.25)"]:
            ranked = strategies[name]
            score_key = (
                "score_mean" if "Mean Score" == name
                else ("score_max" if "Max Score" == name else "score_weighted")
            )
            print(f"\n--- Strategy: {name} ---")
            print(f"{'Rank':<5} | {'Agg Score':<10} | {'Section':<25} | {'Orig Score':<11} | {'Ref1 Score':<11} | {'Ref2 Score':<11}")
            print("-" * 90)
            for r_idx, item in enumerate(ranked[:7], start=1):
                sec = item["section"][:24]
                agg_s = f"{item[score_key]:+.4f}"
                s0 = f"{item['score_orig']:+.4f}"
                s1 = f"{item['score_ref1']:+.4f}"
                s2 = f"{item['score_ref2']:+.4f}"
                print(f"{r_idx:<5} | {agg_s:<10} | {sec:<25} | {s0:<11} | {s1:<11} | {s2:<11}")

        print("\n" + "=" * 90)
        print("RANK-BASED AGGREGATION DIAGNOSTIC")
        print("=" * 90)
        for name in ["Mean Rank", "Best Rank", "Reciprocal Rank (RRF)"]:
            ranked = strategies[name]
            val_key = (
                "rank_mean" if name == "Mean Rank"
                else ("rank_best" if name == "Best Rank" else "score_rrf_rank")
            )
            fmt = "{:.2f}" if "Mean" in name else ("{}" if "Best" in name else "{:.4f}")
            print(f"\n--- Strategy: {name} ---")
            print(f"{'Rank':<5} | {'Metric':<8} | {'Section':<25} | {'Orig Rk':<8} | {'Ref1 Rk':<8} | {'Ref2 Rk':<8}")
            print("-" * 75)
            for r_idx, item in enumerate(ranked[:7], start=1):
                sec = item["section"][:24]
                val = fmt.format(item[val_key])
                print(f"{r_idx:<5} | {val:<8} | {sec:<25} | {item['rank_orig']:<8} | {item['rank_ref1']:<8} | {item['rank_ref2']:<8}")

    return {
        "query": query,
        "query_variants": query_variants,
        "candidate_pool": candidate_pool,
        "annotated_candidates": annotated_candidates,
        "strategies": strategies,
    }


def main():
    print("=" * 90)
    print("CONTROLLED RERANKING ABLATION: CROSSENCODER QUERY-VARIANT EVALUATION")
    print("=" * 90)

    rag = HybridRAG()
    reranker = get_reranker(rag.reranker_model)

    # -------------------------------------------------------------
    # 1. Main Diagnostic Query: "Who is eligible for PMAY-U?"
    # -------------------------------------------------------------
    main_query = "Who is eligible for PMAY-U?"
    main_res = evaluate_query_reranking(
        rag=rag,
        reranker=reranker,
        query=main_query,
        target_section="ELIGIBILITY",
        verbose=True,
    )

    # -------------------------------------------------------------
    # Compare ELIGIBILITY vs APPLICATION PROCESS
    # -------------------------------------------------------------
    annotated = main_res["annotated_candidates"]
    eligibility_item = next((c for c in annotated if c["section"] == "ELIGIBILITY"), None)
    app_process_item = next((c for c in annotated if c["section"] == "APPLICATION PROCESS"), None)

    print("\n" + "=" * 90)
    print("COMPARISON: ELIGIBILITY vs APPLICATION PROCESS")
    print("=" * 90)
    print(f"{'Strategy':<30} | {'ELIGIBILITY (Score / Rank)':<28} | {'APPLICATION PROCESS (Score / Rank)':<35} | {'Eligibility Wins?'}")
    print("-" * 110)

    strategies_order = [
        ("Original Query (Baseline)", "score_orig", "rank_orig"),
        ("Reformulation 1",           "score_ref1", "rank_ref1"),
        ("Reformulation 2",           "score_ref2", "rank_ref2"),
        ("Mean Score",                "score_mean", None),
        ("Max Score",                 "score_max",  None),
        ("Weighted Mean (0.5/0.25/0.25)", "score_weighted", None),
        ("Mean Rank",                 "rank_mean",  None),
        ("Best Rank",                 "rank_best",  None),
        ("Reciprocal Rank (RRF)",     "score_rrf_rank", None),
    ]

    table_data = []
    for strat_name, score_col, rank_col in strategies_order:
        ranked_list = main_res["strategies"][strat_name]
        
        # Determine 1-based rank in strategy's sorted list
        elig_rank = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "ELIGIBILITY"), -1)
        app_rank  = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "APPLICATION PROCESS"), -1)

        elig_val = eligibility_item[score_col] if eligibility_item else 0.0
        app_val  = app_process_item[score_col] if app_process_item else 0.0

        if "Rank" in strat_name and strat_name not in ["Reciprocal Rank (RRF)"]:
            elig_str = f"val={elig_val:.2f}, rank=#{elig_rank}"
            app_str  = f"val={app_val:.2f}, rank=#{app_rank}"
        else:
            elig_str = f"score={elig_val:+.4f}, rank=#{elig_rank}"
            app_str  = f"score={app_val:+.4f}, rank=#{app_rank}"

        wins = "YES (Rank 1)" if elig_rank == 1 else ("YES" if elig_rank < app_rank else "NO")
        print(f"{strat_name:<30} | {elig_str:<28} | {app_str:<35} | {wins}")
        table_data.append((strat_name, elig_str, app_str, wins))

    # -------------------------------------------------------------
    # 2. Run the Full 5-Query Benchmark Evaluation Suite
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FULL 5-QUERY BENCHMARK EVALUATION SUITE")
    print("=" * 90)

    benchmark_queries = [
        {"query": "Who is eligible for PMAY-U?",         "target": "ELIGIBILITY"},
        {"query": "What are the eligibility criteria?", "target": "ELIGIBILITY"},
        {"query": "What benefits are available?",       "target": "BENEFITS"},
        {"query": "How can beneficiaries apply?",       "target": "APPLICATION PROCESS"},
        {"query": "What documents are required?",       "target": "DOCUMENTS REQUIRED"},
    ]

    strategy_names = list(main_res["strategies"].keys())
    # Track hits: {strat_name: {"hit@1": 0, "hit@3": 0, "hit@5": 0, "ranks": []}}
    strategy_metrics = {s: {"hit@1": 0, "hit@3": 0, "hit@5": 0, "ranks": []} for s in strategy_names}

    per_query_results = []

    for b_item in benchmark_queries:
        q = b_item["query"]
        target = b_item["target"]

        q_res = evaluate_query_reranking(
            rag=rag,
            reranker=reranker,
            query=q,
            target_section=target,
            verbose=False,
        )

        query_row = {"query": q, "target": target, "ranks": {}}

        for s_name in strategy_names:
            ranked = q_res["strategies"][s_name]
            # Find target section rank
            target_rank = None
            for r_idx, c in enumerate(ranked, start=1):
                if c["section"] == target:
                    target_rank = r_idx
                    break

            query_row["ranks"][s_name] = target_rank

            if target_rank is not None:
                if target_rank == 1:
                    strategy_metrics[s_name]["hit@1"] += 1
                if target_rank <= 3:
                    strategy_metrics[s_name]["hit@3"] += 1
                if target_rank <= 5:
                    strategy_metrics[s_name]["hit@5"] += 1
                strategy_metrics[s_name]["ranks"].append(target_rank)
            else:
                strategy_metrics[s_name]["ranks"].append(999)

        per_query_results.append(query_row)

    # Print Per-Query Ranks Table
    print("\nPER-QUERY TARGET SECTION RANK ACROSS STRATEGIES:")
    print("-" * 120)
    header = f"{'Query':<36} | {'Target':<18} | " + " | ".join([f"{s[:10]:<10}" for s in strategy_names])
    print(header)
    print("-" * 120)

    for row in per_query_results:
        q_short = row["query"][:35]
        tgt = row["target"][:17]
        r_strs = [f"#{row['ranks'][s]}" if row['ranks'][s] is not None else "N/A" for s in strategy_names]
        r_formatted = " | ".join([f"{r:<10}" for r in r_strs])
        print(f"{q_short:<36} | {tgt:<18} | {r_formatted}")

    # Print Summary Benchmark Table (Hit@1, Hit@3, Hit@5)
    print("\n" + "=" * 90)
    print("BENCHMARK METRICS SUMMARY (5 Queries)")
    print("=" * 90)
    print(f"{'Strategy':<32} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Regressions vs Baseline?'}")
    print("-" * 80)

    baseline_ranks = [row["ranks"]["Original Query (Baseline)"] for row in per_query_results]

    for s_name in strategy_names:
        h1 = strategy_metrics[s_name]["hit@1"] / len(benchmark_queries)
        h3 = strategy_metrics[s_name]["hit@3"] / len(benchmark_queries)
        h5 = strategy_metrics[s_name]["hit@5"] / len(benchmark_queries)

        current_ranks = [row["ranks"][s_name] for row in per_query_results]
        
        # Check if queries 2..5 (index 1..4) regressed compared to baseline
        regressions = []
        for q_idx in range(1, len(benchmark_queries)):
            base_r = baseline_ranks[q_idx]
            curr_r = current_ranks[q_idx]
            if curr_r > base_r:
                regressions.append(f"Q{q_idx+1}: #{base_r}->#{curr_r}")

        reg_str = ", ".join(regressions) if regressions else "None (0 regressions)"
        print(f"{s_name:<32} | {h1:<8.2f} | {h3:<8.2f} | {h5:<8.2f} | {reg_str}")


if __name__ == "__main__":
    main()
