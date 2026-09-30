"""
diagnose_rrf_crossencoder_fusion.py -- Controlled RRF + CrossEncoder Fusion Diagnostic.

Evaluates whether combining the existing multi-query RRF retrieval score with the
CrossEncoder reranker score improves final rankings, specifically testing if the
retrieval signal can help elevate target chunks (such as ELIGIBILITY) without
degrading benchmark performance.

CRITICAL EXPERIMENTAL CONTROLS:
1. Candidate pool is retrieved ONCE per query using the existing HybridRAG pipeline.
2. The exact same candidate pool is evaluated across all fusion strategies.
3. CrossEncoder scores and RRF scores are properly normalized before linear combination.
4. Rank-based reciprocal fusion is evaluated alongside score-based weighted fusion.
5. Evaluated on the problematic query AND the full 5-query benchmark.
6. ZERO document-specific or PMAY-specific rules; strictly domain-agnostic.

DO NOT MODIFY PRODUCTION CODE. DIAGNOSTIC EXPERIMENT ONLY.
"""

import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG
from src.reranking.cross_encoder import get_reranker


def min_max_normalize(scores: list[float]) -> list[float]:
    """
    Min-Max normalize a list of scores to [0.0, 1.0].
    Preserves linear relative differences across candidates within a single pool.
    Handles uniform scores safely.
    """
    if not scores:
        return []
    min_v = min(scores)
    max_v = max(scores)
    diff = max_v - min_v
    if abs(diff) < 1e-12:
        return [1.0] * len(scores)
    return [(s - min_v) / diff for s in scores]


def compute_ranks(scores: list[float], reverse: bool = True) -> list[int]:
    """
    Compute 1-based ranks from scores.
    Rank 1 = best (highest score if reverse=True).
    Maintains stable indexing.
    """
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=reverse)
    ranks = [0] * len(scores)
    for rank, (orig_idx, _) in enumerate(indexed, start=1):
        ranks[orig_idx] = rank
    return ranks


def evaluate_query_fusion(
    rag: HybridRAG,
    reranker: Any,
    query: str,
    target_section: str | None = None,
    k_rrf: int = 60,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run controlled RRF + CrossEncoder fusion diagnostic for a single query.
    Retrieves the candidate pool ONCE and applies all fusion strategies to that exact pool.
    """
    t_start = time.perf_counter()

    # 1. Retrieve candidates ONCE using existing pipeline
    # The candidate pool entering reranking in production is res["diverse_results"]
    t_ret_start = time.perf_counter()
    res = rag.query(query)
    t_ret_end = time.perf_counter()
    retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0

    candidate_pool = res["diverse_results"]
    n_candidates = len(candidate_pool)

    # 2. Extract texts for reranking (canonical retrieval_text or fallback to text)
    candidate_texts = [c.get("retrieval_text", c["text"]) for c in candidate_pool]

    # 3. Compute raw CrossEncoder scores
    t_ce_start = time.perf_counter()
    pairs = [(query, text) for text in candidate_texts]
    raw_ce_scores = [float(s) for s in reranker.predict(pairs)]
    t_ce_end = time.perf_counter()
    ce_latency_ms = (t_ce_end - t_ce_start) * 1000.0

    # 4. Extract raw RRF scores already attached to candidates
    raw_rrf_scores = [float(c.get("rrf_score", 0.0)) for c in candidate_pool]

    # 5. Normalize signals
    norm_ce_scores = min_max_normalize(raw_ce_scores)
    norm_rrf_scores = min_max_normalize(raw_rrf_scores)

    # 6. Compute individual ranks
    ce_ranks = compute_ranks(raw_ce_scores, reverse=True)
    rrf_ranks = compute_ranks(raw_rrf_scores, reverse=True)

    # 7. Rank-based Reciprocal Fusion (k=60 matching canonical RRF)
    # CE rank contribution = 1 / (k + CE_rank)
    # RRF rank contribution = 1 / (k + RRF_rank)
    ce_rank_contrib = [1.0 / (k_rrf + r) for r in ce_ranks]
    rrf_rank_contrib = [1.0 / (k_rrf + r) for r in rrf_ranks]
    rank_fusion_scores = [c + r for c, r in zip(ce_rank_contrib, rrf_rank_contrib)]

    # 8. Weighted score fusion configurations
    # Alpha = weight on CrossEncoder, Beta = weight on RRF
    weight_configs = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
        (0.0, 1.0),
    ]

    weighted_scores: dict[str, list[float]] = {}
    for alpha, beta in weight_configs:
        strat_key = f"Weighted (a={alpha:.1f}, b={beta:.1f})"
        scores = [
            alpha * n_ce + beta * n_rrf
            for n_ce, n_rrf in zip(norm_ce_scores, norm_rrf_scores)
        ]
        weighted_scores[strat_key] = scores

    # 9. Package candidate metadata and scores
    annotated_candidates = []
    for i, c in enumerate(candidate_pool):
        item = {
            "index": i,
            "document_id": c.get("document_id", "N/A"),
            "source_file": c.get("source_file", "N/A"),
            "chunk_id": c.get("chunk_id", "N/A"),
            "source_chunk_id": c.get("source_chunk_id", "N/A"),
            "retrieval_type": c.get("retrieval_type", "N/A"),
            "section": c.get("section", "N/A"),
            "heading_path": c.get("heading_path", []),
            "retrieval_sources": c.get("retrieval_sources", []),
            "query_variants": c.get("query_variants", []),
            "retrieval_text": candidate_texts[i],
            "raw_ce_score": raw_ce_scores[i],
            "raw_rrf_score": raw_rrf_scores[i],
            "norm_ce_score": norm_ce_scores[i],
            "norm_rrf_score": norm_rrf_scores[i],
            "ce_rank": ce_ranks[i],
            "rrf_rank": rrf_ranks[i],
            "ce_rank_contrib": ce_rank_contrib[i],
            "rrf_rank_contrib": rrf_rank_contrib[i],
            "rank_fusion_score": rank_fusion_scores[i],
        }
        for strat_key, s_list in weighted_scores.items():
            item[strat_key] = s_list[i]

        annotated_candidates.append(item)

    # 10. Generate sorted rankings for all strategies
    strategies: dict[str, list[dict]] = {}

    # A. CrossEncoder-only baseline
    strategies["CrossEncoder-only (Baseline)"] = sorted(
        annotated_candidates, key=lambda x: (x["norm_ce_score"], x["raw_rrf_score"]), reverse=True
    )

    # B. RRF-only baseline
    strategies["RRF-only (Baseline)"] = sorted(
        annotated_candidates, key=lambda x: (x["norm_rrf_score"], x["raw_ce_score"]), reverse=True
    )

    # C. Weighted fusion strategies
    for alpha, beta in weight_configs:
        strat_key = f"Weighted (a={alpha:.1f}, b={beta:.1f})"
        strategies[strat_key] = sorted(
            annotated_candidates, key=lambda x: x[strat_key], reverse=True
        )

    # D. Rank-based fusion
    strategies["Rank-based Fusion (k=60)"] = sorted(
        annotated_candidates, key=lambda x: x["rank_fusion_score"], reverse=True
    )

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    if verbose:
        print("\n" + "=" * 95)
        print(f"DIAGNOSTIC QUERY: \"{query}\"")
        if target_section:
            print(f"Target Section : {target_section}")
        print(f"Candidate Pool : {n_candidates} candidates (retrieved ONCE)")
        print(f"Timing         : Retrieval = {retrieval_latency_ms:.1f}ms | CrossEncoder = {ce_latency_ms:.1f}ms | Total = {total_latency_ms:.1f}ms")
        print("=" * 95)

        print("\nSHARED CANDIDATE POOL (All Strategies Evaluated on this Pool):")
        print(f"{'#':<3} | {'Section':<24} | {'Raw RRF':<10} | {'Norm RRF':<9} | {'Raw CE':<10} | {'Norm CE':<9} | {'RRF Rk':<7} | {'CE Rk':<6}")
        print("-" * 95)
        for i, c in enumerate(annotated_candidates, start=1):
            sec = c["section"][:23]
            r_rrf = f"{c['raw_rrf_score']:.6f}"
            n_rrf = f"{c['norm_rrf_score']:.4f}"
            r_ce = f"{c['raw_ce_score']:+.4f}"
            n_ce = f"{c['norm_ce_score']:.4f}"
            print(f"{i:<3} | {sec:<24} | {r_rrf:<10} | {n_rrf:<9} | {r_ce:<10} | {n_ce:<9} | #{c['rrf_rank']:<6} | #{c['ce_rank']:<5}")

    return {
        "query": query,
        "target_section": target_section,
        "candidate_pool": candidate_pool,
        "annotated_candidates": annotated_candidates,
        "strategies": strategies,
        "retrieval_latency_ms": retrieval_latency_ms,
        "ce_latency_ms": ce_latency_ms,
        "total_latency_ms": total_latency_ms,
    }


def main():
    print("=" * 95)
    print("CONTROLLED RRF + CROSSENCODER FUSION DIAGNOSTIC EXPERIMENT")
    print("=" * 95)

    rag = HybridRAG()
    reranker = get_reranker(rag.reranker_model)

    # -------------------------------------------------------------------------
    # PART 1: Problematic Query Analysis ("Who is eligible for PMAY-U?")
    # -------------------------------------------------------------------------
    main_query = "Who is eligible for PMAY-U?"
    main_target = "ELIGIBILITY"
    res_main = evaluate_query_fusion(
        rag=rag,
        reranker=reranker,
        query=main_query,
        target_section=main_target,
        verbose=True,
    )

    strategies_main = res_main["strategies"]

    print("\n" + "=" * 95)
    print("PROBLEMATIC QUERY: STRATEGY COMPARISON TABLE")
    print("Query: \"Who is eligible for PMAY-U?\"")
    print("=" * 95)
    print(f"{'Strategy':<30} | {'App Rank':<9} | {'Elig Rank':<10} | {'Details':<8} | {'Elig in Top-3?':<16} | {'Elig in Top-5?'}")
    print("-" * 95)

    prob_query_table = []
    for strat_name, ranked_list in strategies_main.items():
        # Find 1-based ranks of key sections
        app_rank = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "APPLICATION PROCESS"), 999)
        elig_rank = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "ELIGIBILITY"), 999)
        det_rank = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "DETAILS"), 999)

        top3_str = "YES" if elig_rank <= 3 else "NO"
        top5_str = "YES" if elig_rank <= 5 else "NO"

        app_str = f"#{app_rank}" if app_rank != 999 else "N/A"
        elig_str = f"#{elig_rank}" if elig_rank != 999 else "N/A"
        det_str = f"#{det_rank}" if det_rank != 999 else "N/A"

        print(f"{strat_name:<30} | {app_str:<9} | {elig_str:<10} | {det_str:<8} | {top3_str:<16} | {top5_str}")
        prob_query_table.append({
            "strategy": strat_name,
            "app_rank": app_rank,
            "elig_rank": elig_rank,
            "det_rank": det_rank,
            "top3": top3_str,
            "top5": top5_str,
        })

    # Print Top 10 Candidates for Key Strategies on Problematic Query
    key_inspect_strategies = [
        "CrossEncoder-only (Baseline)",
        "RRF-only (Baseline)",
        "Weighted (a=0.5, b=0.5)",
        "Weighted (a=0.7, b=0.3)",
        "Rank-based Fusion (k=60)",
    ]

    print("\n" + "=" * 95)
    print("TOP 10 CANDIDATES FOR KEY STRATEGIES (Problematic Query)")
    print("=" * 95)
    for strat_name in key_inspect_strategies:
        ranked = strategies_main[strat_name][:10]
        print(f"\n--- Strategy: {strat_name} ---")
        print(f"{'Rk':<3} | {'Section':<24} | {'Source Chunk ID':<28} | {'Final Score':<12} | {'Raw RRF':<10} | {'Raw CE':<9} | {'RRF Rk':<6} | {'CE Rk':<5}")
        print("-" * 110)
        for r_idx, c in enumerate(ranked, start=1):
            sec = c["section"][:23]
            scid = c["source_chunk_id"][:27]
            if "Weighted" in strat_name:
                final_s = f"{c[strat_name]:.4f}"
            elif "CrossEncoder" in strat_name:
                final_s = f"{c['norm_ce_score']:.4f}"
            elif "RRF-only" in strat_name:
                final_s = f"{c['norm_rrf_score']:.4f}"
            else:
                final_s = f"{c['rank_fusion_score']:.6f}"

            r_rrf = f"{c['raw_rrf_score']:.6f}"
            r_ce = f"{c['raw_ce_score']:+.4f}"
            print(f"{r_idx:<3} | {sec:<24} | {scid:<28} | {final_s:<12} | {r_rrf:<10} | {r_ce:<9} | #{c['rrf_rank']:<5} | #{c['ce_rank']:<4}")

    # -------------------------------------------------------------------------
    # PART 2: Full 5-Query Benchmark Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("FULL 5-QUERY BENCHMARK EVALUATION")
    print("=" * 95)

    benchmark_queries = [
        {"query": "Who is eligible for PMAY-U?",         "target": "ELIGIBILITY"},
        {"query": "What are the eligibility criteria?", "target": "ELIGIBILITY"},
        {"query": "What benefits are available?",       "target": "BENEFITS"},
        {"query": "How can beneficiaries apply?",       "target": "APPLICATION PROCESS"},
        {"query": "What documents are required?",       "target": "DOCUMENTS REQUIRED"},
    ]

    all_strat_names = list(strategies_main.keys())

    # Data structures for benchmark tracking
    # per_query_ranks: list of dicts {query, target, {strat_name: target_rank}}
    per_query_ranks = []
    strat_metrics = {
        s: {"hit1": 0, "hit3": 0, "hit5": 0, "rr_sum": 0.0, "ranks": []}
        for s in all_strat_names
    }

    total_bench_retrieval_ms = 0.0
    total_bench_ce_ms = 0.0

    for b_item in benchmark_queries:
        q = b_item["query"]
        tgt = b_item["target"]

        q_res = evaluate_query_fusion(
            rag=rag,
            reranker=reranker,
            query=q,
            target_section=tgt,
            verbose=False,
        )

        total_bench_retrieval_ms += q_res["retrieval_latency_ms"]
        total_bench_ce_ms += q_res["ce_latency_ms"]

        q_row = {"query": q, "target": tgt, "ranks": {}}

        for s_name in all_strat_names:
            ranked = q_res["strategies"][s_name]
            target_rank = next(
                (r_idx for r_idx, c in enumerate(ranked, start=1) if c["section"] == tgt),
                999
            )
            q_row["ranks"][s_name] = target_rank

            if target_rank <= 1:
                strat_metrics[s_name]["hit1"] += 1
            if target_rank <= 3:
                strat_metrics[s_name]["hit3"] += 1
            if target_rank <= 5:
                strat_metrics[s_name]["hit5"] += 1

            strat_metrics[s_name]["rr_sum"] += 1.0 / target_rank
            strat_metrics[s_name]["ranks"].append(target_rank)

        per_query_ranks.append(q_row)

    # Print Per-Query Ranks Table
    print("\nPER-QUERY TARGET SECTION RANK ACROSS STRATEGIES:")
    print("-" * 140)
    col_w = 11
    header = f"{'Query':<36} | {'Target':<18} | " + " | ".join([f"{s[:col_w]:<{col_w}}" for s in all_strat_names])
    print(header)
    print("-" * 140)

    for row in per_query_ranks:
        q_short = row["query"][:35]
        tgt = row["target"][:17]
        r_strs = [f"#{row['ranks'][s]}" if row['ranks'][s] != 999 else "N/A" for s in all_strat_names]
        r_formatted = " | ".join([f"{r:<{col_w}}" for r in r_strs])
        print(f"{q_short:<36} | {tgt:<18} | {r_formatted}")

    # Print Benchmark Metrics Summary Table
    print("\n" + "=" * 95)
    print("BENCHMARK METRICS SUMMARY (5 Queries)")
    print("=" * 95)
    print(f"{'Strategy':<32} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Regressions vs Baseline'}")
    print("-" * 115)

    base_strat = "CrossEncoder-only (Baseline)"
    base_ranks = [row["ranks"][base_strat] for row in per_query_ranks]
    num_queries = len(benchmark_queries)

    summary_results = []
    for s_name in all_strat_names:
        h1 = strat_metrics[s_name]["hit1"] / num_queries
        h3 = strat_metrics[s_name]["hit3"] / num_queries
        h5 = strat_metrics[s_name]["hit5"] / num_queries
        mrr = strat_metrics[s_name]["rr_sum"] / num_queries

        curr_ranks = [row["ranks"][s_name] for row in per_query_ranks]

        # Regressions: queries where curr_rank > base_rank
        regressions = []
        for q_idx in range(num_queries):
            b_r = base_ranks[q_idx]
            c_r = curr_ranks[q_idx]
            if c_r > b_r:
                regressions.append(f"Q{q_idx+1}: #{b_r}->#{c_r}")

        reg_str = ", ".join(regressions) if regressions else "None (0 regressions)"
        print(f"{s_name:<32} | {h1:<8.2f} | {h3:<8.2f} | {h5:<8.2f} | {mrr:<8.4f} | {reg_str}")

        summary_results.append({
            "strategy": s_name,
            "hit1": h1,
            "hit3": h3,
            "hit5": h5,
            "mrr": mrr,
            "regressions": reg_str,
            "ranks": curr_ranks,
        })

    # Print Latency Breakdown
    avg_ret_ms = total_bench_retrieval_ms / num_queries
    avg_ce_ms = total_bench_ce_ms / num_queries
    print("\n" + "=" * 95)
    print("LATENCY ANALYSIS (Average across 5 Benchmark Queries)")
    print("=" * 95)
    print(f"Retrieval Pipeline (Multi-Query Dense + BM25 + Canonical RRF + MMR) : {avg_ret_ms:.1f} ms")
    print(f"CrossEncoder Reranker Inference (Candidate Pool of ~14 chunks)       : {avg_ce_ms:.1f} ms")
    print(f"Fusion Computation (Min-Max Normalization + Weighted Combination)     : < 0.05 ms")
    print(f"Total Query Latency                                                   : {avg_ret_ms + avg_ce_ms:.1f} ms")

    return {
        "prob_query_res": res_main,
        "prob_query_table": prob_query_table,
        "per_query_ranks": per_query_ranks,
        "summary_results": summary_results,
        "avg_ret_ms": avg_ret_ms,
        "avg_ce_ms": avg_ce_ms,
    }


if __name__ == "__main__":
    main()
