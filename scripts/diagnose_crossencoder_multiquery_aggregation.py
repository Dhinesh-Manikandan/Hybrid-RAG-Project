"""
diagnose_crossencoder_multiquery_aggregation.py -- Controlled CrossEncoder Multi-Query Aggregation Diagnostic.

Deep controlled diagnostic evaluating multi-perspective CrossEncoder evidence aggregation.
Compares:
A. Original query only (baseline)
B. Reformulation 1 only
C. Reformulation 2 only
D. Mean CrossEncoder score across all query variants
E. Max CrossEncoder score across query variants
F. Weighted mean: original=0.5, ref1=0.25, ref2=0.25
G. Mean rank across query variants
H. Best rank across query variants
I. Reciprocal-rank aggregation across query variants (k=60)

CRITICAL EXPERIMENTAL CONTROLS:
1. Candidate pool is retrieved ONCE per query using the existing HybridRAG pipeline.
2. The exact same candidate pool is evaluated across all aggregation strategies.
3. Both score-based and rank-based aggregations are evaluated to account for calibration variances.
4. Full 5-query benchmark evaluation measuring Hit@1, Hit@3, Hit@5, MRR, and regressions.
5. Query variant quality and latency overhead are profiled.
6. Strictly document-agnostic; ZERO section-specific heuristics or PMAY rules.

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
from src.retrieval.query_reformulation import reformulate_query
from src.config import NUM_REFORMULATIONS


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


def evaluate_query_crossencoder_multiquery(
    rag: HybridRAG,
    reranker: Any,
    query: str,
    target_section: str | None = None,
    k_rrf: int = 60,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run controlled CrossEncoder multi-query aggregation diagnostic for a single query.
    Retrieves the candidate pool ONCE and applies all 9 aggregation strategies to that pool.
    """
    t_start = time.perf_counter()

    # 1. Retrieve candidate pool ONCE using existing pipeline
    t_ret_start = time.perf_counter()
    res = rag.query(query)
    t_ret_end = time.perf_counter()
    retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0

    query_variants = res["query_variants"]
    candidate_pool = res["diverse_results"]
    n_candidates = len(candidate_pool)

    # 2. Extract texts for reranking (canonical retrieval_text or fallback to text)
    candidate_texts = [c.get("retrieval_text", c["text"]) for c in candidate_pool]

    # 3. Compute CrossEncoder scores for each query variant separately
    variant_scores: list[list[float]] = []
    variant_latencies_ms: list[float] = []

    for q_idx, q_var in enumerate(query_variants):
        t_v_start = time.perf_counter()
        pairs = [(q_var, text) for text in candidate_texts]
        raw_scores = [float(s) for s in reranker.predict(pairs)]
        t_v_end = time.perf_counter()
        variant_scores.append(raw_scores)
        variant_latencies_ms.append((t_v_end - t_v_start) * 1000.0)

    # Ensure variants (Original, Ref1, Ref2) are accessible
    orig_scores = variant_scores[0]
    ref1_scores = variant_scores[1] if len(variant_scores) > 1 else orig_scores
    ref2_scores = variant_scores[2] if len(variant_scores) > 2 else orig_scores

    # 4. Compute per-variant ranks (1-based: 1 = highest CrossEncoder score)
    orig_ranks = compute_ranks(orig_scores, reverse=True)
    ref1_ranks = compute_ranks(ref1_scores, reverse=True)
    ref2_ranks = compute_ranks(ref2_scores, reverse=True)

    # 5. Compute Score-based Aggregations
    mean_scores: list[float] = []
    max_scores: list[float] = []
    weighted_scores: list[float] = []  # 0.50 * orig + 0.25 * ref1 + 0.25 * ref2

    for i in range(n_candidates):
        s_orig = orig_scores[i]
        s_ref1 = ref1_scores[i]
        s_ref2 = ref2_scores[i]
        all_s = [variant_scores[v][i] for v in range(len(query_variants))]

        mean_scores.append(sum(all_s) / len(all_s))
        max_scores.append(max(all_s))
        weighted_scores.append(0.50 * s_orig + 0.25 * s_ref1 + 0.25 * s_ref2)

    # 6. Compute Rank-based Aggregations
    mean_ranks: list[float] = []
    best_ranks: list[int] = []
    reciprocal_ranks: list[float] = []  # sum(1 / (k + rank_v))

    for i in range(n_candidates):
        r_list = [orig_ranks[i], ref1_ranks[i], ref2_ranks[i]]
        mean_ranks.append(sum(r_list) / len(r_list))
        best_ranks.append(min(r_list))
        reciprocal_ranks.append(sum(1.0 / (k_rrf + r) for r in r_list))

    # 7. Package candidate metadata and scores
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
            "raw_rrf_score": c.get("rrf_score", 0.0),
            # Per-variant scores
            "score_orig": orig_scores[i],
            "score_ref1": ref1_scores[i],
            "score_ref2": ref2_scores[i],
            # Per-variant ranks
            "rank_orig": orig_ranks[i],
            "rank_ref1": ref1_ranks[i],
            "rank_ref2": ref2_ranks[i],
            # Score aggregations
            "score_mean": mean_scores[i],
            "score_max": max_scores[i],
            "score_weighted": weighted_scores[i],
            # Rank aggregations
            "rank_mean": mean_ranks[i],
            "rank_best": best_ranks[i],
            "score_rrf_rank": reciprocal_ranks[i],
        }
        annotated_candidates.append(item)

    # 8. Sort candidates for all 9 strategies
    # Tiebreaking: in rank aggregations (where lower is better), tiebreak with higher score_orig
    strategies: dict[str, list[dict]] = {
        "A. Original Query (Baseline)": sorted(annotated_candidates, key=lambda x: x["score_orig"], reverse=True),
        "B. Reformulation 1":           sorted(annotated_candidates, key=lambda x: x["score_ref1"], reverse=True),
        "C. Reformulation 2":           sorted(annotated_candidates, key=lambda x: x["score_ref2"], reverse=True),
        "D. Mean CrossEncoder Score":   sorted(annotated_candidates, key=lambda x: x["score_mean"], reverse=True),
        "E. Max CrossEncoder Score":    sorted(annotated_candidates, key=lambda x: x["score_max"], reverse=True),
        "F. Weighted Mean (0.5/0.25/0.25)": sorted(annotated_candidates, key=lambda x: x["score_weighted"], reverse=True),
        "G. Mean Rank":                 sorted(annotated_candidates, key=lambda x: (x["rank_mean"], -x["score_orig"])),
        "H. Best Rank":                 sorted(annotated_candidates, key=lambda x: (x["rank_best"], -x["score_orig"])),
        "I. Reciprocal Rank (RRF k=60)":sorted(annotated_candidates, key=lambda x: x["score_rrf_rank"], reverse=True),
    }

    total_ce_ms = sum(variant_latencies_ms)
    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    if verbose:
        print("\n" + "=" * 95)
        print(f"DIAGNOSTIC QUERY: \"{query}\"")
        if target_section:
            print(f"Target Section  : {target_section}")
        print("Query Variants Generated:")
        for idx, q_var in enumerate(query_variants):
            lbl = "Original" if idx == 0 else f"Reformulation {idx}"
            lat = f"{variant_latencies_ms[idx]:.1f}ms" if idx < len(variant_latencies_ms) else "N/A"
            print(f"  [{lbl}]: \"{q_var}\" (CE inference: {lat})")

        print(f"\nShared Candidate Pool Size: {n_candidates} candidates (retrieved ONCE)")
        print(f"Timing: Retrieval = {retrieval_latency_ms:.1f}ms | Single CE = {variant_latencies_ms[0]:.1f}ms | Multi-Query CE = {total_ce_ms:.1f}ms")
        print("=" * 95)

    return {
        "query": query,
        "target_section": target_section,
        "query_variants": query_variants,
        "candidate_pool": candidate_pool,
        "annotated_candidates": annotated_candidates,
        "strategies": strategies,
        "retrieval_latency_ms": retrieval_latency_ms,
        "single_ce_latency_ms": variant_latencies_ms[0],
        "multi_ce_latency_ms": total_ce_ms,
        "variant_latencies_ms": variant_latencies_ms,
        "total_latency_ms": total_latency_ms,
    }


def main():
    print("=" * 95)
    print("CONTROLLED CROSSENCODER MULTI-QUERY AGGREGATION DIAGNOSTIC")
    print("=" * 95)

    rag = HybridRAG()
    reranker = get_reranker(rag.reranker_model)

    # -------------------------------------------------------------------------
    # PART 1: Problematic Query Analysis ("Who is eligible for PMAY-U?")
    # -------------------------------------------------------------------------
    main_query = "Who is eligible for PMAY-U?"
    main_target = "ELIGIBILITY"
    res_main = evaluate_query_crossencoder_multiquery(
        rag=rag,
        reranker=reranker,
        query=main_query,
        target_section=main_target,
        verbose=True,
    )

    strategies_main = res_main["strategies"]
    annotated_main = res_main["annotated_candidates"]

    # Mapping score/rank property per strategy
    strat_score_prop = {
        "A. Original Query (Baseline)": "score_orig",
        "B. Reformulation 1":           "score_ref1",
        "C. Reformulation 2":           "score_ref2",
        "D. Mean CrossEncoder Score":   "score_mean",
        "E. Max CrossEncoder Score":    "score_max",
        "F. Weighted Mean (0.5/0.25/0.25)": "score_weighted",
        "G. Mean Rank":                 "rank_mean",
        "H. Best Rank":                 "rank_best",
        "I. Reciprocal Rank (RRF k=60)":"score_rrf_rank",
    }

    # Locate key candidates for tracking
    elig_item = next((c for c in annotated_main if c["section"] == "ELIGIBILITY"), None)
    app_item  = next((c for c in annotated_main if c["section"] == "APPLICATION PROCESS"), None)

    print("\n" + "=" * 95)
    print("PROBLEMATIC QUERY: STRATEGY COMPARISON TABLE")
    print("Query: \"Who is eligible for PMAY-U?\"")
    print("=" * 95)
    print(f"{'Strategy':<33} | {'App Rk':<6} | {'Elig Rk':<7} | {'Elig Score':<11} | {'App Score':<11} | {'Top-1?':<6} | {'Top-3?':<6} | {'Top-5?'}")
    print("-" * 105)

    prob_query_table = []
    for strat_name, ranked_list in strategies_main.items():
        prop = strat_score_prop[strat_name]

        # 1-based ranks
        app_rank = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "APPLICATION PROCESS"), 999)
        elig_rank = next((idx for idx, c in enumerate(ranked_list, start=1) if c["section"] == "ELIGIBILITY"), 999)

        elig_s_val = elig_item[prop] if elig_item else 0.0
        app_s_val  = app_item[prop] if app_item else 0.0

        if "Rank" in strat_name and strat_name != "I. Reciprocal Rank (RRF k=60)":
            elig_s_str = f"{elig_s_val:.2f}"
            app_s_str  = f"{app_s_val:.2f}"
        elif strat_name == "I. Reciprocal Rank (RRF k=60)":
            elig_s_str = f"{elig_s_val:.5f}"
            app_s_str  = f"{app_s_val:.5f}"
        else:
            elig_s_str = f"{elig_s_val:+.4f}"
            app_s_str  = f"{app_s_val:+.4f}"

        top1_str = "YES" if elig_rank == 1 else "NO"
        top3_str = "YES" if elig_rank <= 3 else "NO"
        top5_str = "YES" if elig_rank <= 5 else "NO"

        app_str = f"#{app_rank}" if app_rank != 999 else "N/A"
        elig_str = f"#{elig_rank}" if elig_rank != 999 else "N/A"

        print(f"{strat_name:<33} | {app_str:<6} | {elig_str:<7} | {elig_s_str:<11} | {app_s_str:<11} | {top1_str:<6} | {top3_str:<6} | {top5_str}")
        prob_query_table.append({
            "strategy": strat_name,
            "app_rank": app_rank,
            "elig_rank": elig_rank,
            "elig_score": elig_s_str,
            "app_score": app_s_str,
            "top1": top1_str,
            "top3": top3_str,
            "top5": top5_str,
        })

    # Print Full Top-10 Ranking for ALL Strategies on Problematic Query
    print("\n" + "=" * 95)
    print("TOP-10 RANKINGS ACROSS ALL STRATEGIES (Problematic Query)")
    print("=" * 95)
    for strat_name, ranked_list in strategies_main.items():
        prop = strat_score_prop[strat_name]
        print(f"\n--- Strategy: {strat_name} ---")
        print(f"{'Rk':<3} | {'Section':<25} | {'Source Chunk ID':<28} | {'Strategy Score':<15} | {'Orig Rk':<7} | {'Ref1 Rk':<7} | {'Ref2 Rk':<7}")
        print("-" * 105)
        for r_idx, c in enumerate(ranked_list[:10], start=1):
            sec = c["section"][:24]
            scid = c["source_chunk_id"][:27]
            val = c[prop]
            if "Rank" in strat_name and strat_name != "I. Reciprocal Rank (RRF k=60)":
                val_str = f"{val:.2f}"
            elif strat_name == "I. Reciprocal Rank (RRF k=60)":
                val_str = f"{val:.6f}"
            else:
                val_str = f"{val:+.4f}"

            print(f"{r_idx:<3} | {sec:<25} | {scid:<28} | {val_str:<15} | #{c['rank_orig']:<6} | #{c['rank_ref1']:<6} | #{c['rank_ref2']:<6}")

    # -------------------------------------------------------------------------
    # PART 2: Full 5-Query Benchmark Evaluation Suite
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

    per_query_ranks = []
    strat_metrics = {
        s: {"hit1": 0, "hit3": 0, "hit5": 0, "rr_sum": 0.0, "ranks": []}
        for s in all_strat_names
    }

    all_query_variants_log = []
    total_single_ce_ms = 0.0
    total_multi_ce_ms = 0.0

    for b_item in benchmark_queries:
        q = b_item["query"]
        tgt = b_item["target"]

        q_res = evaluate_query_crossencoder_multiquery(
            rag=rag,
            reranker=reranker,
            query=q,
            target_section=tgt,
            verbose=False,
        )

        all_query_variants_log.append({
            "query": q,
            "target": tgt,
            "variants": q_res["query_variants"],
        })

        total_single_ce_ms += q_res["single_ce_latency_ms"]
        total_multi_ce_ms += q_res["multi_ce_latency_ms"]

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
    print("-" * 135)
    header = f"{'Query':<36} | {'Target':<18} | " + " | ".join([f"{s[:10]:<10}" for s in all_strat_names])
    print(header)
    print("-" * 135)

    for row in per_query_ranks:
        q_short = row["query"][:35]
        tgt = row["target"][:17]
        r_strs = [f"#{row['ranks'][s]}" if row['ranks'][s] != 999 else "N/A" for s in all_strat_names]
        r_formatted = " | ".join([f"{r:<10}" for r in r_strs])
        print(f"{q_short:<36} | {tgt:<18} | {r_formatted}")

    # Print Benchmark Metrics Summary Table
    print("\n" + "=" * 95)
    print("BENCHMARK METRICS SUMMARY (5 Queries)")
    print("=" * 95)
    print(f"{'Strategy':<35} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Regressions vs Baseline'}")
    print("-" * 105)

    base_strat = "A. Original Query (Baseline)"
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
        print(f"{s_name:<35} | {h1:<8.2f} | {h3:<8.2f} | {h5:<8.2f} | {mrr:<8.4f} | {reg_str}")

        summary_results.append({
            "strategy": s_name,
            "hit1": h1,
            "hit3": h3,
            "hit5": h5,
            "mrr": mrr,
            "regressions": reg_str,
            "ranks": curr_ranks,
        })

    # -------------------------------------------------------------------------
    # PART 3: Additional Diagnostics (Consistency & Semantic Quality)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("ADDITIONAL DIAGNOSTIC: QUERY REFORMULATION QUALITY & CONSISTENCY")
    print("=" * 95)

    for item in all_query_variants_log:
        print(f"\nQuery: \"{item['query']}\" (Target: {item['target']})")
        for v_idx, v in enumerate(item["variants"]):
            lbl = "Original" if v_idx == 0 else f"Reformulation {v_idx}"
            print(f"  [{lbl}]: {v}")

    # Latency Profiling Summary
    avg_single_ce = total_single_ce_ms / num_queries
    avg_multi_ce = total_multi_ce_ms / num_queries
    overhead_ms = avg_multi_ce - avg_single_ce
    overhead_pct = (overhead_ms / avg_single_ce) * 100.0 if avg_single_ce > 0 else 0.0

    print("\n" + "=" * 95)
    print("LATENCY PROFILING ANALYSIS (Average across 5 Benchmark Queries)")
    print("=" * 95)
    print(f"Single Query CrossEncoder Latency (Baseline) : {avg_single_ce:.1f} ms")
    print(f"Multi-Query CrossEncoder Latency (3 variants) : {avg_multi_ce:.1f} ms")
    print(f"Additional Latency Per Query Variant         : {avg_single_ce:.1f} ms")
    print(f"Total Reranking Latency Overhead             : +{overhead_ms:.1f} ms (+{overhead_pct:.1f}%)")

    return {
        "prob_query_res": res_main,
        "prob_query_table": prob_query_table,
        "per_query_ranks": per_query_ranks,
        "summary_results": summary_results,
        "all_query_variants_log": all_query_variants_log,
        "avg_single_ce": avg_single_ce,
        "avg_multi_ce": avg_multi_ce,
        "overhead_ms": overhead_ms,
    }


if __name__ == "__main__":
    main()
