"""
run_reformulation_ablation.py -- Comparative Ablation Study of Query Reformulation & Multi-Query Retrieval.

Measures:
1. Ablation A: Query Reformulation DISABLED (Original-only)
2. Ablation B: Query Reformulation ENABLED (Multi-Query Retrieval)
3. Diagnostic trace for "Who is eligible for PMAY-U?":
   - Original query
   - Reformulated queries
   - Per-variant ELIGIBILITY RRF Rank
   - Final results after multi-query fusion
4. Multi-query Hit@1, Hit@3, Hit@5 comparison
5. Latency benchmark:
   - Original-only retrieval time
   - Reformulation time
   - Multi-query retrieval time
   - Total pipeline time
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG
from src.retrieval.query_reformulation import reformulate_query
from src.retrieval.dense import retrieve_dense
from src.fusion.rrf import canonical_rrf

BENCHMARK_QUERIES = [
    {
        "query": "Who is eligible for PMAY-U?",
        "expected_section": "ELIGIBILITY",
    },
    {
        "query": "What are the eligibility criteria?",
        "expected_section": "ELIGIBILITY",
    },
    {
        "query": "What benefits are available?",
        "expected_section": "BENEFITS",
    },
    {
        "query": "How can beneficiaries apply?",
        "expected_section": "APPLICATION PROCESS",
    },
    {
        "query": "What documents are required?",
        "expected_section": "DOCUMENTS REQUIRED",
    },
]

def find_section_rank(finals: list[dict], target_section: str):
    for rank, item in enumerate(finals, 1):
        if target_section.upper() in item.get("section", "").upper():
            return rank, item.get("rerank_score", 0.0)
    return None, None

def run_ablation():
    print("=" * 110)
    print("HYBRID-RAG QUERY REFORMULATION ABLATION EXPERIMENT")
    print("=" * 110)

    rag = HybridRAG()

    # -------------------------------------------------------------
    # 1. DETAILED DIAGNOSTIC FOR: "Who is eligible for PMAY-U?"
    # -------------------------------------------------------------
    target_q = "Who is eligible for PMAY-U?"
    print(f"\n" + "=" * 90)
    print(f"1. DIAGNOSTIC TRACE FOR: \"{target_q}\"")
    print("=" * 90)

    t0_ref = time.perf_counter()
    variants = reformulate_query(target_q, num_reformulations=2)
    t1_ref = time.perf_counter()
    ref_time_ms = (t1_ref - t0_ref) * 1000

    print(f"Original query:\n    \"{variants[0]}\"")
    print("Reformulated queries:")
    for idx, v in enumerate(variants[1:], 1):
        print(f"    {idx}. \"{v}\"")

    # Run each variant individually to get its individual ELIGIBILITY RRF Rank
    print("\nVariant-by-Variant RRF Breakdown:")
    print(f"{'Variant':<22} | {'Query String':<55} | {'ELIGIBILITY RRF Rank':<22}")
    print("-" * 105)

    variant_ranks = []
    for idx, v in enumerate(variants):
        label = "Original" if idx == 0 else f"Reformulation {idx}"
        
        # Single-query retrieval for this variant
        v_naive = retrieve_dense(v, rag.naive_index, rag.naive_meta, rag.embedding_model, k=10)
        v_sw    = retrieve_dense(v, rag.sw_index, rag.sw_meta, rag.embedding_model, k=10)
        v_pc    = retrieve_dense(v, rag.pc_index, rag.pc_meta, rag.embedding_model, k=10)
        v_bm25  = rag.bm25.retrieve(v, k=10)

        v_fused = canonical_rrf([v_naive, v_sw, v_pc, v_bm25], k=60)
        elig_rk = next((r for r, it in enumerate(v_fused, 1) if "ELIGIBILITY" in it.get("section", "").upper()), None)
        elig_disp = f"Rank #{elig_rk}" if elig_rk else "Not in top pool"
        variant_ranks.append((label, elig_disp))
        print(f"{label:<22} | {v:<55} | {elig_disp:<22}")

    # Full Multi-Query Fusion for target query
    mq_res = rag.query(target_q, enable_reformulation=True)
    fused_mq = mq_res["fused_results"]
    finals_mq = mq_res["final_results"]

    mq_elig_rrf = next((r for r, it in enumerate(fused_mq, 1) if "ELIGIBILITY" in it.get("section", "").upper()), None)
    print(f"\nAfter Multi-Query Fusion: ELIGIBILITY RRF Rank = #{mq_elig_rrf}")

    print("\nFinal Top-5 Results after Multi-Query Fusion & CrossEncoder:")
    print(f"{'Rank':<5} | {'Score':<8} | {'Chunk ID':<35} | {'Section':<22} | {'Sources':<25}")
    print("-" * 105)
    for rk, it in enumerate(finals_mq[:5], 1):
        sc = f"{it.get('rerank_score', 0):.4f}"
        cid = it.get("chunk_id", "-")
        sec = it.get("section", "-")
        srcs = str(it.get("retrieval_sources", []))
        print(f"{rk:<5} | {sc:<8} | {cid:<35} | {sec:<22} | {srcs:<25}")

    # -------------------------------------------------------------
    # 2. COMPARATIVE ABLATION (DISABLED vs ENABLED)
    # -------------------------------------------------------------
    print("\n" + "=" * 110)
    print("2. COMPARATIVE ABLATION: REFORMULATION DISABLED vs ENABLED")
    print("=" * 110)

    disabled_results = []
    enabled_results = []

    # ABLATION A: DISABLED
    print("\n--- Running Evaluation: ENABLE_QUERY_REFORMULATION = False ---")
    t0_dis = time.perf_counter()
    for item in BENCHMARK_QUERIES:
        q = item["query"]
        exp = item["expected_section"]
        t_start = time.perf_counter()
        res = rag.query(q, enable_reformulation=False, final_top_k=5)
        t_elapsed = (time.perf_counter() - t_start) * 1000

        finals = res["final_results"]
        sections = [f.get("section", "") for f in finals]
        h1 = int(exp in sections[:1])
        h3 = int(exp in sections[:3])
        h5 = int(exp in sections[:5])
        exp_rk, exp_sc = find_section_rank(finals, exp)
        app_rk, app_sc = find_section_rank(finals, "APPLICATION PROCESS")

        disabled_results.append({
            "query": q,
            "expected": exp,
            "top_section": sections[0] if sections else "None",
            "hit_1": h1, "hit_3": h3, "hit_5": h5,
            "target_rank": exp_rk or ">5",
            "target_score": exp_sc,
            "app_rank": app_rk or ">5",
            "time_ms": t_elapsed,
        })
    total_dis_time = (time.perf_counter() - t0_dis) * 1000

    # ABLATION B: ENABLED
    print("--- Running Evaluation: ENABLE_QUERY_REFORMULATION = True ---")
    t0_en = time.perf_counter()
    for item in BENCHMARK_QUERIES:
        q = item["query"]
        exp = item["expected_section"]
        t_start = time.perf_counter()
        res = rag.query(q, enable_reformulation=True, final_top_k=5)
        t_elapsed = (time.perf_counter() - t_start) * 1000

        finals = res["final_results"]
        sections = [f.get("section", "") for f in finals]
        h1 = int(exp in sections[:1])
        h3 = int(exp in sections[:3])
        h5 = int(exp in sections[:5])
        exp_rk, exp_sc = find_section_rank(finals, exp)
        app_rk, app_sc = find_section_rank(finals, "APPLICATION PROCESS")

        enabled_results.append({
            "query": q,
            "expected": exp,
            "top_section": sections[0] if sections else "None",
            "hit_1": h1, "hit_3": h3, "hit_5": h5,
            "target_rank": exp_rk or ">5",
            "target_score": exp_sc,
            "app_rank": app_rk or ">5",
            "time_ms": t_elapsed,
        })
    total_en_time = (time.perf_counter() - t0_en) * 1000

    # -------------------------------------------------------------
    # 3. COMPARISON TABLES
    # -------------------------------------------------------------
    print("\n" + "=" * 110)
    print("ABLATION RESULTS TABLE: ORIGINAL-ONLY (DISABLED) vs MULTI-QUERY (ENABLED)")
    print("=" * 110)
    print(f"{'Query':<36} | {'Expected':<18} | {'Disabled (Orig)':<16} | {'Enabled (Multi)':<16} | {'Target Rank (Dis -> En)'}")
    print("-" * 110)
    for d, e in zip(disabled_results, enabled_results):
        dis_str = f"H1={d['hit_1']} (Sec: {d['top_section'][:10]})"
        en_str  = f"H1={e['hit_1']} (Sec: {e['top_section'][:10]})"
        rank_change = f"#{d['target_rank']} -> #{e['target_rank']}"
        print(f"{d['query']:<36} | {d['expected']:<18} | {dis_str:<16} | {en_str:<16} | {rank_change}")

    print("\n" + "=" * 90)
    print("SUMMARY METRICS COMPARISON")
    print("=" * 90)
    dis_h1 = sum(d["hit_1"] for d in disabled_results) / len(disabled_results)
    dis_h3 = sum(d["hit_3"] for d in disabled_results) / len(disabled_results)
    dis_h5 = sum(d["hit_5"] for d in disabled_results) / len(disabled_results)

    en_h1 = sum(e["hit_1"] for e in enabled_results) / len(enabled_results)
    en_h3 = sum(e["hit_3"] for e in enabled_results) / len(enabled_results)
    en_h5 = sum(e["hit_5"] for e in enabled_results) / len(enabled_results)

    print(f"Metric   | Original-Only (Disabled) | Multi-Query (Enabled) | Delta")
    print("-" * 65)
    print(f"Hit@1    | {dis_h1:.2f} ({sum(d['hit_1'] for d in disabled_results)}/5)               | {en_h1:.2f} ({sum(e['hit_1'] for e in enabled_results)}/5)             | {en_h1 - dis_h1:+.2f}")
    print(f"Hit@3    | {dis_h3:.2f} ({sum(d['hit_3'] for d in disabled_results)}/5)               | {en_h3:.2f} ({sum(e['hit_3'] for e in enabled_results)}/5)             | {en_h3 - dis_h3:+.2f}")
    print(f"Hit@5    | {dis_h5:.2f} ({sum(d['hit_5'] for d in disabled_results)}/5)               | {en_h5:.2f} ({sum(e['hit_5'] for e in enabled_results)}/5)             | {en_h5 - dis_h5:+.2f}")

    # -------------------------------------------------------------
    # 4. LATENCY BREAKDOWN
    # -------------------------------------------------------------
    avg_dis_lat = total_dis_time / len(disabled_results)
    avg_en_lat  = total_en_time / len(enabled_results)

    print("\n" + "=" * 90)
    print("LATENCY BENCHMARK (Average per query)")
    print("=" * 90)
    print(f"Original-Only Retrieval Time:   {avg_dis_lat:7.2f} ms")
    print(f"Reformulation Generation Time:  {ref_time_ms:7.2f} ms  (Sub-millisecond linguistic engine)")
    print(f"Multi-Query Total Time:         {avg_en_lat:7.2f} ms")
    print(f"Additional Multi-Query Overhead: {avg_en_lat - avg_dis_lat:7.2f} ms")
    print("=" * 90)

if __name__ == "__main__":
    run_ablation()
