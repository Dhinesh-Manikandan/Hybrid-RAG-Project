"""
diagnose_query_formulation.py -- Query-Understanding Diagnostic Experiment.

Measures how generic query reformulations affect stage-by-stage retrieval:
- Dense (Naive, Sentence-Window, Parent-Child)
- BM25
- Canonical RRF Fusion
- CrossEncoder Reranking
for ELIGIBILITY vs APPLICATION PROCESS.

DIAGNOSTIC ONLY -- DOES NOT MODIFY PRODUCTION CODE.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG

def find_rank_and_score(results: list[dict], target_section: str, id_field: str = "chunk_id", score_field: str = "retrieval_score"):
    for idx, item in enumerate(results, start=1):
        sec = item.get("section", "")
        cid = item.get(id_field, "")
        if target_section.upper() in sec.upper():
            return idx, item.get(score_field, 0.0), cid
    return None, None, None

def run_experiment():
    print("Initializing Hybrid-RAG for Query Formulation Diagnostic...")
    rag = HybridRAG()

    queries = [
        ("Original Query", "Who is eligible for PMAY-U?"),
        ("Criteria Formulation", "What are the eligibility criteria for PMAY-U?"),
        ("Conditions/Qualify Formulation", "What conditions must a beneficiary satisfy to qualify for PMAY-U?"),
        ("Requirements Formulation", "What are the qualifying requirements for beneficiaries?"),
        ("Scheme Eligibility Formulation", "What requirements determine eligibility for the scheme?"),
        ("Direct Entity Qualification", "Who qualifies for PMAY-U assistance?"),
    ]

    summary_rows = []

    print("\n" + "=" * 110)
    print("QUERY-UNDERSTANDING DIAGNOSTIC EXPERIMENT RESULTS")
    print("=" * 110)

    for label, q in queries:
        print(f"\n--- Testing Formulation: [{label}] ---")
        print(f"Query: \"{q}\"")
        
        # Run retrieval with full top_k to see all positions
        res = rag.query(q, retrieval_top_k=15, fusion_top_k=15, final_top_k=15)

        # 1. Dense Naive
        d_naive_rk, d_naive_sc, _ = find_rank_and_score(res["naive_results"], "ELIGIBILITY")
        # Dense SW
        d_sw_rk, d_sw_sc, _ = find_rank_and_score(res["sw_results"], "ELIGIBILITY")
        # Dense PC
        d_pc_rk, d_pc_sc, _ = find_rank_and_score(res["pc_results"], "ELIGIBILITY")

        # 2. BM25
        bm25_rk, bm25_sc, _ = find_rank_and_score(res["bm25_results"], "ELIGIBILITY")

        # 3. RRF
        rrf_rk, rrf_sc, _ = find_rank_and_score(res["fused_results"], "ELIGIBILITY", id_field="canonical_id", score_field="rrf_score")
        app_rrf_rk, app_rrf_sc, _ = find_rank_and_score(res["fused_results"], "APPLICATION PROCESS", id_field="canonical_id", score_field="rrf_score")

        # 4. Final CrossEncoder
        final_rk, final_sc, _ = find_rank_and_score(res["final_results"], "ELIGIBILITY", score_field="rerank_score")
        app_final_rk, app_final_sc, _ = find_rank_and_score(res["final_results"], "APPLICATION PROCESS", score_field="rerank_score")

        top_3 = res["final_results"][:3]
        top_3_str = ", ".join(f"#{i} {it.get('section')} ({it.get('rerank_score', 0):.2f})" for i, it in enumerate(top_3, 1))

        print(f"  Dense Naive:     Rank #{d_naive_rk} (Score: {d_naive_sc:.4f})" if d_naive_rk else "  Dense Naive:     Not in top 15")
        print(f"  Dense SW:        Rank #{d_sw_rk} (Score: {d_sw_sc:.4f})" if d_sw_rk else "  Dense SW:        Not in top 15")
        print(f"  Dense PC:        Rank #{d_pc_rk} (Score: {d_pc_sc:.4f})" if d_pc_rk else "  Dense PC:        Not in top 15")
        print(f"  BM25:            Rank #{bm25_rk} (Score: {bm25_sc:.4f})" if bm25_rk else "  BM25:            Not in top 15")
        print(f"  Canonical RRF:   ELIGIBILITY Rank #{rrf_rk} | APPLICATION PROCESS Rank #{app_rrf_rk}")
        print(f"  CrossEncoder:    ELIGIBILITY Rank #{final_rk} ({final_sc:.4f}) | APPLICATION PROCESS Rank #{app_final_rk} ({app_final_sc:.4f})")
        print(f"  Final Top-3:     {top_3_str}")

        summary_rows.append({
            "label": label,
            "query": q,
            "naive_rk": d_naive_rk or ">15",
            "pc_rk": d_pc_rk or ">15",
            "bm25_rk": bm25_rk or ">15",
            "rrf_rk": rrf_rk or ">15",
            "final_rk": final_rk or ">15",
            "final_sc": f"{final_sc:.2f}" if final_sc is not None else "N/A",
            "app_final_rk": app_final_rk or ">15",
            "app_final_sc": f"{app_final_sc:.2f}" if app_final_sc is not None else "N/A",
            "winner": "ELIGIBILITY" if (final_rk and (not app_final_rk or final_rk < app_final_rk)) else "APPLICATION PROCESS",
            "top_3": top_3_str,
        })

    print("\n" + "=" * 125)
    print("COMPARISON SUMMARY TABLE")
    print("=" * 125)
    print(f"{'Query Formulation':<35} | {'PC Rank':<7} | {'BM25':<5} | {'RRF Rank':<8} | {'Elig Final':<10} | {'App Final':<9} | {'Winner':<20}")
    print("-" * 125)
    for row in summary_rows:
        q_disp = row["query"] if len(row["query"]) <= 33 else row["query"][:30] + "..."
        elig_str = f"#{row['final_rk']} ({row['final_sc']})"
        app_str = f"#{row['app_final_rk']} ({row['app_final_sc']})"
        print(f"{q_disp:<35} | #{row['pc_rk']:<6} | #{row['bm25_rk']:<4} | #{row['rrf_rk']:<7} | {elig_str:<10} | {app_str:<9} | {row['winner']:<20}")
    print("=" * 125)

if __name__ == "__main__":
    run_experiment()
