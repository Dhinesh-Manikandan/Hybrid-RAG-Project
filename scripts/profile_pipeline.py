"""
profile_pipeline.py -- Detailed Stage-by-Stage Latency Profiler for Hybrid-RAG.

Instruments every stage of the retrieval pipeline:
A. Query reformulation
B. Dense Naive retrieval (all variants)
C. Dense Sentence-Window retrieval (all variants)
D. Dense Parent-Child retrieval (all variants)
E. BM25 retrieval (all variants)
F. Canonical RRF fusion
G. Candidate-pool construction & MMR diversification
H. CrossEncoder inference
I. Final aggregation & sorting
J. Total end-to-end latency

Runs across the frozen benchmark queries to compute:
- Mean
- Median
- P95
- Percentage contribution of each stage to total latency
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from retrieve import HybridRAG
from src.retrieval.dense import retrieve_dense
from src.retrieval.query_reformulation import reformulate_query
from src.fusion.rrf import canonical_rrf, mmr_diversify
from src.reranking.cross_encoder import get_reranker
from src.embeddings.encoder import encode_query
from src.config import (
    RETRIEVAL_TOP_K, FUSION_TOP_K, FINAL_TOP_K,
    RRF_K, MMR_LAMBDA, NUM_REFORMULATIONS,
    ENABLE_QUERY_REFORMULATION, ENABLE_MULTI_QUERY_RERANKING
)


def profile_pipeline(queries_to_run: int = 82):
    benchmark_file = PROJECT_ROOT / "evaluation" / "multi_document_queries.json"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if queries_to_run < len(queries):
        queries = queries[:queries_to_run]

    print(f"\nInitializing Hybrid-RAG engine for profiling...")
    rag = HybridRAG()

    # Pre-warm models to avoid cold-start bias
    _ = rag.query("Warmup query for models")

    timings = {
        "A_reformulation": [],
        "B_dense_naive": [],
        "C_dense_sw": [],
        "D_dense_pc": [],
        "E_bm25": [],
        "F_rrf_fusion": [],
        "G_mmr_diversify": [],
        "H_crossencoder_inference": [],
        "I_final_aggregation": [],
        "J_total_latency": []
    }

    print(f"\nProfiling {len(queries)} queries...")

    for i, item in enumerate(queries, 1):
        q = item["query"]
        t_total_start = time.perf_counter()

        # A. Query Reformulation
        t0 = time.perf_counter()
        if ENABLE_QUERY_REFORMULATION:
            variants = reformulate_query(q, num_reformulations=NUM_REFORMULATIONS)
        else:
            variants = [q]
        t_reform = time.perf_counter() - t0
        timings["A_reformulation"].append(t_reform)

        # B, C, D, E: Dense and BM25 across variants
        t_naive = 0.0
        t_sw = 0.0
        t_pc = 0.0
        t_bm25 = 0.0
        all_lists = []

        for q_idx, q_str in enumerate(variants):
            tag = "original" if q_idx == 0 else f"reformulated_{q_idx}"

            # B. Dense Naive
            t_b0 = time.perf_counter()
            naive_res = retrieve_dense(q_str, rag.naive_index, rag.naive_meta, rag.embedding_model, k=RETRIEVAL_TOP_K)
            t_naive += (time.perf_counter() - t_b0)

            # C. Dense SW
            t_c0 = time.perf_counter()
            sw_res = retrieve_dense(q_str, rag.sw_index, rag.sw_meta, rag.embedding_model, k=RETRIEVAL_TOP_K)
            t_sw += (time.perf_counter() - t_c0)

            # D. Dense PC
            t_d0 = time.perf_counter()
            pc_res = retrieve_dense(q_str, rag.pc_index, rag.pc_meta, rag.embedding_model, k=RETRIEVAL_TOP_K)
            t_pc += (time.perf_counter() - t_d0)

            for res_list in [naive_res, sw_res, pc_res]:
                for item_res in res_list:
                    item_res["query_variant"] = tag
                    item_res["query_text"] = q_str
            all_lists.extend([naive_res, sw_res, pc_res])

            # E. BM25
            t_e0 = time.perf_counter()
            bm25_res = rag.bm25.retrieve(q_str, k=RETRIEVAL_TOP_K)
            t_bm25 += (time.perf_counter() - t_e0)
            for item_res in bm25_res:
                item_res["query_variant"] = tag
                item_res["query_text"] = q_str
            all_lists.append(bm25_res)

        timings["B_dense_naive"].append(t_naive)
        timings["C_dense_sw"].append(t_sw)
        timings["D_dense_pc"].append(t_pc)
        timings["E_bm25"].append(t_bm25)

        # F. Canonical RRF
        t_f0 = time.perf_counter()
        fused = canonical_rrf(all_lists, k=RRF_K)
        fused = fused[:FUSION_TOP_K]
        t_rrf = time.perf_counter() - t_f0
        timings["F_rrf_fusion"].append(t_rrf)

        # G. Candidate Pool / MMR
        t_g0 = time.perf_counter()
        diverse = fused
        if len(fused) > FINAL_TOP_K:
            query_embedding = encode_query(q, rag.embedding_model)
            diverse = mmr_diversify(
                candidates=fused,
                query_embedding=query_embedding,
                model_name=rag.embedding_model,
                top_k=FUSION_TOP_K,
                lambda_param=MMR_LAMBDA,
            )
        t_mmr = time.perf_counter() - t_g0
        timings["G_mmr_diversify"].append(t_mmr)

        # H. CrossEncoder Inference
        # Instrument exact CrossEncoder predict time
        reranker = get_reranker(rag.reranker_model)
        candidate_texts = [c.get("retrieval_text", c["text"]) for c in diverse]
        variants_for_reranking = variants if ENABLE_MULTI_QUERY_RERANKING else [q]
        all_pairs = [(v, text) for v in variants_for_reranking for text in candidate_texts]

        t_h0 = time.perf_counter()
        raw_scores = reranker.predict(all_pairs)
        t_ce_inference = time.perf_counter() - t_h0
        timings["H_crossencoder_inference"].append(t_ce_inference)

        # I. Final Aggregation & Sorting
        t_i0 = time.perf_counter()
        num_candidates = len(diverse)
        num_variants = len(variants_for_reranking)
        reranked = []
        for c_idx, candidate in enumerate(diverse):
            candidate_variant_scores = [
                float(raw_scores[v * num_candidates + c_idx])
                for v in range(num_variants)
            ]
            mean_score = sum(candidate_variant_scores) / num_variants
            res_item = candidate.copy()
            res_item["rerank_score"] = float(mean_score)
            reranked.append(res_item)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_results = reranked[:FINAL_TOP_K]
        t_final_agg = time.perf_counter() - t_i0
        timings["I_final_aggregation"].append(t_final_agg)

        # J. Total
        t_total = time.perf_counter() - t_total_start
        timings["J_total_latency"].append(t_total)

        if i % 10 == 0 or i == len(queries):
            print(f"  [{i:02d}/{len(queries)}] Last: {t_total:.2f}s | CE Inf: {t_ce_inference:.2f}s | Dense(all): {(t_naive+t_sw+t_pc):.2f}s")

    # ============================================================
    # SUMMARY STATS
    # ============================================================
    total_mean = float(np.mean(timings["J_total_latency"]))

    print("\n" + "=" * 85)
    print("DETAILED RETRIEVAL STAGE LATENCY PROFILING RESULTS")
    print("=" * 85)
    header = f"{'Stage':<35} | {'Mean (s)':<10} | {'Median (s)':<10} | {'P95 (s)':<10} | {'% of Total':<10}"
    print(header)
    print("-" * len(header))

    stage_labels = [
        ("A. Query Reformulation (T5/ling)", "A_reformulation"),
        ("B. Dense Naive Retrieval", "B_dense_naive"),
        ("C. Dense Sentence-Window Retrieval", "C_dense_sw"),
        ("D. Dense Parent-Child Retrieval", "D_dense_pc"),
        ("E. BM25 Lexical Retrieval", "E_bm25"),
        ("F. Canonical RRF Fusion", "F_rrf_fusion"),
        ("G. Candidate Pool / MMR Diversify", "G_mmr_diversify"),
        ("H. CrossEncoder Inference", "H_crossencoder_inference"),
        ("I. Final Aggregation & Sorting", "I_final_aggregation"),
        ("J. TOTAL END-TO-END", "J_total_latency"),
    ]

    profile_summary = {}

    for label, key in stage_labels:
        arr = timings[key]
        m = float(np.mean(arr))
        med = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        pct = (m / total_mean) * 100.0 if key != "J_total_latency" else 100.0

        profile_summary[key] = {
            "label": label,
            "mean": m,
            "median": med,
            "p95": p95,
            "percent_of_total": pct
        }

        print(f"{label:<35} | {m:<10.4f} | {med:<10.4f} | {p95:<10.4f} | {pct:<9.1f}%")

    print("-" * len(header))

    # Dense total
    dense_total_mean = profile_summary["B_dense_naive"]["mean"] + profile_summary["C_dense_sw"]["mean"] + profile_summary["D_dense_pc"]["mean"]
    print(f"{'Combined Dense Retrieval (B+C+D)':<35} | {dense_total_mean:<10.4f} | {'-':<10} | {'-':<10} | {(dense_total_mean/total_mean)*100:<9.1f}%")

    # Save profiler output to JSON
    profile_out_path = PROJECT_ROOT / "evaluation" / "results" / "profiling_baseline.json"
    with open(profile_out_path, "w", encoding="utf-8") as pf:
        json.dump(profile_summary, pf, indent=2)
    print(f"\nProfiling results saved to: {profile_out_path}")

    return profile_summary


if __name__ == "__main__":
    profile_pipeline(82)
