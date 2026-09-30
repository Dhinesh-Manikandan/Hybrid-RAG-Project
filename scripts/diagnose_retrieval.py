"""
diagnose_retrieval.py -- Comprehensive Diagnostic Utility for Hybrid-RAG Pipeline.

Traces a query stage-by-stage through:
1. Dense Retrieval (Naive, Sentence-Window, Parent-Child)
2. BM25 Lexical Retrieval
3. Canonical RRF Fusion (with per-source rank tracking)
4. MMR Diversification
5. CrossEncoder Reranking (Before vs After comparison)
6. Representation Inspection (Embedding vs BM25 vs Reranker text)
7. Controlled Representation Experiment (Rep A vs Rep B vs Rep C)
8. Stage-by-Stage Diagnosis

Usage:
    python scripts/diagnose_retrieval.py
    python scripts/diagnose_retrieval.py --query "Who is eligible for PMAY-U?"
    python scripts/diagnose_retrieval.py --multi
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K, FUSION_TOP_K, FINAL_TOP_K,
    RRF_K, MMR_LAMBDA,
)
from src.retrieval.dense import retrieve_dense
from src.fusion.rrf import canonical_rrf, mmr_diversify
from src.reranking.cross_encoder import rerank
from src.embeddings.encoder import encode_query
from retrieve import HybridRAG


def _format_snippet(text: str, max_chars: int = 150) -> str:
    """Format snippet cleanly on one line."""
    clean = " ".join(text.replace("\n", " ").split())
    if len(clean) > max_chars:
        return clean[:max_chars] + "..."
    return clean


def get_source_ranks(canonical_id: str, lists_with_names: list[tuple[str, list[dict]]]) -> dict[str, int]:
    """Find the 1-based rank of canonical_id in each retrieval list."""
    contributions = {}
    for name, r_list in lists_with_names:
        for rank, item in enumerate(r_list, start=1):
            item_canon = f"{item.get('document_id')}::{item.get('source_chunk_id')}"
            if item_canon == canonical_id:
                contributions[name] = rank
                break
    return contributions


def run_diagnostic(rag: HybridRAG, query: str, top_k: int = 10, fusion_k: int = 15, final_k: int = 5) -> dict:
    """Run full diagnostic trace for a single query."""
    print("\n" + "=" * 80)
    print(f"QUERY: \"{query}\"")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. DENSE RETRIEVAL
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. DENSE RETRIEVAL (BEFORE FUSION)")
    print("=" * 80)

    naive_res = retrieve_dense(query, rag.naive_index, rag.naive_meta, rag.embedding_model, k=top_k)
    sw_res    = retrieve_dense(query, rag.sw_index, rag.sw_meta, rag.embedding_model, k=top_k)
    pc_res    = retrieve_dense(query, rag.pc_index, rag.pc_meta, rag.embedding_model, k=top_k)

    dense_suites = [
        ("Naive RAG", naive_res),
        ("Sentence-Window", sw_res),
        ("Parent-Child", pc_res),
    ]

    for suite_name, results in dense_suites:
        print(f"\n--- Strategy: {suite_name} (Top {len(results)}) ---")
        print(f"{'Rank':<5} | {'Score':<8} | {'Chunk ID':<30} | {'Section':<25} | {'Heading Path'}")
        print("-" * 100)
        for r in results:
            rank = r.get("rank", "-")
            score = f"{r.get('retrieval_score', 0.0):.4f}"
            cid = r.get("chunk_id", "-")
            sec = r.get("section", "-")[:24]
            hpath = str(r.get("heading_path", []))
            print(f"{rank:<5} | {score:<8} | {cid:<30} | {sec:<25} | {hpath}")
            print(f"      Text: {_format_snippet(r.get('text', ''))}")

    # ---------------------------------------------------------
    # 2. BM25 RETRIEVAL
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. BM25 RETRIEVAL (BEFORE FUSION)")
    print("=" * 80)

    bm25_res = rag.bm25.retrieve(query, k=top_k)
    print(f"{'Rank':<5} | {'BM25 Score':<10} | {'Chunk ID':<30} | {'Section':<25} | {'Heading Path'}")
    print("-" * 100)
    for r in bm25_res:
        rank = r.get("rank", "-")
        score = f"{r.get('retrieval_score', 0.0):.4f}"
        cid = r.get("chunk_id", "-")
        sec = r.get("section", "-")[:24]
        hpath = str(r.get("heading_path", []))
        print(f"{rank:<5} | {score:<10} | {cid:<30} | {sec:<25} | {hpath}")
        print(f"      Text: {_format_snippet(r.get('text', ''))}")

    # ---------------------------------------------------------
    # 3. RRF FUSION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. CANONICAL RRF FUSION")
    print("=" * 80)

    lists_with_names = [
        ("naive", naive_res),
        ("sentence_window", sw_res),
        ("parent_child", pc_res),
        ("bm25", bm25_res),
    ]
    all_lists = [l for _, l in lists_with_names]

    fused = canonical_rrf(all_lists, k=RRF_K)[:fusion_k]

    print(f"{'Rank':<5} | {'RRF Score':<10} | {'Canonical ID':<35} | {'Section':<25} | {'Contributing Sources (Ranks)'}")
    print("-" * 115)
    for rank, item in enumerate(fused, start=1):
        canon_id = item.get("canonical_id", "-")
        score = f"{item.get('rrf_score', 0.0):.6f}"
        sec = item.get("section", "-")[:24]
        contribs = get_source_ranks(canon_id, lists_with_names)
        contrib_str = ", ".join(f"{src}: #{rk}" for src, rk in contribs.items())
        print(f"{rank:<5} | {score:<10} | {canon_id:<35} | {sec:<25} | {contrib_str}")
        print(f"      Text: {_format_snippet(item.get('text', ''))}")

    # MMR Diversification
    query_embedding = encode_query(query, rag.embedding_model)
    diverse = mmr_diversify(
        candidates=fused,
        query_embedding=query_embedding,
        model_name=rag.embedding_model,
        top_k=fusion_k,
        lambda_param=MMR_LAMBDA,
    )

    # ---------------------------------------------------------
    # 4. CROSS-ENCODER BEFORE/AFTER
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. CROSS-ENCODER RERANKING (BEFORE vs AFTER)")
    print("=" * 80)

    print("\n[BEFORE RERANKING: Candidate Pool passed to CrossEncoder]")
    for rrf_rank, item in enumerate(diverse, start=1):
        item["_rrf_rank"] = rrf_rank
        cid = item.get("chunk_id", "-")
        sec = item.get("section", "-")
        print(f"  RRF #{rrf_rank:<2} -> {cid} | Section: {sec} | Text: {_format_snippet(item.get('text', ''), 80)}")

    reranked = rerank(
        query=query,
        candidates=diverse,
        model_name=rag.reranker_model,
        top_k=final_k,
    )

    print("\n[AFTER RERANKING: Final Results]")
    print(f"{'Final':<6} | {'Orig RRF':<9} | {'Rerank Score':<13} | {'Chunk ID':<30} | {'Section':<25}")
    print("-" * 95)
    for final_rank, item in enumerate(reranked, start=1):
        orig_rrf = item.get("_rrf_rank", "-")
        score = f"{item.get('rerank_score', 0.0):.4f}"
        cid = item.get("chunk_id", "-")
        sec = item.get("section", "-")[:24]
        print(f"{final_rank:<6} | #{orig_rrf:<8} | {score:<13} | {cid:<30} | {sec:<25}")
        print(f"      Text: {_format_snippet(item.get('text', ''))}")

    # ---------------------------------------------------------
    # 5. EXACT TEXT REPRESENTATIONS INSPECTION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. EXACT TEXT REPRESENTATIONS INSPECTION")
    print("=" * 80)

    # Find naive__5 (eligibility) and naive__6 (application process)
    elig_naive = next((c for c in rag.naive_meta if "naive__5" in c.get("chunk_id", "") or "ELIGIBILITY" in c.get("section", "")), None)
    app_naive  = next((c for c in rag.naive_meta if "naive__6" in c.get("chunk_id", "") or "APPLICATION PROCESS" in c.get("section", "")), None)

    for label, chunk in [("ELIGIBILITY CHUNK", elig_naive), ("APPLICATION PROCESS CHUNK", app_naive)]:
        if chunk:
            print(f"\n--- {label} ({chunk.get('chunk_id')}) ---")
            print(f"Section Metadata:      {chunk.get('section')}")
            print(f"Heading Path Metadata: {chunk.get('heading_path')}")
            print(f"Source File Metadata:  {chunk.get('source_file')}")
            print("\nExact Text passed to Models (Dense Embedding, BM25, & CrossEncoder):")
            print("-" * 60)
            print(chunk.get("text", ""))
            print("-" * 60)

    # ---------------------------------------------------------
    # 6. CONTROLLED REPRESENTATION EXPERIMENT
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. CONTROLLED REPRESENTATION EXPERIMENT (DIAGNOSTIC ONLY)")
    print("=" * 80)
    print("Testing CrossEncoder scoring under 3 representations without modifying production:\n")

    if elig_naive and app_naive:
        # Strip existing prefix for Rep A
        def _strip_prefix(text: str) -> str:
            lines = text.split("\n\n")
            if lines and lines[0].startswith("[Section:"):
                return "\n\n".join(lines[1:])
            return text

        elig_bare = _strip_prefix(elig_naive["text"])
        app_bare  = _strip_prefix(app_naive["text"])

        source_file = elig_naive.get("source_file", "document.txt")
        doc_stem = Path(source_file).stem

        experiments = [
            ("Representation A: Bare Chunk Text (No Heading/Doc Context)", {
                "eligibility": elig_bare,
                "application": app_bare,
            }),
            ("Representation B: Current Production ([Section: <heading_path>])", {
                "eligibility": elig_naive["text"],
                "application": app_naive["text"],
            }),
            ("Representation C: Document + Section Context ([Document: <doc> | Section: <heading>])", {
                "eligibility": f"[Document: {doc_stem} | Section: {' > '.join(elig_naive.get('heading_path', ['ELIGIBILITY']))}]\n\n{elig_bare}",
                "application": f"[Document: {doc_stem} | Section: {' > '.join(app_naive.get('heading_path', ['APPLICATION PROCESS']))}]\n\n{app_bare}",
            }),
        ]

        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(rag.reranker_model)

        print(f"{'Experiment':<45} | {'ELIGIBILITY Score':<18} | {'APPLICATION Score':<18} | {'Winner'}")
        print("-" * 95)

        for exp_name, texts in experiments:
            pairs = [
                (query, texts["eligibility"]),
                (query, texts["application"]),
            ]
            scores = cross_encoder.predict(pairs)
            elig_score = float(scores[0])
            app_score  = float(scores[1])
            winner = "ELIGIBILITY" if elig_score > app_score else "APPLICATION PROCESS"
            print(f"{exp_name:<45} | {elig_score:<18.4f} | {app_score:<18.4f} | {winner}")

    # ---------------------------------------------------------
    # 7. DIAGNOSTIC SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. DIAGNOSTIC SUMMARY FOR QUERY: \"" + query + "\"")
    print("=" * 80)

    # Analyze ranks of eligibility
    elig_naive_rank = next((r["rank"] for r in naive_res if "ELIGIBILITY" in r.get("section", "")), None)
    elig_sw_rank    = next((r["rank"] for r in sw_res if "ELIGIBILITY" in r.get("section", "")), None)
    elig_pc_rank    = next((r["rank"] for r in pc_res if "ELIGIBILITY" in r.get("section", "")), None)
    elig_bm25_rank  = next((r["rank"] for r in bm25_res if "ELIGIBILITY" in r.get("section", "")), None)
    elig_rrf_rank   = next((idx + 1 for idx, r in enumerate(fused) if "ELIGIBILITY" in r.get("section", "")), None)
    elig_final_rank = next((idx + 1 for idx, r in enumerate(reranked) if "ELIGIBILITY" in r.get("section", "")), None)

    app_naive_rank = next((r["rank"] for r in naive_res if "APPLICATION PROCESS" in r.get("section", "")), None)
    app_bm25_rank  = next((r["rank"] for r in bm25_res if "APPLICATION PROCESS" in r.get("section", "")), None)
    app_rrf_rank   = next((idx + 1 for idx, r in enumerate(fused) if "APPLICATION PROCESS" in r.get("section", "")), None)
    app_final_rank = next((idx + 1 for idx, r in enumerate(reranked) if "APPLICATION PROCESS" in r.get("section", "")), None)

    print(f"1. Does ELIGIBILITY appear in Dense Naive?       {'Yes, Rank #' + str(elig_naive_rank) if elig_naive_rank else 'No (not in top ' + str(top_k) + ')'}")
    print(f"   Does ELIGIBILITY appear in Sentence Window?   {'Yes, Rank #' + str(elig_sw_rank) if elig_sw_rank else 'No (not in top ' + str(top_k) + ')'}")
    print(f"   Does ELIGIBILITY appear in Parent Child?       {'Yes, Rank #' + str(elig_pc_rank) if elig_pc_rank else 'No (not in top ' + str(top_k) + ')'}")
    print(f"2. Does ELIGIBILITY appear in BM25?              {'Yes, Rank #' + str(elig_bm25_rank) if elig_bm25_rank else 'No (not in top ' + str(top_k) + ')'}")
    print(f"3. Rank in Canonical RRF Fusion:                 {'Rank #' + str(elig_rrf_rank) if elig_rrf_rank else 'Not in fused candidates'}")
    print(f"4. Rank in Final CrossEncoder Reranking:         {'Rank #' + str(elig_final_rank) if elig_final_rank else 'Not in top ' + str(final_k)}")
    print(f"5. APPLICATION PROCESS Ranks:                    Dense: #{app_naive_rank} | BM25: #{app_bm25_rank} | RRF: #{app_rrf_rank} | Final: #{app_final_rank}")

    return {
        "naive_res": naive_res,
        "bm25_res": bm25_res,
        "fused": fused,
        "reranked": reranked,
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid-RAG Retrieval Diagnostic Utility")
    parser.add_argument("--query", type=str, default="Who is eligible for PMAY-U?", help="Query to diagnose")
    parser.add_argument("--multi", action="store_true", help="Run diagnostic across a suite of test queries")
    args = parser.parse_args()

    rag = HybridRAG()

    if args.multi:
        test_queries = [
            "Who is eligible for PMAY-U?",
            "What are the eligibility criteria?",
            "What benefits are available?",
            "How can beneficiaries apply?",
            "What documents are required?",
        ]
        for q in test_queries:
            run_diagnostic(rag, q)
    else:
        run_diagnostic(rag, args.query)


if __name__ == "__main__":
    main()
