"""
retrieve.py -- Complete Hybrid-RAG retrieval pipeline.

Usage:
    from retrieve import HybridRAG
    rag = HybridRAG()
    results = rag.query("What are the eligibility criteria?")

No document-specific logic. Automatically searches ALL indexed documents.
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from src.config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K, FUSION_TOP_K, FINAL_TOP_K,
    RRF_K, MMR_LAMBDA,
    BM25_CORPUS_PATH,
    ENABLE_QUERY_REFORMULATION, NUM_REFORMULATIONS,
    ENABLE_MULTI_QUERY_RERANKING,
)
from src.indexing.faiss_store   import load_all_indexes
from src.retrieval.dense        import retrieve_dense
from src.retrieval.bm25         import BM25Retriever
from src.retrieval.query_reformulation import reformulate_query
from src.fusion.rrf             import canonical_rrf, mmr_diversify
from src.reranking.cross_encoder import rerank
from src.embeddings.encoder     import encode_query


class HybridRAG:
    """
    Complete Hybrid-RAG retrieval system.

    Combines:
    - Dense Naive RAG retrieval
    - Dense Sentence-Window retrieval
    - Dense Parent-Child retrieval
    - BM25 lexical retrieval

    Followed by:
    - Canonical RRF fusion (groups by document_id + source_chunk_id)
    - MMR diversification (embedding-based, domain-agnostic)
    - CrossEncoder reranking (domain-agnostic)
    """

    def __init__(self) -> None:
        print("Loading Hybrid-RAG system...")

        # Load FAISS indexes
        indexes = load_all_indexes()
        self.naive_index, self.naive_meta         = indexes["naive"]
        self.sw_index,    self.sw_meta            = indexes["sentence_window"]
        self.pc_index,    self.pc_meta            = indexes["parent_child"]
        self.embedding_info                       = indexes["embedding_info"]

        # Load BM25
        self.bm25 = BM25Retriever()
        self.bm25.load(BM25_CORPUS_PATH)

        # Model names from config
        self.embedding_model = EMBEDDING_MODEL_NAME
        self.reranker_model  = RERANKER_MODEL_NAME

        print("Hybrid-RAG system ready.")

    def query(
        self,
        query: str,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        fusion_top_k:    int = FUSION_TOP_K,
        final_top_k:     int = FINAL_TOP_K,
        rrf_k:           int = RRF_K,
        mmr_lambda:      float = MMR_LAMBDA,
        use_bm25:        bool = True,
        use_mmr:         bool = True,
        enable_reformulation: Optional[bool] = None,
    ) -> dict:
        """
        Run the full Hybrid-RAG pipeline for a query.

        Steps:
        1. Formulate queries (original + semantically diverse alternatives if enabled)
        2. Retrieve from all dense strategies (Naive, SW, PC) across query variants
        3. Lexical BM25 retrieval across query variants (optional)
        4. Canonical RRF fusion (combines all variants, preserving canonical identity)
        5. MMR diversification anchored to original user query
        6. CrossEncoder reranking anchored to original user query

        Args:
            query:                Query string.
            retrieval_top_k:      Candidates per retrieval strategy per query variant.
            fusion_top_k:         Candidates after RRF.
            final_top_k:          Final results after reranking.
            rrf_k:                RRF smoothing constant.
            mmr_lambda:           MMR relevance/diversity trade-off.
            use_bm25:             Whether to include BM25 results.
            use_mmr:              Whether to apply MMR diversification.
            enable_reformulation: Whether to generate diverse query formulations.
                                  Defaults to ENABLE_QUERY_REFORMULATION.

        Returns:
            Dict with keys:
                "query", "query_variants", "naive_results", "sw_results",
                "pc_results", "bm25_results", "fused_results",
                "diverse_results", "final_results"
        """
        if enable_reformulation is None:
            enable_reformulation = ENABLE_QUERY_REFORMULATION

        # --------------------------------------------------
        # Step 1: Query Reformulation
        # --------------------------------------------------
        if enable_reformulation:
            query_variants = reformulate_query(query, num_reformulations=NUM_REFORMULATIONS)
        else:
            query_variants = [query]

        # --------------------------------------------------
        # Step 2 & 3: Multi-Query Dense and BM25 Retrieval
        # --------------------------------------------------
        all_lists = []
        orig_naive_results = []
        orig_sw_results    = []
        orig_pc_results    = []
        orig_bm25_results  = []

        for q_idx, q_str in enumerate(query_variants):
            tag = "original" if q_idx == 0 else f"reformulated_{q_idx}"

            naive_res = retrieve_dense(
                q_str, self.naive_index, self.naive_meta,
                self.embedding_model, k=retrieval_top_k
            )
            sw_res = retrieve_dense(
                q_str, self.sw_index, self.sw_meta,
                self.embedding_model, k=retrieval_top_k
            )
            pc_res = retrieve_dense(
                q_str, self.pc_index, self.pc_meta,
                self.embedding_model, k=retrieval_top_k
            )

            # Annotate query provenance
            for res_list in [naive_res, sw_res, pc_res]:
                for item in res_list:
                    item["query_variant"] = tag
                    item["query_text"]    = q_str

            all_lists.extend([naive_res, sw_res, pc_res])

            bm25_res = []
            if use_bm25:
                bm25_res = self.bm25.retrieve(q_str, k=retrieval_top_k)
                for item in bm25_res:
                    item["query_variant"] = tag
                    item["query_text"]    = q_str
                all_lists.append(bm25_res)

            if q_idx == 0:
                orig_naive_results = naive_res
                orig_sw_results    = sw_res
                orig_pc_results    = pc_res
                orig_bm25_results  = bm25_res

        # --------------------------------------------------
        # Step 4: Canonical RRF fusion
        # --------------------------------------------------
        fused = canonical_rrf(all_lists, k=rrf_k)
        fused = fused[:fusion_top_k]

        # --------------------------------------------------
        # Step 5: MMR diversification (anchored to original query)
        # --------------------------------------------------
        diverse = fused
        if use_mmr and len(fused) > final_top_k:
            query_embedding = encode_query(query, self.embedding_model)
            diverse = mmr_diversify(
                candidates=fused,
                query_embedding=query_embedding,
                model_name=self.embedding_model,
                top_k=fusion_top_k,
                lambda_param=mmr_lambda,
            )

        # --------------------------------------------------
        # Step 6: CrossEncoder reranking (multi-query evidence aggregation)
        # --------------------------------------------------
        variants_for_reranking = query_variants if ENABLE_MULTI_QUERY_RERANKING else None
        final_results = rerank(
            query=query,
            candidates=diverse,
            model_name=self.reranker_model,
            top_k=final_top_k,
            query_variants=variants_for_reranking,
        )

        return {
            "query":             query,
            "query_variants":    query_variants,
            "naive_results":     orig_naive_results,
            "sw_results":        orig_sw_results,
            "pc_results":        orig_pc_results,
            "bm25_results":      orig_bm25_results,
            "fused_results":     fused,
            "diverse_results":   diverse,
            "final_results":     final_results,
        }

    def display(self, results: dict) -> None:
        """Pretty-print the final results with full traceability."""
        query  = results["query"]
        finals = results["final_results"]

        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        if not finals:
            print("No results found.")
            return

        for i, r in enumerate(finals, start=1):
            print(f"\nRank {i}")
            print(f"  document_id      : {r.get('document_id', 'N/A')}")
            print(f"  source_file      : {r.get('source_file', 'N/A')}")
            print(f"  retrieval_type   : {r.get('retrieval_type', 'N/A')}")
            print(f"  chunk_id         : {r.get('chunk_id', 'N/A')}")
            print(f"  source_chunk_id  : {r.get('source_chunk_id', 'N/A')}")
            print(f"  section          : {r.get('section', 'N/A')}")
            print(f"  retrieval_sources: {r.get('retrieval_sources', [])}")
            print(f"  retrieval_score  : {r.get('retrieval_score', 0):.4f}")
            print(f"  rrf_score        : {r.get('rrf_score', 0):.6f}")
            print(f"  rerank_score     : {r.get('rerank_score', 0):.4f}")
            print(f"  text (preview)   : {r.get('text', '')[:300]}")

        print(f"\n{'='*70}")
