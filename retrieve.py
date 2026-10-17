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
from typing import Any, Optional

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
    GENERATION_CONTEXT_TOP_K,
    RERANK_RELEVANCE_THRESHOLD,
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
        self.generation_context_top_k = GENERATION_CONTEXT_TOP_K
        self._generator = None

        print("Hybrid-RAG system ready.")

    def get_generator(self):
        """Lazy load LLM generator instance."""
        if self._generator is None:
            from src.generation.llm_generator import get_generator
            self._generator = get_generator()
        return self._generator

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
        generate_answer: bool = False,
        context_top_k:   int = GENERATION_CONTEXT_TOP_K,
        generator:       Optional[Any] = None,
        status_callback: Optional[Any] = None,
        **kwargs,
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
            generate_answer:      Whether to build context and generate an LLM answer.
            context_top_k:        Number of top candidates used for context generation.
            generator:            Optional custom LLMGenerator instance.
            status_callback:      Optional callback(stage: str, msg: str, data: dict) for live step updates.

        Returns:
            Dict with keys:
                "query", "query_variants", "naive_results", "sw_results",
                "pc_results", "bm25_results", "fused_results",
                "diverse_results", "final_results"
        """
        if status_callback is None and "status_callback" in kwargs:
            status_callback = kwargs.pop("status_callback")

        def _notify(stage: str, msg: str, data: Optional[dict] = None):
            if status_callback is not None:
                try:
                    status_callback(stage, msg, data or {})
                except Exception:
                    pass

        if enable_reformulation is None:
            enable_reformulation = ENABLE_QUERY_REFORMULATION

        # --------------------------------------------------
        # Step 1: Query Reformulation
        # --------------------------------------------------
        _notify("reformulation_start", "Formulating multi-query search perspectives...")
        if enable_reformulation:
            # NUM_REFORMULATIONS in config specifies total query perspectives (V=3: original + 2 alternatives)
            num_alts = NUM_REFORMULATIONS - 1 if NUM_REFORMULATIONS > 1 else 1
            query_variants = reformulate_query(query, num_reformulations=num_alts)
        else:
            query_variants = [query]
        _notify("reformulation_done", f"Formulated {len(query_variants)} diverse search perspectives", {"variants": query_variants})

        # --------------------------------------------------
        # Step 2 & 3: Multi-Query Dense and BM25 Retrieval
        # --------------------------------------------------
        _notify("retrieval_start", f"Searching Dense (Naive, Sentence-Window, Parent-Child) & BM25 indices...")
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

        _notify("retrieval_done", f"Retrieved candidate pools across 4 retrieval strategies ({len(all_lists)} search passes)")

        # --------------------------------------------------
        # Step 4: Canonical RRF fusion
        # --------------------------------------------------
        _notify("fusion_start", "Fusing candidate pools using Canonical Reciprocal Rank Fusion (RRF)...")
        fused = canonical_rrf(all_lists, k=rrf_k)
        fused = fused[:fusion_top_k]

        # --------------------------------------------------
        # Step 5: MMR diversification (anchored to original query)
        # --------------------------------------------------
        diverse = fused
        if use_mmr and len(fused) > final_top_k:
            _notify("mmr_start", "Applying MMR diversification across semantic embeddings...")
            query_embedding = encode_query(query, self.embedding_model)
            diverse = mmr_diversify(
                candidates=fused,
                query_embedding=query_embedding,
                model_name=self.embedding_model,
                top_k=fusion_top_k,
                lambda_param=mmr_lambda,
            )
        _notify("fusion_done", f"Fused and diversified candidates into top-{len(diverse)} candidate pool")

        # --------------------------------------------------
        # Step 6: CrossEncoder reranking (multi-query evidence aggregation)
        # --------------------------------------------------
        _notify("rerank_start", "Neural reranking with Cross-Encoder...")
        variants_for_reranking = query_variants if ENABLE_MULTI_QUERY_RERANKING else None
        final_results = rerank(
            query=query,
            candidates=diverse,
            model_name=self.reranker_model,
            top_k=final_top_k,
            query_variants=variants_for_reranking,
        )
        _notify("rerank_done", f"Cross-Encoder scored and selected top-{len(final_results)} most relevant passages")

        output = {
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

        if generate_answer:
            # ── Relevance gate ────────────────────────────────────────────────
            # The CrossEncoder already scored every retrieved passage against the
            # user query.  If even the best-scoring passage is below our calibrated
            # threshold, the query is out-of-scope for this knowledge base and we
            # skip the LLM entirely, returning a clean refusal instead of a
            # hallucinated answer.
            top_rerank_score = (
                max((r.get("rerank_score", -999) for r in final_results), default=-999)
                if final_results else -999
            )

            if top_rerank_score < RERANK_RELEVANCE_THRESHOLD:
                _notify(
                    "generation_done",
                    "Query is out-of-scope: no relevant passages found above relevance threshold",
                )
                output["context"] = ""
                output["answer"] = (
                    "I'm sorry, but I can only answer questions about the topics and documents "
                    "in my knowledge base.\n\n"
                    "Your question does not appear to be covered by the retrieved documents. "
                    "Please ask a relevant question and I'll be happy to help."
                )

                output["sources"] = []
            else:
                _notify("generation_start", "Synthesizing grounded answer with local Qwen model...")
                from src.generation.context_builder import build_context
                gen = generator if generator is not None else self.get_generator()
                context_str, sources = build_context(final_results, top_k=context_top_k, query=query)
                gen_res = gen.generate(
                    query=query,
                    context=context_str,
                    candidates=final_results,
                    top_k=context_top_k,
                )
                output["context"] = context_str
                output["answer"]  = gen_res["answer"]
                output["sources"] = gen_res["sources"] if gen_res.get("sources") else sources
                _notify("generation_done", f"Generated grounded response with {len(output.get('sources', []))} supporting citations")

        return output

    def answer(
        self,
        query: str,
        context_top_k: int = GENERATION_CONTEXT_TOP_K,
        generator: Optional[Any] = None,
        status_callback: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        """
        Convenience end-to-end interface: retrieve relevant passages and generate
        a grounded answer with source attribution using Qwen3-8B.
        """
        return self.query(
            query=query,
            generate_answer=True,
            context_top_k=context_top_k,
            generator=generator,
            status_callback=status_callback,
            **kwargs,
        )

    def display(self, results: dict) -> None:
        """Pretty-print the final results with full traceability."""
        query  = results["query"]
        finals = results["final_results"]

        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        if "answer" in results:
            print("\n" + "-"*70)
            print("GROUNDED ANSWER (Qwen3-8B):")
            print("-"*70)
            print(results["answer"])
            print("\n" + "-"*70)
            print("CITED SOURCES:")
            print("-"*70)
            for s in results.get("sources", []):
                print(f"  [{s.get('source_index', '?')}] {s.get('document_title', 'Unknown')} ({s.get('source_file', 'N/A')})")
                print(f"      Section : {s.get('section', 'N/A')}")
                print(f"      Chunk ID: {s.get('chunk_id', 'N/A')}")
            print("-"*70)

        if not finals:
            print("No results found.")
            return

        print("\nRETRIEVAL CANDIDATES:")
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
