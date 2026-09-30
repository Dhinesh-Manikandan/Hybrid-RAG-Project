"""
test_crossencoder_multiquery.py -- Comprehensive unit tests for Multi-Query CrossEncoder Mean Aggregation.

Verifies:
1. Single-query fallback (query_variants = [original] or None)
2. Multiple query variants: every candidate receives scores across all variants
3. Mean aggregation: rerank_score == sum(scores) / num_variants
4. Candidate ordering: candidates are sorted descending by mean rerank_score
5. Metadata preservation: document_id, chunk_id, section, etc. remain intact
6. Empty/no reformulation fallback
7. CrossEncoder failure fallback (graceful degradation)
8. Document-agnostic behavior (zero PMAY/domain dependencies, generic candidates)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reranking.cross_encoder import rerank


def _create_generic_candidates():
    """Create generic, domain-agnostic candidate chunks with rich metadata."""
    return [
        {
            "document_id": "doc_arch_001",
            "source_file": "system_design.md",
            "chunk_id": "doc_arch_001__naive__0",
            "source_chunk_id": "doc_arch_001__naive__0",
            "retrieval_type": "naive",
            "section": "System Architecture",
            "heading_path": ["Architecture", "Overview"],
            "retrieval_sources": ["dense", "bm25"],
            "rrf_score": 0.085,
            "text": "The microservices architecture employs event sourcing and CQRS patterns.",
            "retrieval_text": "[Document: System Design]\n\n[Section: System Architecture]\n\nThe microservices architecture employs event sourcing and CQRS patterns.",
        },
        {
            "document_id": "doc_arch_001",
            "source_file": "system_design.md",
            "chunk_id": "doc_arch_001__naive__1",
            "source_chunk_id": "doc_arch_001__naive__1",
            "retrieval_type": "naive",
            "section": "Database Schema",
            "heading_path": ["Architecture", "Storage"],
            "retrieval_sources": ["dense"],
            "rrf_score": 0.072,
            "text": "PostgreSQL is configured with streaming replication and connection pooling.",
            "retrieval_text": "[Document: System Design]\n\n[Section: Database Schema]\n\nPostgreSQL is configured with streaming replication and connection pooling.",
        },
        {
            "document_id": "doc_arch_002",
            "source_file": "deployment_guide.md",
            "chunk_id": "doc_arch_002__naive__0",
            "source_chunk_id": "doc_arch_002__naive__0",
            "retrieval_type": "naive",
            "section": "Deployment Pipeline",
            "heading_path": ["Operations", "CI/CD"],
            "retrieval_sources": ["bm25"],
            "rrf_score": 0.065,
            "text": "Kubernetes helm charts automate zero-downtime rolling deployments.",
            "retrieval_text": "[Document: Deployment Guide]\n\n[Section: Deployment Pipeline]\n\nKubernetes helm charts automate zero-downtime rolling deployments.",
        },
    ]


def test_1_single_query_fallback():
    """When query_variants is [original_query] or None, reduces to single-query CrossEncoder."""
    candidates = _create_generic_candidates()
    mock_reranker = MagicMock()
    # Predict returns 3 scores for 3 candidates
    mock_reranker.predict.return_value = [2.5, 1.0, 3.2]

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        # Case A: query_variants is None
        res_none = rerank("how is database configured", candidates, "dummy-model", top_k=3, query_variants=None)
        assert len(res_none) == 3
        assert mock_reranker.predict.call_count == 1
        assert res_none[0]["section"] == "Deployment Pipeline"  # score 3.2
        assert res_none[0]["rerank_score"] == 3.2

        # Case B: query_variants = [original]
        mock_reranker.reset_mock()
        mock_reranker.predict.return_value = [2.5, 1.0, 3.2]
        res_single = rerank("how is database configured", candidates, "dummy-model", top_k=3, query_variants=["how is database configured"])
        assert len(res_single) == 3
        assert res_single[0]["rerank_score"] == 3.2
        print("PASS: Test 1 - Single-query fallback matches single-query behavior")


def test_2_multiple_query_variants_batching():
    """Verify that multiple query variants construct V x N pairs in a single batched call."""
    candidates = _create_generic_candidates()  # 3 candidates
    variants = [
        "how is database configured",
        "database storage specifications and replication",
        "data layer persistence architecture",
    ]  # 3 variants -> 3 x 3 = 9 pairs
    mock_reranker = MagicMock()
    # Mock scores for 9 pairs
    mock_reranker.predict.return_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        reranked = rerank("how is database configured", candidates, "dummy-model", top_k=3, query_variants=variants)

        # Confirm exactly ONE batched call was made to predict
        assert mock_reranker.predict.call_count == 1
        pairs_called = mock_reranker.predict.call_args[0][0]
        assert len(pairs_called) == 9  # 3 variants x 3 candidates

        # Check pair ordering: (v0, c0), (v0, c1), (v0, c2), (v1, c0)...
        assert pairs_called[0][0] == variants[0]
        assert pairs_called[3][0] == variants[1]
        assert pairs_called[6][0] == variants[2]
        print("PASS: Test 2 - Multiple query variants scored in single batched predict call")


def test_3_mean_aggregation_exactness():
    """Verify that rerank_score is the exact arithmetic mean of scores across all variants."""
    candidates = _create_generic_candidates()  # 3 candidates: C0, C1, C2
    variants = ["Q_orig", "Q_var1", "Q_var2"]  # 3 variants

    # Candidate 0: scores [3.0, 6.0, 9.0] -> Mean = 6.0
    # Candidate 1: scores [2.0, 4.0, 6.0] -> Mean = 4.0
    # Candidate 2: scores [1.0, 5.0, 12.0] -> Mean = 6.0
    # Pairs order: (v0, c0), (v0, c1), (v0, c2), (v1, c0), (v1, c1), (v1, c2), (v2, c0), (v2, c1), (v2, c2)
    scores = [3.0, 2.0, 1.0, 6.0, 4.0, 5.0, 9.0, 6.0, 12.0]

    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = scores

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        reranked = rerank("Q_orig", candidates, "dummy-model", top_k=3, query_variants=variants)

        # Locate candidates by chunk_id
        c0 = next(c for c in reranked if c["chunk_id"] == "doc_arch_001__naive__0")
        c1 = next(c for c in reranked if c["chunk_id"] == "doc_arch_001__naive__1")
        c2 = next(c for c in reranked if c["chunk_id"] == "doc_arch_002__naive__0")

        assert abs(c0["rerank_score"] - 6.0) < 1e-6
        assert abs(c1["rerank_score"] - 4.0) < 1e-6
        assert abs(c2["rerank_score"] - 6.0) < 1e-6
        assert c0["rerank_scores_by_variant"] == [3.0, 6.0, 9.0]
        assert c1["rerank_scores_by_variant"] == [2.0, 4.0, 6.0]
        assert c2["rerank_scores_by_variant"] == [1.0, 5.0, 12.0]
        print("PASS: Test 3 - Mean aggregation calculates exact arithmetic mean across variants")


def test_4_candidate_ordering():
    """Verify that results are strictly ordered descending by rerank_score."""
    candidates = _create_generic_candidates()
    variants = ["Q1", "Q2"]
    # Candidate 0: [1.0, 1.0] -> 1.0
    # Candidate 1: [5.0, 7.0] -> 6.0
    # Candidate 2: [2.0, 4.0] -> 3.0
    scores = [1.0, 5.0, 2.0, 1.0, 7.0, 4.0]

    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = scores

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        reranked = rerank("Q1", candidates, "dummy-model", top_k=3, query_variants=variants)

        assert reranked[0]["chunk_id"] == "doc_arch_001__naive__1"  # Score 6.0
        assert reranked[1]["chunk_id"] == "doc_arch_002__naive__0"  # Score 3.0
        assert reranked[2]["chunk_id"] == "doc_arch_001__naive__0"  # Score 1.0
        assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"] >= reranked[2]["rerank_score"]
        print("PASS: Test 4 - Candidates strictly sorted descending by mean rerank_score")


def test_5_metadata_preservation():
    """Verify that all lineage metadata fields are preserved intact."""
    candidates = _create_generic_candidates()
    variants = ["Q1", "Q2"]
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        reranked = rerank("Q1", candidates, "dummy-model", top_k=3, query_variants=variants)

        orig_by_id = {c["chunk_id"]: c for c in candidates}
        for rerank_c in reranked:
            orig_c = orig_by_id[rerank_c["chunk_id"]]
            for field in ["document_id", "source_file", "chunk_id", "source_chunk_id", "retrieval_type", "section", "heading_path", "rrf_score", "retrieval_sources"]:
                assert field in rerank_c, f"Missing field {field}"
                assert rerank_c[field] == orig_c[field]
        print("PASS: Test 5 - All candidate metadata fields strictly preserved")


def test_6_empty_or_malformed_variants_fallback():
    """Verify that empty strings, empty lists, or malformed variant inputs fall back safely."""
    candidates = _create_generic_candidates()
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [1.0, 2.0, 3.0]

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        # Empty list
        r1 = rerank("valid query", candidates, "dummy-model", top_k=3, query_variants=[])
        assert len(r1) == 3

        # List with only empty strings
        mock_reranker.reset_mock()
        mock_reranker.predict.return_value = [1.0, 2.0, 3.0]
        r2 = rerank("valid query", candidates, "dummy-model", top_k=3, query_variants=["   ", ""])
        assert len(r2) == 3
        # Should execute single query on "valid query"
        pairs = mock_reranker.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[0][0] == "valid query"
        print("PASS: Test 6 - Empty or malformed variants fall back safely to primary query")


def test_7_crossencoder_failure_fallback():
    """Verify that if multi-query scoring throws an exception, system falls back to single query."""
    candidates = _create_generic_candidates()
    variants = ["Q_orig", "Q_var1", "Q_var2"]

    mock_reranker = MagicMock()
    # First call (multi-query with 9 pairs) raises an exception, second call (single-query fallback) succeeds
    mock_reranker.predict.side_effect = [RuntimeError("CUDA OOM or tensor error"), [4.0, 2.0, 5.0]]

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        reranked = rerank("Q_orig", candidates, "dummy-model", top_k=3, query_variants=variants)

        # Must have fallen back to single-query without raising exception
        assert mock_reranker.predict.call_count == 2
        assert len(reranked) == 3
        assert reranked[0]["rerank_score"] == 5.0
        print("PASS: Test 7 - Multi-query failure gracefully falls back to single-query reranking")


def test_8_document_agnostic_arbitrary_metadata():
    """Verify that multi-query reranking functions with arbitrary domain metadata."""
    # Custom biomedical domain chunks
    bio_candidates = [
        {
            "document_id": "pmc_882910",
            "source_file": "genetics_paper.pdf",
            "chunk_id": "pmc_882910__0",
            "text": "CRISPR-Cas9 endonuclease mediates targeted double-strand breaks.",
            "section": "Methods and Reagents",
        },
        {
            "document_id": "pmc_994120",
            "source_file": "clinical_trial.pdf",
            "chunk_id": "pmc_994120__0",
            "text": "Phase II trial demonstrated 78% efficacy with minimal off-target effects.",
            "section": "Results and Efficacy",
        },
    ]
    bio_variants = ["CRISPR efficacy trial results", "Cas9 gene editing outcomes"]
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [2.0, 5.0, 3.0, 6.0]

    with patch("src.reranking.cross_encoder.get_reranker", return_value=mock_reranker):
        reranked = rerank("CRISPR efficacy trial results", bio_candidates, "dummy-model", top_k=2, query_variants=bio_variants)

        assert len(reranked) == 2
        assert reranked[0]["document_id"] == "pmc_994120"
        assert reranked[0]["section"] == "Results and Efficacy"
        assert abs(reranked[0]["rerank_score"] - 5.5) < 1e-6
        print("PASS: Test 8 - Functions seamlessly across arbitrary domains and metadata")


if __name__ == "__main__":
    test_1_single_query_fallback()
    test_2_multiple_query_variants_batching()
    test_3_mean_aggregation_exactness()
    test_4_candidate_ordering()
    test_5_metadata_preservation()
    test_6_empty_or_malformed_variants_fallback()
    test_7_crossencoder_failure_fallback()
    test_8_document_agnostic_arbitrary_metadata()
    print("\nALL 8 MULTI-QUERY CROSSENCODER TESTS PASSED!")
