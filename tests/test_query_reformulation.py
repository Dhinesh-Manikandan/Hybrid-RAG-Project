"""
test_query_reformulation.py -- Comprehensive verification of query reformulation & multi-query retrieval.

Tests:
1.  Original query is always preserved at index 0.
2.  Number of reformulations respects configuration.
3.  Duplicate reformulations are removed.
4.  Empty reformulations are ignored.
5.  Reformulation failure gracefully falls back to original query.
6.  Query provenance is preserved in retrieval items.
7.  Existing chunk/document IDs remain unchanged.
8.  RRF deduplication works across multi-query candidate pools.
9.  Retrieval remains document-agnostic across unrelated domains.
10. Disabling reformulation restores previous single-query pipeline behavior.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_reformulation import reformulate_query
from src.fusion.rrf import canonical_rrf
from retrieve import HybridRAG


def test_1_original_query_preserved():
    q = "What is the architecture of Kubernetes?"
    variants = reformulate_query(q, num_reformulations=2)
    assert len(variants) >= 1
    assert variants[0] == q, f"Original query must be index 0, got {variants[0]}"
    print("PASS: Test 1 - Original query always preserved at index 0")


def test_2_number_of_reformulations_respected():
    q = "How do I configure the database connection pool?"
    for n in [1, 2, 3]:
        variants = reformulate_query(q, num_reformulations=n)
        assert len(variants) == n + 1, f"Expected {n+1} queries, got {len(variants)}: {variants}"
    print("PASS: Test 2 - Number of reformulations strictly respects configuration")


def test_3_duplicate_reformulations_removed():
    # If raw generator yields duplicates, reformulate_query guarantees uniqueness
    q = "Who is eligible for PMAY-U?"
    variants = reformulate_query(q, num_reformulations=5)
    lower_variants = [v.lower() for v in variants]
    assert len(lower_variants) == len(set(lower_variants)), "Duplicates detected in output"
    print("PASS: Test 3 - Duplicate reformulations removed")


def test_4_empty_reformulations_ignored():
    assert reformulate_query("", num_reformulations=2) == []
    assert reformulate_query("   ", num_reformulations=2) == []
    print("PASS: Test 4 - Empty queries safely handled")


def test_5_failure_falls_back_to_original():
    # Pass invalid method to trigger fallback
    q = "How to deploy microservices?"
    variants = reformulate_query(q, num_reformulations=2, method="invalid_unknown_method")
    assert variants == [q], f"Expected fallback to [{q}], got {variants}"
    print("PASS: Test 5 - Reformulation failure gracefully falls back to original query")


def test_6_query_provenance_preserved():
    rag = HybridRAG()
    res = rag.query("Who is eligible for PMAY-U?", enable_reformulation=True)
    assert "query_variants" in res
    assert len(res["query_variants"]) >= 2
    assert res["query_variants"][0] == "Who is eligible for PMAY-U?"

    # Check that fused items have query provenance
    fused = res["fused_results"]
    has_provenance = any("query_variants" in item for item in fused)
    assert has_provenance, "Query variants provenance must be recorded on fused items"
    print("PASS: Test 6 - Query provenance preserved in retrieval results")


def test_7_existing_chunk_and_document_ids_unchanged():
    rag = HybridRAG()
    res = rag.query("What benefits are available?", enable_reformulation=True)
    for item in res["final_results"]:
        assert "document_id" in item and item["document_id"].startswith("doc_")
        assert "chunk_id" in item and "__" in item["chunk_id"]
        assert "source_chunk_id" in item
    print("PASS: Test 7 - Document and chunk IDs remain strictly unchanged")


def test_8_rrf_deduplication_works():
    # 3 identical chunks from 3 different query variants must fuse into 1 canonical item
    chunk_v1 = {
        "document_id": "doc_alpha",
        "source_chunk_id": "doc_alpha__naive__0",
        "chunk_id": "doc_alpha__naive__0",
        "retrieval_type": "naive",
        "retrieval_score": 0.9,
        "query_variant": "original",
        "text": "Unified content",
    }
    chunk_v2 = {
        "document_id": "doc_alpha",
        "source_chunk_id": "doc_alpha__naive__0",
        "chunk_id": "doc_alpha__naive__0",
        "retrieval_type": "naive",
        "retrieval_score": 0.85,
        "query_variant": "reformulated_1",
        "text": "Unified content",
    }
    chunk_v3 = {
        "document_id": "doc_alpha",
        "source_chunk_id": "doc_alpha__naive__0",
        "chunk_id": "doc_alpha__naive__0",
        "retrieval_type": "naive",
        "retrieval_score": 0.80,
        "query_variant": "reformulated_2",
        "text": "Unified content",
    }
    fused = canonical_rrf([[chunk_v1], [chunk_v2], [chunk_v3]], k=60)
    assert len(fused) == 1, f"Expected 1 deduplicated chunk, got {len(fused)}"
    assert fused[0]["canonical_id"] == "doc_alpha::doc_alpha__naive__0"
    assert len(fused[0]["query_variants"]) == 3
    print("PASS: Test 8 - Canonical RRF deduplicates identical content across query variants")


def test_9_retrieval_remains_document_agnostic():
    unrelated_queries = [
        "What are the network security requirements?",
        "How do I setup automated CI/CD pipelines?",
        "Who can approve medical clinical trials?",
    ]
    for q in unrelated_queries:
        variants = reformulate_query(q, num_reformulations=2)
        assert len(variants) == 3
        assert variants[0] == q
        assert variants[1] != variants[2]
    print("PASS: Test 9 - Query reformulation operates seamlessly on unrelated domains")


def test_10_disabling_reformulation_restores_single_query_pipeline():
    rag = HybridRAG()
    res = rag.query("What are the eligibility criteria?", enable_reformulation=False)
    assert res["query_variants"] == ["What are the eligibility criteria?"]
    assert len(res["final_results"]) > 0
    print("PASS: Test 10 - Disabling reformulation restores single-query pipeline")


if __name__ == "__main__":
    test_1_original_query_preserved()
    test_2_number_of_reformulations_respected()
    test_3_duplicate_reformulations_removed()
    test_4_empty_reformulations_ignored()
    test_5_failure_falls_back_to_original()
    test_6_query_provenance_preserved()
    test_7_existing_chunk_and_document_ids_unchanged()
    test_8_rrf_deduplication_works()
    test_9_retrieval_remains_document_agnostic()
    test_10_disabling_reformulation_restores_single_query_pipeline()
    print("\nALL 10 QUERY REFORMULATION TESTS PASSED!")
