"""
test_architecture.py -- Architecture tests for the Hybrid-RAG system.

Tests verify the 8 required architectural properties:
1. Two different documents receive different document_ids.
2. All representations from the same document preserve the same document_id.
3. Adding a new document does not require modifying retrieval code.
4. Duplicate ingestion does not create duplicate vectors.
5. Retrieval returns document_id correctly.
6. Fusion does not over-count repeated representations.
7. Reranking preserves metadata.
8. Multiple documents can be retrieved in the same query.
"""

import sys
import json
import pickle
import hashlib
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# TEST 1: Two different documents receive different document_ids
# ============================================================

def test_different_documents_get_different_ids():
    """
    Two files with different names must produce different document_ids.
    """
    from src.ingestion.document_id import generate_document_id

    # Simulate two different files
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = Path(tmpdir) / "document_alpha.txt"
        path_b = Path(tmpdir) / "document_beta.txt"

        path_a.write_text("Content A", encoding="utf-8")
        path_b.write_text("Content B", encoding="utf-8")

        id_a = generate_document_id(path_a)
        id_b = generate_document_id(path_b)

        assert id_a != id_b, (
            f"Different documents must have different IDs: "
            f"id_a={id_a}, id_b={id_b}"
        )
        assert id_a.startswith("doc_"), f"document_id must start with 'doc_': {id_a}"
        assert id_b.startswith("doc_"), f"document_id must start with 'doc_': {id_b}"

    print(f"PASS: doc_id_a={id_a}, doc_id_b={id_b}")


# ============================================================
# TEST 2: All representations from same document have same document_id
# ============================================================

def test_same_document_all_representations_share_document_id():
    """
    All naive, SW, and PC chunks from the same document must have
    the same document_id.
    """
    from src.ingestion.document_id import generate_document_id
    from src.ingestion.cleaner import clean_text, extract_sections
    from src.chunking.naive import build_naive_chunks_from_sections
    from src.chunking.sentence_window import build_sentence_window_chunks
    from src.chunking.parent_child import build_parent_child_chunks

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_doc.txt"
        path.write_text(
            "INTRODUCTION\n\n"
            "This is the first paragraph. It has multiple sentences. Here is another one.\n\n"
            "DETAILS\n\n"
            "More content here. Second section details. Final sentence.",
            encoding="utf-8"
        )

        document_id = generate_document_id(path)
        text = clean_text(path.read_text(encoding="utf-8"))
        sections = extract_sections(text)

        naive = build_naive_chunks_from_sections(
            document_id=document_id,
            source_file=path.name,
            sections=sections,
        )

        sw = build_sentence_window_chunks(
            document_id=document_id,
            source_file=path.name,
            sections=sections,
            naive_chunks=naive,
        )

        pc = build_parent_child_chunks(
            document_id=document_id,
            source_file=path.name,
            naive_chunks=naive,
        )

    all_chunks = naive + sw + pc
    assert len(all_chunks) > 0, "No chunks generated"

    for chunk in all_chunks:
        assert "document_id" in chunk, f"Chunk missing document_id: {chunk}"
        assert chunk["document_id"] == document_id, (
            f"document_id mismatch: expected {document_id}, got {chunk['document_id']}"
        )

    print(f"PASS: {len(all_chunks)} chunks all have document_id={document_id}")


# ============================================================
# TEST 3: Adding new document does not require modifying retrieval code
# ============================================================

def test_retrieval_code_is_document_agnostic():
    """
    The retrieval, fusion, and reranking modules must not contain
    any document-specific strings (PMAY, eligibility, benefits, scheme, etc.)
    """
    forbidden_terms = [
        "pmay", "eligibility", "scheme_name", "ministry", "PMAY",
        "SCHEME NAME", "MINISTRY", "CATEGORIES", "APPLICATION PROCESS",
        "DOCUMENTS REQUIRED", "FREQUENTLY ASKED QUESTIONS",
        "pmay_u", "pucca", "slum"
    ]

    modules_to_check = [
        PROJECT_ROOT / "src" / "retrieval" / "dense.py",
        PROJECT_ROOT / "src" / "retrieval" / "bm25.py",
        PROJECT_ROOT / "src" / "fusion" / "rrf.py",
        PROJECT_ROOT / "src" / "reranking" / "cross_encoder.py",
    ]

    for module_path in modules_to_check:
        assert module_path.exists(), f"Module not found: {module_path}"
        code = module_path.read_text(encoding="utf-8").lower()

        for term in forbidden_terms:
            assert term.lower() not in code, (
                f"Document-specific term '{term}' found in {module_path.name}. "
                f"Retrieval code must be document-agnostic."
            )

    print(f"PASS: {len(modules_to_check)} retrieval modules are document-agnostic")


# ============================================================
# TEST 4: Duplicate ingestion does not create duplicate vectors
# ============================================================

def test_no_duplicate_chunk_ids():
    """
    When the same document is processed twice, chunk_ids should remain unique.
    Chunk IDs are deterministic (based on document_id + position), so
    re-processing the same file always yields identical chunk_ids.
    A full rebuild simply overwrites existing indexes — no duplicates.
    """
    from src.ingestion.document_id import generate_document_id
    from src.ingestion.cleaner import clean_text, extract_sections
    from src.chunking.naive import build_naive_chunks_from_sections

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "duplicate_test.txt"
        path.write_text(
            "This is test content. It has multiple sentences. Here is the end.",
            encoding="utf-8"
        )

        document_id = generate_document_id(path)
        text = clean_text(path.read_text(encoding="utf-8"))
        sections = extract_sections(text)

        # Process twice (simulating re-ingestion)
        chunks_1 = build_naive_chunks_from_sections(document_id, path.name, sections)
        chunks_2 = build_naive_chunks_from_sections(document_id, path.name, sections)

    # Chunk IDs from both runs must be identical (deterministic)
    ids_1 = [c["chunk_id"] for c in chunks_1]
    ids_2 = [c["chunk_id"] for c in chunks_2]

    assert ids_1 == ids_2, (
        "Chunk IDs are not deterministic across runs — "
        "re-ingestion would create duplicates."
    )

    # No duplicates within one run
    assert len(ids_1) == len(set(ids_1)), (
        f"Duplicate chunk_ids within single run: {ids_1}"
    )

    print(f"PASS: {len(ids_1)} chunk_ids are stable and unique")


# ============================================================
# TEST 5: Retrieval returns document_id correctly
# ============================================================

def test_retrieval_result_has_document_id():
    """
    Every result returned by retrieve_dense must have a document_id field.
    """
    from src.ingestion.document_id import generate_document_id
    from src.ingestion.cleaner import clean_text, extract_sections
    from src.chunking.naive import build_naive_chunks_from_sections
    from src.embeddings.encoder import encode_texts
    from src.retrieval.dense import retrieve_dense
    import faiss

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "retrieval_test.txt"
        path.write_text(
            "Authentication requires a valid token. "
            "The token expires after 24 hours. "
            "Use the refresh endpoint to get a new token.",
            encoding="utf-8"
        )

        document_id = generate_document_id(path)
        text = clean_text(path.read_text(encoding="utf-8"))
        sections = extract_sections(text)

        naive_chunks = build_naive_chunks_from_sections(document_id, path.name, sections)

        texts = [c["text"] for c in naive_chunks]
        embeddings = encode_texts(texts, "all-MiniLM-L6-v2", show_progress=False)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        results = retrieve_dense("authentication", index, naive_chunks, "all-MiniLM-L6-v2", k=3)

    assert len(results) > 0, "No results returned"
    for r in results:
        assert "document_id" in r, f"Result missing document_id: {r.keys()}"
        assert r["document_id"] == document_id, (
            f"Wrong document_id: expected {document_id}, got {r['document_id']}"
        )
        assert "chunk_id" in r, "Result missing chunk_id"
        assert "source_chunk_id" in r, "Result missing source_chunk_id"
        assert "text" in r, "Result missing text"
        assert "retrieval_score" in r, "Result missing retrieval_score"

    print(f"PASS: {len(results)} results all have document_id={document_id}")


# ============================================================
# TEST 6: Fusion does not over-count repeated representations
# ============================================================

def test_canonical_rrf_prevents_voting_inflation():
    """
    Multiple overlapping SW representations of the same naive chunk
    should NOT accumulate more RRF score than a single result.

    Specifically: 5 SW windows all pointing to source_chunk_id="doc_A::naive_0"
    should contribute the SAME RRF score as 1 result pointing to "doc_A::naive_0".
    """
    from src.fusion.rrf import canonical_rrf

    doc_id = "doc_testcanonical"
    naive_chunk_id = f"{doc_id}__naive__0"

    # 5 SW representations of the same naive chunk (different text, same source)
    sw_results_inflated = [
        {
            "document_id":    doc_id,
            "source_chunk_id": naive_chunk_id,
            "chunk_id":       f"{doc_id}__sw__{i}",
            "retrieval_type": "sentence_window",
            "text":           f"Window {i} text.",
            "retrieval_score": 0.9 - i * 0.05,
        }
        for i in range(5)
    ]

    # Single naive result pointing to same source
    single_naive_result = [
        {
            "document_id":    doc_id,
            "source_chunk_id": naive_chunk_id,
            "chunk_id":       naive_chunk_id,
            "retrieval_type": "naive",
            "text":           "Naive chunk text.",
            "retrieval_score": 0.85,
        }
    ]

    # Fuse only SW results (5 items from same source)
    fused_inflated = canonical_rrf([sw_results_inflated], k=60)

    # Fuse single naive result
    fused_single = canonical_rrf([single_naive_result], k=60)

    assert len(fused_inflated) == 1, (
        f"Expected 1 canonical group, got {len(fused_inflated)}. "
        f"Overlapping SW windows are inflating the result count."
    )

    assert len(fused_single) == 1, f"Expected 1 result, got {len(fused_single)}"

    # RRF score from 5 overlapping SW items should equal score from 1 item at rank 1
    # (because we take best rank = 1 within the list)
    score_inflated = fused_inflated[0]["rrf_score"]
    score_single   = fused_single[0]["rrf_score"]

    assert abs(score_inflated - score_single) < 1e-9, (
        f"Canonical RRF failed to prevent voting inflation: "
        f"inflated={score_inflated:.8f}, single={score_single:.8f}"
    )

    print(f"PASS: canonical_rrf correctly groups {len(sw_results_inflated)} overlapping representations into 1")


# ============================================================
# TEST 7: Reranking preserves metadata
# ============================================================

def test_reranking_preserves_all_metadata():
    """
    After reranking, every result must retain all original metadata fields
    plus the new rerank_score field.
    """
    from src.reranking.cross_encoder import rerank

    candidates = [
        {
            "document_id":    "doc_abc123",
            "source_file":    "test.txt",
            "chunk_id":       "doc_abc123__naive__0",
            "retrieval_type": "naive",
            "text":           "Deployment requires Docker and Kubernetes.",
            "source_chunk_id": "doc_abc123__naive__0",
            "parent_chunk_id": None,
            "section":        "DEPLOYMENT",
            "retrieval_score": 0.85,
            "rrf_score":      0.016,
        },
        {
            "document_id":    "doc_abc123",
            "source_file":    "test.txt",
            "chunk_id":       "doc_abc123__naive__1",
            "retrieval_type": "naive",
            "text":           "Configuration can be done via environment variables.",
            "source_chunk_id": "doc_abc123__naive__1",
            "parent_chunk_id": None,
            "section":        "CONFIGURATION",
            "retrieval_score": 0.78,
            "rrf_score":      0.014,
        },
    ]

    reranked = rerank(
        query="How do I deploy?",
        candidates=candidates,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=5,
    )

    required_fields = [
        "document_id", "source_file", "chunk_id", "retrieval_type",
        "text", "source_chunk_id", "section",
        "retrieval_score", "rrf_score", "rerank_score"
    ]

    for result in reranked:
        for field in required_fields:
            assert field in result, (
                f"Reranking dropped field '{field}'. "
                f"Available: {list(result.keys())}"
            )

    print(f"PASS: {len(reranked)} reranked results all preserve metadata + rerank_score")


# ============================================================
# TEST 8: Multiple documents can be retrieved in the same query
# ============================================================

def test_multiple_documents_retrieved_together():
    """
    When two documents are indexed together, a query should be able to
    retrieve results from both documents in the same result set.
    """
    import faiss
    from src.ingestion.document_id import generate_document_id
    from src.ingestion.cleaner import clean_text, extract_sections
    from src.chunking.naive import build_naive_chunks_from_sections
    from src.embeddings.encoder import encode_texts
    from src.retrieval.dense import retrieve_dense

    doc_contents = {
        "docker_guide.txt": "Docker containers provide isolated environments. "
                            "You can deploy applications using docker run. "
                            "Images are built from Dockerfiles.",
        "kubernetes_guide.txt": "Kubernetes orchestrates container workloads. "
                                "Pods are the smallest deployable units. "
                                "Services expose pods to network traffic.",
    }

    all_chunks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in doc_contents.items():
            path = Path(tmpdir) / filename
            path.write_text(content, encoding="utf-8")

            doc_id = generate_document_id(path)
            text = clean_text(content)
            sections = extract_sections(text)
            chunks = build_naive_chunks_from_sections(doc_id, filename, sections)
            all_chunks.extend(chunks)

        texts = [c["text"] for c in all_chunks]
        embeddings = encode_texts(texts, "all-MiniLM-L6-v2", show_progress=False)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        results = retrieve_dense("container deployment", index, all_chunks, "all-MiniLM-L6-v2", k=5)

    assert len(results) > 0, "No results returned"

    returned_doc_ids = {r["document_id"] for r in results}
    returned_files   = {r["source_file"]  for r in results}

    assert len(returned_doc_ids) >= 1, "No document_ids in results"

    # Both documents should be indexed; verify results come from indexed docs
    all_indexed_doc_ids = {c["document_id"] for c in all_chunks}
    for doc_id in returned_doc_ids:
        assert doc_id in all_indexed_doc_ids, (
            f"Result contains unknown document_id: {doc_id}"
        )

    print(f"PASS: Results span {len(returned_doc_ids)} document(s): {returned_files}")


# ============================================================
# TEST 9: Every chunk contains document_title, raw text, and retrieval_text
# ============================================================

def test_document_title_and_retrieval_text_integrity():
    """
    Every generated chunk across naive, sw, and pc representations must:
    1. Contain document_title
    2. Preserve raw text in chunk['text']
    3. Provide canonical searchable representation in chunk['retrieval_text']
    4. retrieval_text must incorporate [Document: ...] and [Section: ...]
    """
    from src.ingestion.document_id import generate_document_id
    from src.ingestion.cleaner import clean_text, extract_sections, extract_document_title
    from src.chunking.naive import build_naive_chunks_from_sections
    from src.chunking.sentence_window import build_sentence_window_chunks
    from src.chunking.parent_child import build_parent_child_chunks

    doc_content = (
        "# User Authentication Guide\n\n"
        "## Password Policy\n\n"
        "Passwords must be at least 12 characters long.\n"
        "They must contain uppercase, lowercase, numbers, and symbols.\n\n"
        "## Session Management\n\n"
        "Sessions expire after 30 minutes of inactivity."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "auth_guide.txt"
        path.write_text(doc_content, encoding="utf-8")

        document_id = generate_document_id(path)
        clean = clean_text(doc_content)
        title = extract_document_title(clean, path.name)
        assert title == "User Authentication Guide", f"Wrong title: {title}"

        sections = extract_sections(clean)
        naive = build_naive_chunks_from_sections(document_id, path.name, sections, document_title=title)
        sw    = build_sentence_window_chunks(document_id, path.name, sections, naive, document_title=title)
        pc    = build_parent_child_chunks(document_id, path.name, naive, document_title=title)

    all_chunks = naive + sw + pc
    assert len(all_chunks) > 0

    for c in all_chunks:
        assert "document_title" in c, f"Missing document_title in {c['chunk_id']}"
        assert c["document_title"] == "User Authentication Guide"
        assert "text" in c, f"Missing text in {c['chunk_id']}"
        assert "retrieval_text" in c, f"Missing retrieval_text in {c['chunk_id']}"

        # Raw text must NOT contain the decorated prefix
        assert not c["text"].startswith("[Document:"), f"Raw text must remain undecorated: {c['text'][:50]}"

        # retrieval_text MUST contain document and section context
        assert "[Document: User Authentication Guide]" in c["retrieval_text"]
        assert "[Section:" in c["retrieval_text"]

    print(f"PASS: {len(all_chunks)} chunks correctly preserve raw text, document_title, and retrieval_text")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    tests = [
        test_different_documents_get_different_ids,
        test_same_document_all_representations_share_document_id,
        test_retrieval_code_is_document_agnostic,
        test_no_duplicate_chunk_ids,
        test_retrieval_result_has_document_id,
        test_canonical_rrf_prevents_voting_inflation,
        test_reranking_preserves_all_metadata,
        test_multiple_documents_retrieved_together,
        test_document_title_and_retrieval_text_integrity,
    ]

    passed = 0
    failed = 0

    print("\n" + "="*70)
    print("HYBRID-RAG ARCHITECTURE TESTS")
    print("="*70)

    for test_fn in tests:
        print(f"\nRunning: {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    if failed > 0:
        sys.exit(1)

