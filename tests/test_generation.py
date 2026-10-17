"""
test_generation.py -- Unit tests for LLM Answer Generation and Context Builder.

Verifies:
1. Context builder creates correctly formatted context.
2. Context builder preserves metadata (document_id, title, section, heading_path, chunk_id).
3. Multiple documents can be represented simultaneously in context.
4. Empty retrieval results are handled safely (graceful fallback answer).
5. Generation configuration is read from src/config.py.
6. The generator receives query + retrieved context.
7. Mocked LLM generates structured answer + source attribution without downloading 8B weights.
8. Non-thinking mode output cleaning.
9. HybridRAG answer() and query(generate_answer=True) integration with mocked generator.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.generation.context_builder import (
    build_context,
    build_context_string,
    extract_source_metadata,
    extract_sources,
    filter_candidates_by_query_scope,
    format_source_entry,
)
from src.generation.llm_generator import (
    NO_CONTEXT_FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    QwenGenerator,
    clean_non_thinking_output,
    generate_answer,
)


def _create_sample_candidates():
    """Create sample multi-document candidates with rich metadata."""
    return [
        {
            "document_id": "doc_pmay_u",
            "document_title": "Pradhan Mantri Awas Yojana - Urban",
            "source_file": "pmay_u.txt",
            "chunk_id": "doc_pmay_u__naive__0",
            "source_chunk_id": "doc_pmay_u__naive__0",
            "retrieval_type": "naive",
            "section": "ELIGIBILITY",
            "heading_path": ["Guidelines", "Eligibility"],
            "retrieval_sources": ["dense", "bm25"],
            "rrf_score": 0.085,
            "rerank_score": 0.92,
            "text": "Beneficiaries of EWS and LIG categories are eligible under PMAY-U.",
        },
        {
            "document_id": "doc_pmay_g",
            "document_title": "Pradhan Mantri Awas Yojana - Gramin",
            "source_file": "pmay_g.txt",
            "chunk_id": "doc_pmay_g__sw__1",
            "source_chunk_id": "doc_pmay_g__sw__1",
            "retrieval_type": "sentence_window",
            "section": "BENEFITS",
            "heading_path": ["Scheme Overview", "Financial Assistance"],
            "retrieval_sources": ["dense"],
            "rrf_score": 0.072,
            "rerank_score": 0.85,
            "text": "Financial assistance of Rs. 1.20 lakh in plains and Rs. 1.30 lakh in hilly states.",
        },
        {
            "document_id": "doc_pmjay",
            "document_title": "Ayushman Bharat PM-JAY",
            "source_file": "pm_jay.txt",
            "chunk_id": "doc_pmjay__pc__2",
            "source_chunk_id": "doc_pmjay__pc__2",
            "retrieval_type": "parent_child",
            "section": "COVERAGE",
            "heading_path": ["Benefits", "Coverage Amount"],
            "retrieval_sources": ["bm25"],
            "rrf_score": 0.065,
            "rerank_score": 0.78,
            "text": "Health cover of Rs. 5 lakh per family per year for secondary and tertiary care hospitalization.",
        },
    ]


# ---------------------------------------------------------------------------
# Test 1: Context builder creates correctly formatted context
# ---------------------------------------------------------------------------
def test_1_context_builder_formatting():
    candidates = _create_sample_candidates()
    context_str, sources = build_context(candidates, top_k=2)

    assert "[Source 1]" in context_str
    assert "[Source 2]" in context_str
    assert "[Source 3]" not in context_str  # top_k=2 respected

    assert "Document: Pradhan Mantri Awas Yojana - Urban" in context_str
    assert "Section: Guidelines > Eligibility" in context_str
    assert "Source File: pmay_u.txt" in context_str
    assert "Beneficiaries of EWS and LIG categories" in context_str
    print("PASS: Test 1 - Context builder creates correctly formatted context")


# ---------------------------------------------------------------------------
# Test 2: Context builder preserves metadata
# ---------------------------------------------------------------------------
def test_2_context_builder_preserves_metadata():
    candidates = _create_sample_candidates()
    _, sources = build_context(candidates, top_k=3)

    assert len(sources) == 3

    # Check first source
    s1 = sources[0]
    assert s1["source_index"] == 1
    assert s1["document_id"] == "doc_pmay_u"
    assert s1["document_title"] == "Pradhan Mantri Awas Yojana - Urban"
    assert s1["source_file"] == "pmay_u.txt"
    assert s1["section"] == "ELIGIBILITY"
    assert s1["heading_path"] == ["Guidelines", "Eligibility"]
    assert s1["chunk_id"] == "doc_pmay_u__naive__0"
    assert s1["retrieval_type"] == "naive"
    assert s1["rerank_score"] == 0.92

    # Check second source
    s2 = sources[1]
    assert s2["source_index"] == 2
    assert s2["document_id"] == "doc_pmay_g"
    assert s2["chunk_id"] == "doc_pmay_g__sw__1"
    print("PASS: Test 2 - Context builder preserves metadata")


# ---------------------------------------------------------------------------
# Test 3: Multiple documents can be represented
# ---------------------------------------------------------------------------
def test_3_multiple_documents_represented():
    candidates = _create_sample_candidates()
    context_str, sources = build_context(candidates, top_k=3)

    # Check that distinct document titles appear
    doc_titles = [s["document_title"] for s in sources]
    assert "Pradhan Mantri Awas Yojana - Urban" in doc_titles
    assert "Pradhan Mantri Awas Yojana - Gramin" in doc_titles
    assert "Ayushman Bharat PM-JAY" in doc_titles

    # Verify distinct document IDs
    doc_ids = {s["document_id"] for s in sources}
    assert len(doc_ids) == 3
    print("PASS: Test 3 - Multiple documents represented simultaneously")


# ---------------------------------------------------------------------------
# Test 4: Empty retrieval results are handled safely
# ---------------------------------------------------------------------------
def test_4_empty_retrieval_results_handled_safely():
    # Empty list
    context_str, sources = build_context([], top_k=5)
    assert context_str == ""
    assert sources == []

    # None
    context_str_none, sources_none = build_context(None, top_k=5)
    assert context_str_none == ""
    assert sources_none == []

    # Generator behavior with empty context
    gen = QwenGenerator()
    res = gen.generate(query="What are the benefits?", candidates=[])
    assert res["answer"] == NO_CONTEXT_FALLBACK_ANSWER
    assert res["sources"] == []

    res_empty_ctx = gen.generate(query="What are the benefits?", context="")
    assert res_empty_ctx["answer"] == NO_CONTEXT_FALLBACK_ANSWER
    assert res_empty_ctx["sources"] == []
    print("PASS: Test 4 - Empty retrieval results handled safely")


# ---------------------------------------------------------------------------
# Test 5: Generation configuration is read from src/config.py
# ---------------------------------------------------------------------------
def test_5_generation_configuration_from_config():
    assert hasattr(config, "LLM_MODEL_NAME")
    assert config.LLM_MODEL_NAME == "Qwen/Qwen3-0.6B"

    assert hasattr(config, "LLM_TEMPERATURE")
    assert config.LLM_TEMPERATURE == 0.1

    assert hasattr(config, "LLM_MAX_NEW_TOKENS")
    assert config.LLM_MAX_NEW_TOKENS == 512

    assert hasattr(config, "LLM_ENABLE_THINKING")
    assert config.LLM_ENABLE_THINKING is False

    assert hasattr(config, "GENERATION_CONTEXT_TOP_K")
    assert config.GENERATION_CONTEXT_TOP_K == 5

    # Verify generator instantiates defaults from config
    gen = QwenGenerator()
    assert gen.model_name == "Qwen/Qwen3-0.6B"
    assert gen.temperature == 0.1
    assert gen.max_new_tokens == 512
    assert gen.enable_thinking is False
    print("PASS: Test 5 - Generation configuration read from src/config.py")


# ---------------------------------------------------------------------------
# Test 6: The generator receives query + retrieved context
# ---------------------------------------------------------------------------
def test_6_generator_receives_query_and_context():
    candidates = _create_sample_candidates()
    query = "Who is eligible for PMAY-U?"

    # Mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<mock_prompt>"
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 102]])}
    mock_tokenizer.decode.return_value = "EWS and LIG beneficiaries are eligible."

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[101, 102, 201, 202]])

    gen = QwenGenerator(
        tokenizer=mock_tokenizer,
        model=mock_model,
        device="cpu",
    )

    result = gen.generate(query=query, candidates=candidates, top_k=2)

    # Verify tokenizer was called with the prompt containing query and context
    assert mock_tokenizer.apply_chat_template.called
    template_call_args = mock_tokenizer.apply_chat_template.call_args
    messages = template_call_args[0][0]

    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]
    assert "Beneficiaries of EWS and LIG" in messages[1]["content"]

    # Verify returned result
    assert result["answer"] == "EWS and LIG beneficiaries are eligible."
    assert len(result["sources"]) == 2
    assert result["sources"][0]["document_id"] == "doc_pmay_u"
    print("PASS: Test 6 - Generator receives query + retrieved context")


# ---------------------------------------------------------------------------
# Test 7: Non-thinking mode cleans any reasoning blocks
# ---------------------------------------------------------------------------
def test_7_non_thinking_mode_cleaning():
    raw_with_thinking = "<think>\nThinking about eligibility rules...\n</think>\nEligible categories are EWS and LIG."
    cleaned = clean_non_thinking_output(raw_with_thinking)
    assert cleaned == "Eligible categories are EWS and LIG."
    assert "<think>" not in cleaned

    # Already clean text
    plain = "This is a direct answer."
    assert clean_non_thinking_output(plain) == plain

    # Stripping appended out-of-scope disclaimer when answer is already present
    with_disclaimer = (
        "Under PM-KISAN, eligible farmers receive Rs. 6,000 per year in three equal installments.\n\n"
        "I'm sorry, I can only answer questions related to government welfare schemes covered in my knowledge base. "
        "Your question does not appear to be in scope. Please ask about a scheme such as PMAY, PM-KISAN, APY, PM SVANidhi, Stand-Up India, or PM Vishwakarma."
    )
    cleaned_disc = clean_non_thinking_output(with_disclaimer)
    assert "Under PM-KISAN" in cleaned_disc
    assert "I'm sorry" not in cleaned_disc

    # Stripping dummy comparison rows
    with_dummy = "### Key Differences\n| Scheme | Benefit |\n| PM-KISAN | Rs 6,000 |\n| Other Schemes | Not specified | Not mentioned |"
    cleaned_dummy = clean_non_thinking_output(with_dummy)
    assert "Other Schemes" not in cleaned_dummy

    print("PASS: Test 7 - Non-thinking mode guarantees clean answer and strips extraneous disclaimers/dummy rows")



# ---------------------------------------------------------------------------
# Test 8: HybridRAG answer() and query() integration with generator
# ---------------------------------------------------------------------------
def test_8_hybrid_rag_integration_with_generator():
    from retrieve import HybridRAG

    candidates = _create_sample_candidates()

    mock_generator = MagicMock()
    mock_generator.generate.return_value = {
        "answer": "PMAY-U eligibility includes EWS families.",
        "sources": [
            {
                "source_index": 1,
                "document_id": "doc_pmay_u",
                "source_file": "pmay_u.txt",
                "section": "ELIGIBILITY",
                "chunk_id": "doc_pmay_u__naive__0",
            }
        ],
    }

    # Patch retrieve_dense, BM25, and rerank to avoid running full retrieval models in unit test
    with patch("retrieve.retrieve_dense", return_value=candidates[:1]), \
         patch("retrieve.rerank", return_value=candidates[:1]), \
         patch.object(HybridRAG, "__init__", lambda self: None):
        rag = HybridRAG()
        rag.naive_index, rag.naive_meta = None, []
        rag.sw_index, rag.sw_meta = None, []
        rag.pc_index, rag.pc_meta = None, []
        rag.embedding_info = {}
        rag.bm25 = MagicMock()
        rag.bm25.retrieve.return_value = []
        rag.embedding_model = config.EMBEDDING_MODEL_NAME
        rag.reranker_model = config.RERANKER_MODEL_NAME
        rag._generator = mock_generator

        # 1. Calling query(generate_answer=False) preserves pure retrieval output
        res_no_gen = rag.query("Who is eligible?", generate_answer=False, enable_reformulation=False)
        assert "final_results" in res_no_gen
        assert "answer" not in res_no_gen
        assert "sources" not in res_no_gen

        # 2. Calling query(generate_answer=True) includes answer and sources
        res_with_gen = rag.query(
            "Who is eligible?",
            generate_answer=True,
            generator=mock_generator,
            enable_reformulation=False,
        )
        assert "answer" in res_with_gen
        assert "sources" in res_with_gen
        assert res_with_gen["answer"] == "PMAY-U eligibility includes EWS families."
        assert len(res_with_gen["sources"]) == 1

        # 3. Calling answer() convenience method
        res_answer = rag.answer(
            "Who is eligible?",
            generator=mock_generator,
            enable_reformulation=False,
        )
        assert "answer" in res_answer
        assert "sources" in res_answer
        assert res_answer["answer"] == "PMAY-U eligibility includes EWS families."

    print("PASS: Test 8 - HybridRAG generation integration verified")


# ---------------------------------------------------------------------------
# Test 9: Query-scope filtering excludes unrelated scheme chunks
# ---------------------------------------------------------------------------
def test_9_query_scope_filtering_excludes_unrelated_schemes():
    """When query asks about PM-KISAN, SVANidhi / PMAY chunks must be pruned."""
    candidates = [
        {
            "document_id": "doc_pmkisan",
            "document_title": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
            "chunk_id": "doc_pmkisan__naive__0",
            "rerank_score": -7.5,
            "text": "A financial benefit of Rs. 6,000 per annum is provided to eligible farmer families.",
        },
        {
            "document_id": "doc_svanidhi",
            "document_title": "PM Street Vendor's AtmaNirbhar Nidhi (PM SVANidhi)",
            "chunk_id": "doc_svanidhi__naive__0",
            "rerank_score": -8.2,
            "text": "Financial assistance of up to Rs. 10,000 is provided to eligible street vendors.",
        },
        {
            "document_id": "doc_pmay_u",
            "document_title": "Pradhan Mantri Awas Yojana - Urban",
            "chunk_id": "doc_pmay_u__naive__1",
            "rerank_score": -8.9,
            "text": "Housing subsidy under PMAY-U for EWS families.",
        },
    ]

    # Query specifically about PM-KISAN
    query = "What is the annual financial benefit provided under PM-KISAN and how is it paid?"
    filtered = filter_candidates_by_query_scope(candidates, query)

    doc_titles = [c["document_title"] for c in filtered]
    assert any("kisan" in t.lower() or "pm-kisan" in t.lower() for t in doc_titles), \
        "PM-KISAN document should be kept"
    assert not any("svanidhi" in t.lower() for t in doc_titles), \
        "SVANidhi should be pruned for a PM-KISAN query"
    assert not any("awas" in t.lower() or "pmay" in t.lower() for t in doc_titles), \
        "PMAY should be pruned for a PM-KISAN query"

    # Verify build_context with query param also prunes correctly
    context_str, sources = build_context(candidates, top_k=5, query=query)
    assert "SVANidhi" not in context_str, "SVANidhi content must not appear in context"
    assert "street vendor" not in context_str.lower(), "Street vendor content must not appear"
    assert "kisan" in context_str.lower() or "pm-kisan" in context_str.lower(), \
        "PM-KISAN content should be present in filtered context"

    # Broad question with no specific scheme → all candidates kept
    broad_query = "What welfare schemes are available in India?"
    broad_filtered = filter_candidates_by_query_scope(candidates, broad_query)
    assert len(broad_filtered) == len(candidates), \
        "Broad query should not filter any candidates"

    print("PASS: Test 9 - Query-scope filtering prunes unrelated scheme chunks")


# ---------------------------------------------------------------------------
# Test 10: Single-scheme query with 'and' does not trigger comparison prompt
# ---------------------------------------------------------------------------
def test_10_single_scheme_query_no_false_comparison():
    gen = QwenGenerator()
    query = "What is the annual financial benefit provided under PM-KISAN and how is it paid?"
    context = "[Source 1] Document: PM-KISAN\nRs. 6,000 per year in 3 installments."
    prompt = gen.build_prompt(query, context)

    # Should NOT contain multi-scheme comparison instructions or Key Differences
    assert "Key Differences" not in prompt, "Single-scheme query with 'and' must not trigger Key Differences instruction"
    assert "Other Schemes" not in prompt
    assert "Concise & Direct" in prompt
    print("PASS: Test 10 - Single-scheme query does not trigger comparison instructions")


if __name__ == "__main__":
    test_1_context_builder_formatting()
    test_2_context_builder_preserves_metadata()
    test_3_multiple_documents_represented()
    test_4_empty_retrieval_results_handled_safely()
    test_5_generation_configuration_from_config()
    test_6_generator_receives_query_and_context()
    test_7_non_thinking_mode_cleaning()
    test_8_hybrid_rag_integration_with_generator()
    test_9_query_scope_filtering_excludes_unrelated_schemes()
    test_10_single_scheme_query_no_false_comparison()
