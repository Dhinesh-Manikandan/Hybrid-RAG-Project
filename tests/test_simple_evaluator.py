import pytest
from src.evaluation.simple_evaluator import SimpleEvaluator

@pytest.fixture(scope="module")
def evaluator():
    return SimpleEvaluator()

def test_correctness_calculation(evaluator):
    query = "What is the capital of France?"
    ref = "The capital of France is Paris."
    
    # Very similar
    res1 = evaluator.evaluate(query, "Paris is the capital of France.", ref, "Paris is the capital of France.")
    assert res1["correctness"]["score"] >= 4
    
    # Completely wrong
    res2 = evaluator.evaluate(query, "Elephants are large mammals native to Africa.", ref, "Paris is the capital of France.")
    assert res2["correctness"]["score"] < 4

def test_relevancy_calculation(evaluator):
    query = "What is the capital of France?"
    
    res = evaluator.evaluate(query, "Paris is the capital of France.", "The capital of France is Paris.", "Paris is the capital of France.")
    assert 0.0 <= res["relevancy"]["score"] <= 1.0
    assert res["relevancy"]["score"] > 0.2 # Scaled contextualized sim should be > 0.2 for short sentences

def test_faithfulness_fully_supported(evaluator):
    context = "The sky is blue. The grass is green."
    response = "The sky is blue. The grass is green."
    res = evaluator.evaluate("query", response, "ref", context)
    assert res["faithfulness"]["score"] == 1.0
    assert res["faithfulness"]["supported_claims"] == 2
    assert res["faithfulness"]["total_claims"] == 2

def test_faithfulness_partially_supported(evaluator):
    context = "The sky is blue."
    response = "The sky is blue. The grass is green."
    res = evaluator.evaluate("query", response, "ref", context)
    assert res["faithfulness"]["score"] == 0.5
    assert res["faithfulness"]["supported_claims"] == 1
    assert res["faithfulness"]["total_claims"] == 2

def test_faithfulness_unsupported(evaluator):
    context = "The sun is hot."
    response = "The sky is blue. The grass is green."
    res = evaluator.evaluate("query", response, "ref", context)
    assert res["faithfulness"]["score"] < 0.5
    assert res["faithfulness"]["supported_claims"] == 0
    assert res["faithfulness"]["total_claims"] == 2

def test_empty_answer(evaluator):
    res = evaluator.evaluate("query", "", "ref", "context")
    assert res["correctness"]["score"] == 1
    assert res["relevancy"]["score"] == 0.0
    assert res["faithfulness"]["score"] == 0.0

def test_empty_context(evaluator):
    res = evaluator.evaluate("query", "The sky is blue.", "ref", "")
    assert res["faithfulness"]["score"] == 0.0
    assert res["faithfulness"]["total_claims"] == 1

def test_score_ranges(evaluator):
    res = evaluator.evaluate("test", "test", "test", "test")
    assert 1 <= res["correctness"]["score"] <= 5
    assert 0.0 <= res["relevancy"]["score"] <= 1.0
    assert 0.0 <= res["faithfulness"]["score"] <= 1.0

def test_evaluate_strategies(evaluator):
    strategies = {
        "Naive RAG": {
            "answer": "Paris is the capital city of France.",
            "context": "Paris is the capital of France and its largest city."
        },
        "Hybrid RAG": {
            "answer": "The capital of France is Paris.",
            "context": "Paris is the capital of France. It is located in Europe."
        }
    }
    query = "What is the capital of France?"
    ref = "The capital of France is Paris."
    res = evaluator.evaluate_strategies(query, strategies, reference_answer=ref)
    
    assert "strategies" in res
    assert "Naive RAG" in res["strategies"]
    assert "Hybrid RAG" in res["strategies"]
    assert res["strategies"]["Hybrid RAG"]["evaluation"]["faithfulness"]["score"] == 1.0
    assert res["strategies"]["Hybrid RAG"]["evaluation"]["correctness"]["score"] >= 4
    assert len(res["summary_table"]) == 2

def test_find_reference_match(evaluator):
    # Semantic match for APY query
    matched = evaluator.find_reference_match("What is the age limit for Atal Pension Yojana?")
    if matched:
        assert "expected_answer" in matched
        assert "18" in matched["expected_answer"] or "40" in matched["expected_answer"]


def test_faithfulness_comparative_compound_claims(evaluator):
    context = (
        "Pradhan Mantri Awas Yojana - Urban (PMAY-U) addresses urban housing shortage among the EWS/LIG "
        "and MIG categories including the slum dwellers by ensuring a pucca house.\n\n"
        "Pradhan Mantri Awaas Yojana - Gramin (PMAY-G) prioritises houseless households and households "
        "living in zero, one or two room houses with kutcha wall and kutcha roof."
    )
    ans = (
        "PMAY-U and PMAY-G differ in their focus and eligibility criteria:\n\n"
        "- Beneficiaries: PMAY-U targets urban households, including slum dwellers, while PMAY-G prioritizes houseless households in zero, one, or two-room houses.\n"
        "- Housing Support: PMAY-U offers pucca houses and supports slum rehabilitation, whereas PMAY-G provides housing and prioritization based on household status and room count."
    )
    res = evaluator.evaluate(
        query="Compare PMAY-U and PMAY-G in terms of beneficiaries and housing support.",
        generated_answer=ans,
        retrieved_context=context,
    )
    assert res["faithfulness"]["score"] >= 0.75
    assert res["faithfulness"]["supported_claims"] >= 3
    assert res["relevancy"]["score"] >= 0.80


def test_evaluate_strategies_reuses_precalculated_eval(evaluator):
    pre_eval = {
        "correctness": {"score": 5, "similarity": 0.95, "reference_available": True},
        "relevancy": {"score": 0.88, "raw_similarity": 0.72},
        "faithfulness": {"score": 0.82, "supported_claims": 27, "total_claims": 33},
    }
    strategies = {
        "Hybrid RAG": {
            "answer": "This is already answered.",
            "context": "Context text.",
            "evaluation": pre_eval,
        }
    }
    res = evaluator.evaluate_strategies("test query", strategies)
    assert res["strategies"]["Hybrid RAG"]["evaluation"] == pre_eval
    assert res["summary_table"][0]["Strategy"] == "Hybrid RAG"
    assert res["summary_table"][0]["Correctness"] == "5/5"
    assert res["summary_table"][0]["Relevancy"] == 0.88
    assert res["summary_table"][0]["Faithfulness"] == 0.82
    assert res["summary_table"][0]["Supported Claims"] == "27/33"



