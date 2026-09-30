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
