import pytest
from src.evaluation.metrics import RetrievalMetrics

def test_precision_recall_at_k():
    relevance = [1, 0, 1, 0, 0]
    p_at_5 = RetrievalMetrics.precision_at_k(relevance, k=5)
    assert p_at_5 == 2.0 / 5.0

    r_at_5 = RetrievalMetrics.recall_at_k(relevance, total_relevant=4, k=5)
    assert r_at_5 == 2.0 / 4.0

def test_reciprocal_rank():
    assert RetrievalMetrics.reciprocal_rank([0, 1, 0]) == 0.5
    assert RetrievalMetrics.reciprocal_rank([1, 0, 0]) == 1.0
    assert RetrievalMetrics.reciprocal_rank([0, 0, 0]) == 0.0

def test_ndcg_at_k():
    # Perfect ranking
    score_perfect = RetrievalMetrics.ndcg_at_k([1, 1, 0, 0], k=4)
    assert score_perfect == 1.0

    # Suboptimal ranking
    score_sub = RetrievalMetrics.ndcg_at_k([0, 1, 0, 0], k=4)
    assert 0.0 < score_sub < 1.0

def test_calculate_faithfulness():
    evidence = ["Global warming causes extreme heatwaves across South Asia."]
    answer = "Global warming causes extreme heatwaves."
    faith = RetrievalMetrics.calculate_faithfulness(answer, evidence)
    assert faith > 0.8

