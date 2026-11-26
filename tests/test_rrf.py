from src.core import rrf


def test_rrf_returns_highest_ranked_first():
    result = rrf([["A", "B"], ["B", "A"]], k=60)
    assert result[0][0] == "A" or result[0][0] == "B"
    assert len(result) == 2


def test_rrf_weight_changes_score():
    result = rrf([["BM25", "X"], ["VECTOR", "X"]], weights=[0.7, 0.3])
    scores = dict(result)
    assert scores["BM25"] > scores["VECTOR"]


def test_rrf_rejects_mismatched_weights():
    try:
        rrf([["A"]], weights=[0.5, 0.5])
    except ValueError:
        return
    assert False, "Expected ValueError"
