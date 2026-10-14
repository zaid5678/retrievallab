from src.retrievallab.evaluation import (
    base_control,
    first_relevant_rank,
    retrieval_metrics,
)


RESULTS = [
    {"control_ids": ["SC-12(2)"]},
    {"control_ids": ["MP-6", "MP-7"]},
]


def test_base_control_strips_enhancement_number():
    assert base_control("SC-12(2)") == "SC-12"
    assert base_control("SC-12") == "SC-12"


def test_first_relevant_rank_strict_and_base_matching():
    assert first_relevant_rank(RESULTS, ["SC-12"]) is None
    assert first_relevant_rank(RESULTS, ["SC-12"], match_base=True) == 1
    assert first_relevant_rank(RESULTS, ["MP-7"]) == 2


def test_retrieval_metrics():
    metrics = retrieval_metrics([1, 2, None, 6], k=5)

    assert metrics["hit_rate_at_1"] == 0.25
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == (1 + 0.5) / 4
