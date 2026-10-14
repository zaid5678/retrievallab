import re


ENHANCEMENT_SUFFIX = re.compile(r"\(\d+\)$")


def base_control(control_id: str) -> str:
    """Map an enhancement such as 'SC-12(2)' to its base control 'SC-12'."""
    return ENHANCEMENT_SUFFIX.sub("", control_id)


def first_relevant_rank(
    results: list[dict],
    expected_ids: list[str],
    match_base: bool = False,
) -> int | None:
    """Return the 1-based rank of the first chunk covering an expected control.

    With match_base=True, an enhancement and its base control count as the
    same control, so retrieving 'SC-12(2)' satisfies an expected 'SC-12'.
    """
    normalise = base_control if match_base else (lambda control_id: control_id)
    expected = {normalise(control_id) for control_id in expected_ids}

    for rank, result in enumerate(results, start=1):
        retrieved = {normalise(control_id) for control_id in result["control_ids"]}
        if expected & retrieved:
            return rank

    return None


def retrieval_metrics(ranks: list[int | None], k: int) -> dict:
    """Compute Hit Rate@1, Recall@k, and MRR@k from first-relevant ranks."""
    if not ranks:
        return {"hit_rate_at_1": 0.0, f"recall_at_{k}": 0.0, "mrr": 0.0}

    found = [rank for rank in ranks if rank is not None and rank <= k]

    return {
        "hit_rate_at_1": sum(rank == 1 for rank in found) / len(ranks),
        f"recall_at_{k}": len(found) / len(ranks),
        "mrr": sum(1 / rank for rank in found) / len(ranks),
    }
