"""Filtering for positional candidate-ID alignment."""

from pragmata.core.types import CandidateItemT


def filter_aligned_candidate_ids(
    items: list[CandidateItemT],
    expected_candidate_ids: list[str],
) -> list[CandidateItemT]:
    """Filter items by positional candidate-ID agreement.

    Args:
        items: Generated items carrying ``candidate_id`` attributes.
        expected_candidate_ids: Ordered candidate IDs expected for the current
            stage output.

    Returns:
        A filtered list containing only positionally aligned items.
    """
    kept_items: list[CandidateItemT] = []

    for expected_candidate_id, item in zip(expected_candidate_ids, items, strict=False):
        if item.candidate_id == expected_candidate_id:
            kept_items.append(item)

    return kept_items
