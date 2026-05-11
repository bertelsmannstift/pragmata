"""Filtering for candidate-ID membership alignment."""

from pragmata.core.types import CandidateItemT


def filter_aligned_candidate_ids(
    items: list[CandidateItemT],
    expected_candidate_ids: list[str],
) -> list[CandidateItemT]:
    """Filter generated items by candidate-ID membership.

    The first returned item for each expected candidate_id is kept. Unexpected
    candidate IDs and later duplicates are ignored. Returned items follow the
    expected candidate-ID order.
    """
    expected_ids = set(expected_candidate_ids)
    first_item_by_id: dict[str, CandidateItemT] = {}

    for item in items:
        candidate_id = item.candidate_id

        if candidate_id not in expected_ids:
            continue

        if candidate_id in first_item_by_id:
            continue

        first_item_by_id[candidate_id] = item

    return [
        first_item_by_id[candidate_id] for candidate_id in expected_candidate_ids if candidate_id in first_item_by_id
    ]
