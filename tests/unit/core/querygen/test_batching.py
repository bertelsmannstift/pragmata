"""Unit tests for synthetic query generation batching helpers."""

import pytest

from pragmata.core.querygen.batching import (
    build_candidate_ids,
    chunk_blueprints,
    iter_batch_sizes,
)
from pragmata.core.schemas.querygen_plan import QueryBlueprint


@pytest.fixture()
def blueprint_list() -> list[QueryBlueprint]:
    """Return a small ordered list of query blueprints."""
    return [
        QueryBlueprint.model_validate(
            {
                "candidate_id": f"c{i:03d}",
                "domain": "education policy",
                "role": "policy analyst",
                "language": "en",
                "topic": f"topic-{i}",
                "intent": "find evidence",
                "task": "literature search",
                "difficulty": "medium",
                "format": "bullet list",
                "user_scenario": f"Scenario {i}",
                "information_need": f"Need {i}",
            }
        )
        for i in range(1, 6)
    ]


def test_build_candidate_ids_returns_deterministic_ordered_unique_ids() -> None:
    candidate_ids = build_candidate_ids(5)

    assert candidate_ids == ["c001", "c002", "c003", "c004", "c005"]
    assert len(candidate_ids) == 5
    assert len(set(candidate_ids)) == 5


def test_build_candidate_ids_scales_width_for_large_runs() -> None:
    candidate_ids = build_candidate_ids(1000)

    assert candidate_ids[0] == "c0001"
    assert candidate_ids[-1] == "c1000"


@pytest.mark.parametrize(
    ("n_queries", "batch_size", "expected"),
    [
        (6, 3, [3, 3]),
        (7, 3, [3, 3, 1]),
        (2, 5, [2]),
        (0, 5, []),
    ],
)
def test_iter_batch_sizes_yields_expected_batches(
    n_queries: int,
    batch_size: int,
    expected: list[int],
) -> None:
    assert list(iter_batch_sizes(n_queries, batch_size)) == expected


def test_chunk_blueprints_preserves_input_order(
    blueprint_list: list[QueryBlueprint],
) -> None:
    chunks = list(chunk_blueprints(blueprint_list, 2))

    assert [[bp.candidate_id for bp in chunk] for chunk in chunks] == [
        ["c001", "c002"],
        ["c003", "c004"],
        ["c005"],
    ]


def test_chunk_blueprints_returns_no_chunks_for_empty_input() -> None:
    assert list(chunk_blueprints([], 3)) == []


def test_chunk_blueprints_rejects_non_positive_chunk_size(
    blueprint_list: list[QueryBlueprint],
) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        list(chunk_blueprints(blueprint_list, 0))


def test_build_candidate_ids_returns_empty_list_for_zero_queries() -> None:
    assert build_candidate_ids(0) == []


def test_build_candidate_ids_rejects_negative_query_count() -> None:
    with pytest.raises(ValueError, match="n_queries must be non-negative"):
        build_candidate_ids(-1)


def test_iter_batch_sizes_rejects_negative_query_count() -> None:
    with pytest.raises(ValueError, match="n_queries must be non-negative"):
        list(iter_batch_sizes(-1, 3))


def test_iter_batch_sizes_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        list(iter_batch_sizes(5, 0))


def test_chunk_blueprints_rejects_negative_chunk_size(
    blueprint_list: list[QueryBlueprint],
) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        list(chunk_blueprints(blueprint_list, -1))
