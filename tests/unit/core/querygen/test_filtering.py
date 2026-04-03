"""Tests for deterministic positional candidate-ID alignment filtering."""

import pytest

from pragmata.core.querygen.filtering import filter_aligned_candidate_ids
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery


def _blueprint(candidate_id: str) -> QueryBlueprint:
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain="healthcare",
        role="patient",
        language="en",
        topic="insurance coverage",
        intent="understand benefits",
        task="ask a question",
        difficulty=None,
        format=None,
        user_scenario="I need to understand whether a treatment is covered.",
        information_need="What coverage rules apply to my treatment?",
    )


def _realized_query(candidate_id: str) -> RealizedQuery:
    return RealizedQuery(
        candidate_id=candidate_id,
        query=f"query for {candidate_id}",
    )


@pytest.mark.parametrize(
    ("item_candidate_ids", "expected_candidate_ids", "kept_candidate_ids"),
    [
        (["c001", "c002", "c003"], ["c001", "c002", "c003"], ["c001", "c002", "c003"]),
        (["c001", "c999", "c003"], ["c001", "c002", "c003"], ["c001", "c003"]),
        (["c002", "c001", "c003"], ["c001", "c002", "c003"], ["c003"]),
        (["c001", "c002"], ["c001", "c002", "c003"], ["c001", "c002"]),
        (["c001", "c002", "c003"], ["c001", "c002"], ["c001", "c002"]),
        ([], ["c001", "c002"], []),
    ],
)
def test_filter_aligned_candidate_ids_for_query_blueprints(
    item_candidate_ids: list[str],
    expected_candidate_ids: list[str],
    kept_candidate_ids: list[str],
) -> None:
    items = [_blueprint(candidate_id) for candidate_id in item_candidate_ids]

    result = filter_aligned_candidate_ids(
        items=items,
        expected_candidate_ids=expected_candidate_ids,
    )

    assert [item.candidate_id for item in result] == kept_candidate_ids


def test_filter_aligned_candidate_ids_supports_realized_queries() -> None:
    items = [
        _realized_query("c001"),
        _realized_query("c999"),
        _realized_query("c003"),
    ]

    result = filter_aligned_candidate_ids(
        items=items,
        expected_candidate_ids=["c001", "c002", "c003"],
    )

    assert [item.candidate_id for item in result] == ["c001", "c003"]
    assert result == [items[0], items[2]]
