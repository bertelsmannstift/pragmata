"""Unit tests for synthetic query output assembly."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pragmata.core.querygen.assembly import (
    _build_query_ids,
    assemble_planning_summary,
    assemble_queries_meta,
    assemble_query_rows,
)
from pragmata.core.querygen.planning_memory import fingerprint_querygen_spec
from pragmata.core.schemas.querygen_input import QueryGenSpec
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery
from pragmata.core.schemas.querygen_summary import PlanningSummaryState


def _make_blueprint(
    candidate_id: str,
    *,
    domain: str = "foundation publications",
    role: str = "program officer",
    language: str = "en",
    topic: str = "education outcomes",
    intent: str = "understand",
    task: str = "summarize",
    difficulty: str | None = None,
    format: str | None = None,
    user_scenario: str = "I am preparing for an internal briefing.",
    information_need: str = "I need a concise overview of the main findings.",
) -> QueryBlueprint:
    """Build a valid QueryBlueprint for tests."""
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain=domain,
        role=role,
        language=language,
        topic=topic,
        intent=intent,
        task=task,
        difficulty=difficulty,
        format=format,
        user_scenario=user_scenario,
        information_need=information_need,
    )


def _make_realized_query(
    candidate_id: str,
    query: str,
) -> RealizedQuery:
    """Build a valid RealizedQuery for tests."""
    return RealizedQuery(
        candidate_id=candidate_id,
        query=query,
    )


def _make_spec() -> QueryGenSpec:
    """Build a valid QueryGenSpec for planning-summary assembly tests."""
    return QueryGenSpec(
        domain_context={
            "domains": "foundation research",
            "roles": "program officer",
            "languages": "en",
        },
        knowledge_scope={
            "topics": "education outcomes",
        },
        scenario={
            "intents": "understand",
            "tasks": "summarize",
            "difficulty": "basic",
        },
        format_requests={
            "formats": "bullet list",
        },
        safety={
            "disallowed_topics": ["medical diagnosis"],
        },
    )


def _make_planning_summary_state() -> PlanningSummaryState:
    """Build a valid PlanningSummaryState for assembly tests."""
    return PlanningSummaryState(
        redundancy_patterns="Repeated briefing-style education-overview requests for internal staff.",
        diversification_targets="Add more distinct user situations and information needs within the same spec.",
        coverage_notes="English foundation-research summaries on education outcomes are already well covered.",
    )


def test_build_query_ids_returns_deterministic_run_scoped_ids() -> None:
    """_build_query_ids should generate deterministic final query IDs."""
    assert _build_query_ids(run_id="run123", n_queries=3) == [
        "run123_q1",
        "run123_q2",
        "run123_q3",
    ]


def test_build_query_ids_returns_empty_list_for_zero_queries() -> None:
    """_build_query_ids should return an empty list when no queries are requested."""
    assert _build_query_ids(run_id="run123", n_queries=0) == []


def test_assemble_query_rows_joins_realized_queries_to_blueprints() -> None:
    """assemble_query_rows should join stage-2 queries to stage-1 metadata by candidate_id."""
    blueprints = [
        _make_blueprint(
            "c001",
            domain="foundation research",
            role="research associate",
            language="en",
            topic="school climate",
            intent="understand",
            task="summarize",
            difficulty="medium",
            format="bullet list",
        ),
        _make_blueprint(
            "c002",
            domain="foundation strategy",
            role="program manager",
            language="de",
            topic="youth participation",
            intent="compare",
            task="analyze",
            difficulty=None,
            format=None,
        ),
    ]
    realized_queries = [
        _make_realized_query(
            "c001",
            "Summarize the main findings on school climate as a bullet list.",
        ),
        _make_realized_query(
            "c002",
            "Vergleiche die wichtigsten Erkenntnisse zur Jugendbeteiligung.",
        ),
    ]

    rows = assemble_query_rows(
        blueprints=blueprints,
        realized_queries=realized_queries,
        run_id="run123",
    )

    assert [row.query_id for row in rows] == ["run123_q1", "run123_q2"]
    assert [row.query for row in rows] == [
        "Summarize the main findings on school climate as a bullet list.",
        "Vergleiche die wichtigsten Erkenntnisse zur Jugendbeteiligung.",
    ]

    assert rows[0].model_dump(exclude={"query_id", "query"}) == {
        "domain": "foundation research",
        "role": "research associate",
        "language": "en",
        "topic": "school climate",
        "intent": "understand",
        "task": "summarize",
        "difficulty": "medium",
        "format": "bullet list",
    }
    assert rows[1].model_dump(exclude={"query_id", "query"}) == {
        "domain": "foundation strategy",
        "role": "program manager",
        "language": "de",
        "topic": "youth participation",
        "intent": "compare",
        "task": "analyze",
        "difficulty": None,
        "format": None,
    }


def test_assemble_query_rows_preserves_realized_query_order() -> None:
    """assemble_query_rows should preserve the order of filtered stage-2 realized queries."""
    blueprints = [
        _make_blueprint("c001", topic="teacher retention"),
        _make_blueprint("c002", topic="digital literacy"),
    ]
    realized_queries = [
        _make_realized_query("c002", "Compare digital literacy interventions."),
        _make_realized_query("c001", "What are the main issues in teacher retention?"),
    ]

    rows = assemble_query_rows(
        blueprints=blueprints,
        realized_queries=realized_queries,
        run_id="run123",
    )

    assert [row.model_dump(include={"query_id", "query", "topic"}) for row in rows] == [
        {
            "query_id": "run123_q1",
            "query": "Compare digital literacy interventions.",
            "topic": "digital literacy",
        },
        {
            "query_id": "run123_q2",
            "query": "What are the main issues in teacher retention?",
            "topic": "teacher retention",
        },
    ]


def test_assemble_query_rows_returns_empty_list_for_empty_realized_queries() -> None:
    """assemble_query_rows should return an empty list when no realized queries are present."""
    blueprints = [_make_blueprint("c001")]

    rows = assemble_query_rows(
        blueprints=blueprints,
        realized_queries=[],
        run_id="run123",
    )

    assert rows == []


def test_assemble_queries_meta_builds_dataset_level_metadata() -> None:
    """assemble_queries_meta should construct validated dataset-level metadata."""
    meta = assemble_queries_meta(
        run_id="run123",
        n_requested_queries=50,
        n_returned_queries=37,
        model_provider="mistralai",
        planning_model="magistral-medium-latest",
        realization_model="mistral-medium-latest",
    )

    assert meta.model_dump(exclude={"created_at"}) == {
        "run_id": "run123",
        "n_requested_queries": 50,
        "n_returned_queries": 37,
        "model_provider": "mistralai",
        "planning_model": "magistral-medium-latest",
        "realization_model": "mistral-medium-latest",
    }
    assert isinstance(meta.created_at, datetime)
    assert meta.created_at.tzinfo == UTC


def test_assemble_queries_meta_stamps_created_at_in_utc_now_range() -> None:
    """assemble_queries_meta should stamp created_at internally at construction time."""
    before = datetime.now(UTC)

    meta = assemble_queries_meta(
        run_id="run123",
        n_requested_queries=10,
        n_returned_queries=8,
        model_provider="mistralai",
        planning_model="plan-model",
        realization_model="realize-model",
    )

    after = datetime.now(UTC)

    assert before <= meta.created_at <= after


def test_assemble_planning_summary_builds_artifact_from_spec_run_id_and_state() -> None:
    """assemble_planning_summary should construct a validated planning-summary artifact."""
    spec = _make_spec()
    state = _make_planning_summary_state()

    artifact = assemble_planning_summary(
        spec=spec,
        run_id="run123",
        state=state,
    )

    assert artifact.model_dump(exclude={"created_at"}) == {
        "spec_fingerprint": fingerprint_querygen_spec(spec),
        "source_run_id": "run123",
        "state": state.model_dump(mode="json"),
    }
    assert isinstance(artifact.created_at, datetime)
    assert artifact.created_at.tzinfo == UTC


def test_assemble_planning_summary_stamps_created_at_in_utc_now_range() -> None:
    """assemble_planning_summary should stamp created_at internally at construction time."""
    spec = _make_spec()
    state = _make_planning_summary_state()
    before = datetime.now(UTC)

    artifact = assemble_planning_summary(
        spec=spec,
        run_id="run123",
        state=state,
    )

    after = datetime.now(UTC)

    assert before <= artifact.created_at <= after


def test_assemble_queries_meta_rejects_non_positive_requested_queries() -> None:
    """assemble_queries_meta should surface schema validation for invalid requested counts."""
    with pytest.raises(ValidationError):
        assemble_queries_meta(
            run_id="run123",
            n_requested_queries=0,
            n_returned_queries=0,
            model_provider="mistralai",
            planning_model="plan-model",
            realization_model="realize-model",
        )


def test_assemble_queries_meta_rejects_negative_returned_queries() -> None:
    """assemble_queries_meta should surface schema validation for invalid returned counts."""
    with pytest.raises(ValidationError):
        assemble_queries_meta(
            run_id="run123",
            n_requested_queries=10,
            n_returned_queries=-1,
            model_provider="mistralai",
            planning_model="plan-model",
            realization_model="realize-model",
        )
