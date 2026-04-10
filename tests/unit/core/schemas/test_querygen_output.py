"""Unit tests for the synthetic query generation output contract."""

from datetime import UTC, datetime, timezone

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.querygen_output import PlanningMemoryArtifact, SyntheticQueriesMeta, SyntheticQueryRow


@pytest.fixture()
def valid_row_payload() -> dict[str, str | None]:
    """Return a minimal valid synthetic query row payload."""
    return {
        "query_id": "q_001",
        "query": "How do I apply for housing support?",
        "domain": "public_policy",
        "role": "citizen",
        "language": "en",
        "topic": "housing",
        "intent": "information",
        "task": "eligibility",
        "difficulty": "easy",
        "format": "steps",
    }


@pytest.fixture()
def valid_meta_payload() -> dict[str, object]:
    """Return a valid synthetic query metadata payload."""
    return {
        "run_id": "run_20260309_001",
        "created_at": "2026-03-09T10:30:00Z",
        "n_requested_queries": 25,
        "n_returned_queries": 21,
        "model_provider": "openai",
        "planning_model": "gpt-4.1-mini",
        "realization_model": "gpt-4.1-mini",
    }


@pytest.fixture()
def valid_planning_memory_payload() -> dict[str, object]:
    """Return a valid planning-memory artifact payload."""
    return {
        "spec_fingerprint": "9d8b6d94d8f3a4b74e9e2cb7d5d4a3a2a6d4f7b5b2f8e3c1d6a9b7c4e2f1a0d3",
        "source_run_id": "run_20260309_001",
        "created_at": "2026-03-09T10:45:00Z",
        "state": {
            "redundancy_patterns": (
                "Avoid repeating candidate blueprints centered on first-time housing-benefit "
                "eligibility checks with near-identical applicant situations."
            ),
            "diversification_targets": (
                "Favor candidate blueprints about document preparation, appeal procedures, "
                "and deadline clarification with distinct scenarios and information needs."
            ),
            "coverage_notes": (
                "Basic housing-support eligibility scenarios have already appeared and should "
                "not be revisited too closely in the next planning batch."
            ),
        },
    }


def test_synthetic_query_row_accepts_valid_payload(valid_row_payload: dict[str, str | None]) -> None:
    """SyntheticQueryRow validates a complete row payload."""
    row = SyntheticQueryRow.model_validate(valid_row_payload)
    assert row.query_id == "q_001"
    assert row.query == "How do I apply for housing support?"
    assert row.language == "en"


@pytest.mark.parametrize("field_name", ["query_id", "query"])
def test_synthetic_query_row_rejects_blank_required_fields(
    valid_row_payload: dict[str, str | None], field_name: str
) -> None:
    """Required row fields reject blank strings."""
    payload = dict(valid_row_payload)
    payload[field_name] = "   "
    with pytest.raises(ValidationError):
        SyntheticQueryRow.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ["domain", "role", "language", "topic", "intent", "task", "difficulty", "format"],
)
def test_synthetic_query_row_optional_fields_accept_none(
    valid_row_payload: dict[str, str | None], field_name: str
) -> None:
    """Optional metadata fields accept None."""
    payload = dict(valid_row_payload)
    payload[field_name] = None
    row = SyntheticQueryRow.model_validate(payload)
    assert getattr(row, field_name) is None


def test_synthetic_query_row_rejects_extra_keys(valid_row_payload: dict[str, str | None]) -> None:
    """SyntheticQueryRow rejects unexpected fields."""
    payload = dict(valid_row_payload)
    payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        SyntheticQueryRow.model_validate(payload)


def test_synthetic_query_row_column_order_matches_model_field_order() -> None:
    """CSV column order is derived from model_fields keys."""
    assert list(SyntheticQueryRow.model_fields.keys()) == [
        "query_id",
        "query",
        "domain",
        "role",
        "language",
        "topic",
        "intent",
        "task",
        "difficulty",
        "format",
    ]


def test_synthetic_queries_meta_accepts_valid_payload(valid_meta_payload: dict[str, object]) -> None:
    """SyntheticQueriesMeta validates a complete metadata payload."""
    meta = SyntheticQueriesMeta.model_validate(valid_meta_payload)
    assert meta.run_id == "run_20260309_001"
    assert meta.created_at == datetime(2026, 3, 9, 10, 30, tzinfo=timezone.utc)
    assert meta.n_requested_queries == 25
    assert meta.n_returned_queries == 21
    assert meta.model_provider == "openai"
    assert meta.planning_model == "gpt-4.1-mini"
    assert meta.realization_model == "gpt-4.1-mini"


def test_synthetic_queries_meta_rejects_non_positive_n_queries(valid_meta_payload: dict[str, object]) -> None:
    """n_requested_queries must be strictly positive."""
    payload = dict(valid_meta_payload)
    payload["n_requested_queries"] = 0
    with pytest.raises(ValidationError):
        SyntheticQueriesMeta.model_validate(payload)


def test_synthetic_queries_meta_rejects_negative_n_returned_queries(
    valid_meta_payload: dict[str, object],
) -> None:
    """n_returned_queries must be non-negative."""
    payload = dict(valid_meta_payload)
    payload["n_returned_queries"] = -1
    with pytest.raises(ValidationError):
        SyntheticQueriesMeta.model_validate(payload)


def test_synthetic_queries_meta_rejects_extra_keys(valid_meta_payload: dict[str, object]) -> None:
    """SyntheticQueriesMeta rejects unexpected fields."""
    payload = dict(valid_meta_payload)
    payload["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        SyntheticQueriesMeta.model_validate(payload)

    meta = SyntheticQueriesMeta(
        run_id="run_123",
        created_at=datetime.now(UTC),
        n_requested_queries=5,
        n_returned_queries=0,
        model_provider="mistralai",
        planning_model="magistral-medium-latest",
        realization_model="mistral-medium-latest",
    )

    assert meta.n_requested_queries == 5
    assert meta.n_returned_queries == 0


def test_synthetic_queries_meta_rejects_returned_queries_above_requested() -> None:
    with pytest.raises(ValidationError, match="n_returned_queries must be less than or equal to n_requested_queries"):
        SyntheticQueriesMeta(
            run_id="run_123",
            created_at=datetime.now(UTC),
            n_requested_queries=5,
            n_returned_queries=6,
            model_provider="mistralai",
            planning_model="magistral-medium-latest",
            realization_model="mistral-medium-latest",
        )


def test_planning_memory_artifact_accepts_valid_payload(
    valid_planning_memory_payload: dict[str, object],
) -> None:
    """PlanningMemoryArtifact validates a complete nested payload."""
    artifact = PlanningMemoryArtifact.model_validate(valid_planning_memory_payload)

    assert artifact.spec_fingerprint.startswith("9d8b6d94")
    assert artifact.source_run_id == "run_20260309_001"
    assert artifact.created_at == datetime(2026, 3, 9, 10, 45, tzinfo=timezone.utc)
    assert artifact.state.redundancy_patterns.startswith("Avoid repeating")
    assert artifact.state.diversification_targets.startswith("Favor candidate blueprints")
    assert artifact.state.coverage_notes.startswith("Basic housing-support")


def test_planning_memory_artifact_rejects_extra_keys(
    valid_planning_memory_payload: dict[str, object],
) -> None:
    """PlanningMemoryArtifact rejects unexpected top-level fields."""
    payload = dict(valid_planning_memory_payload)
    payload["unexpected"] = "boom"

    with pytest.raises(ValidationError):
        PlanningMemoryArtifact.model_validate(payload)


def test_planning_memory_artifact_rejects_extra_keys_in_nested_state(
    valid_planning_memory_payload: dict[str, object],
) -> None:
    """PlanningMemoryArtifact rejects unexpected nested state fields."""
    payload = dict(valid_planning_memory_payload)
    payload["state"] = dict(payload["state"])
    payload["state"]["unexpected"] = "boom"

    with pytest.raises(ValidationError):
        PlanningMemoryArtifact.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ["redundancy_patterns", "diversification_targets", "coverage_notes"],
)
def test_planning_memory_artifact_validates_nested_state_field_lengths(
    valid_planning_memory_payload: dict[str, object],
    field_name: str,
) -> None:
    """Nested PlanningSummaryState field constraints are enforced through the artifact."""
    payload = dict(valid_planning_memory_payload)
    payload["state"] = dict(payload["state"])
    payload["state"][field_name] = ""

    with pytest.raises(ValidationError):
        PlanningMemoryArtifact.model_validate(payload)


def test_planning_memory_artifact_rejects_missing_nested_state_field(
    valid_planning_memory_payload: dict[str, object],
) -> None:
    """Nested PlanningSummaryState requires all summary fields."""
    payload = dict(valid_planning_memory_payload)
    payload["state"] = dict(payload["state"])
    payload["state"].pop("coverage_notes")

    with pytest.raises(ValidationError):
        PlanningMemoryArtifact.model_validate(payload)
