"""Tests for the synthetic query-generation stage-1 planning executor."""

from unittest.mock import Mock

import pytest

from pragmata.core.querygen.llm import LlmInitializationError
from pragmata.core.querygen.planning import (
    PlanningStageError,
    _build_planning_prompt_vars,
    _format_planning_summary,
    _format_planning_summary_task_context,
    run_planning_stage
)
from pragmata.core.schemas.querygen_summary import PlanningSummaryState
from pragmata.core.querygen.prompts import SYSTEM_PROMPT_PLANNING, USER_PROMPT_PLANNING
from pragmata.core.schemas.querygen_input import QueryGenSpec
from pragmata.core.schemas.querygen_plan import QueryBlueprint, QueryBlueprintList
from pragmata.core.settings.querygen_settings import LlmSettings


@pytest.fixture
def querygen_spec() -> QueryGenSpec:
    """Return a representative resolved query-generation spec."""
    return QueryGenSpec.model_validate(
        {
            "domain_context": {
                "domains": [
                    {"value": "healthcare", "weight": 0.7},
                    {"value": "education", "weight": 0.3},
                ],
                "roles": "patient",
                "languages": ["English", "German"],
            },
            "knowledge_scope": {
                "topics": [
                    {"value": "insurance coverage", "weight": 0.6},
                    {"value": "treatment options", "weight": 0.4},
                ]
            },
            "scenario": {
                "intents": ["understand", "compare"],
                "tasks": [
                    {"value": "summarize", "weight": 0.25},
                    {"value": "recommend", "weight": 0.75},
                ],
                "difficulty": "advanced",
            },
            "format_requests": {
                "formats": ["bullet list", "table"],
            },
            "safety": {
                "disallowed_topics": ["self-harm", "hate speech"],
            },
        }
    )


@pytest.fixture
def minimal_querygen_spec() -> QueryGenSpec:
    """Return a spec with optional prompt fields omitted."""
    return QueryGenSpec.model_validate(
        {
            "domain_context": {
                "domains": "healthcare",
                "roles": "patient",
                "languages": "English",
            },
            "knowledge_scope": {
                "topics": "insurance coverage",
            },
            "scenario": {
                "intents": "understand",
                "tasks": "summarize",
                "difficulty": None,
            },
            "format_requests": {
                "formats": None,
            },
            "safety": {
                "disallowed_topics": None,
            },
        }
    )


@pytest.fixture
def planning_summary_state() -> PlanningSummaryState:
    """Return a representative advisory planning summary."""
    return PlanningSummaryState(
        redundancy_patterns="Repeated benefits-letter clarification scenarios for individual patients.",
        diversification_targets="Add more comparison and decision-support scenarios across adjacent service contexts.",
        coverage_notes="Basic insurance coverage lookups already well represented across English requests.",
    )


@pytest.fixture
def llm_settings() -> LlmSettings:
    """Return representative planning-stage LLM settings."""
    return LlmSettings(
        model_provider="mistralai",
        planning_model="magistral-medium-latest",
        realization_model="mistral-medium-latest",
        requests_per_second=2.5,
        check_every_n_seconds=0.2,
        max_bucket_size=3,
        base_url="https://example.invalid/v1",
        model_kwargs={"temperature": 0.2},
    )


@pytest.fixture
def expected_prompt_vars() -> dict[str, object]:
    """Return the expected formatted planning prompt payload for one candidate."""
    return {
        "candidate_ids": "\n    - C001",
        "domains": "healthcare (weight=0.7), education (weight=0.3)",
        "roles": "patient (weight=1)",
        "languages": "English (weight=0.5), German (weight=0.5)",
        "topics": "insurance coverage (weight=0.6), treatment options (weight=0.4)",
        "intents": "understand (weight=0.5), compare (weight=0.5)",
        "tasks": "summarize (weight=0.25), recommend (weight=0.75)",
        "difficulty": "advanced (weight=1)",
        "formats": "bullet list (weight=0.5), table (weight=0.5)",
        "disallowed_topics": "self-harm, hate speech",
        "n_queries": 1,
        "planning_summary": "",
        "planning_summary_task_context": "",
    }


def _make_blueprint(candidate_id: str = "C001") -> QueryBlueprint:
    """Build a valid query blueprint for tests."""
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain="healthcare",
        role="patient",
        language="English",
        topic="insurance coverage",
        intent="understand",
        task="summarize",
        difficulty=None,
        format=None,
        user_scenario="A patient reviews a benefits letter.",
        information_need="Clarify what is covered.",
    )


def test_format_planning_summary_returns_empty_string_when_absent() -> None:
    assert _format_planning_summary(None) == ""


def test_format_planning_summary_formats_state_deterministically(
    planning_summary_state: PlanningSummaryState,
) -> None:
    result = _format_planning_summary(planning_summary_state)

    assert result == (
        "The following prior planning summary is provided as advisory context from earlier planning batches.\n\n"
        "- prior_planning_summary:\n"
        "  - redundancy_patterns: Repeated benefits-letter clarification scenarios for individual patients.\n"
        "  - diversification_targets: Add more comparison and decision-support scenarios across adjacent service contexts.\n"
        "  - coverage_notes: Basic insurance coverage lookups already well represented across English requests.\n\n"
    )


def test_format_planning_summary_task_context_returns_empty_string_when_absent() -> None:
    assert _format_planning_summary_task_context(None) == ""


def test_format_planning_summary_task_context_returns_sentence_when_present(
    planning_summary_state: PlanningSummaryState,
) -> None:
    assert _format_planning_summary_task_context(planning_summary_state) == (
        " Use the planning summary as advisory context for avoiding near-duplicate candidates."
    )


def test_build_planning_prompt_vars_formats_resolved_spec(
    querygen_spec: QueryGenSpec,
) -> None:
    result = _build_planning_prompt_vars(
        spec=querygen_spec,
        batch_candidate_ids=["C001", "C002", "C003"],
    )

    assert result == {
        "candidate_ids": "\n    - C001\n    - C002\n    - C003",
        "domains": "healthcare (weight=0.7), education (weight=0.3)",
        "roles": "patient (weight=1)",
        "languages": "English (weight=0.5), German (weight=0.5)",
        "topics": "insurance coverage (weight=0.6), treatment options (weight=0.4)",
        "intents": "understand (weight=0.5), compare (weight=0.5)",
        "tasks": "summarize (weight=0.25), recommend (weight=0.75)",
        "difficulty": "advanced (weight=1)",
        "formats": "bullet list (weight=0.5), table (weight=0.5)",
        "disallowed_topics": "self-harm, hate speech",
        "n_queries": 3,
        "planning_summary": "",
        "planning_summary_task_context": "",
    }


def test_build_planning_prompt_vars_uses_not_specified_for_missing_optional_fields(
    minimal_querygen_spec: QueryGenSpec,
) -> None:
    result = _build_planning_prompt_vars(
        spec=minimal_querygen_spec,
        batch_candidate_ids=["C001"],
    )

    assert result["difficulty"] == "Not specified"
    assert result["formats"] == "Not specified"
    assert result["disallowed_topics"] == "Not specified"
    assert result["n_queries"] == 1
    assert result["planning_summary"] == ""
    assert result["planning_summary_task_context"] == ""


def test_build_planning_prompt_vars_includes_planning_summary_when_present(
    querygen_spec: QueryGenSpec,
    planning_summary_state: PlanningSummaryState,
) -> None:
    result = _build_planning_prompt_vars(
        spec=querygen_spec,
        batch_candidate_ids=["C001"],
        planning_summary=planning_summary_state,
    )

    assert result["planning_summary"] == (
        "The following prior planning summary is provided as advisory context from earlier planning batches.\n\n"
        "- prior_planning_summary:\n"
        "  - redundancy_patterns: Repeated benefits-letter clarification scenarios for individual patients.\n"
        "  - diversification_targets: Add more comparison and decision-support scenarios across adjacent service contexts.\n"
        "  - coverage_notes: Basic insurance coverage lookups already well represented across English requests.\n\n"
    )
    assert result["planning_summary_task_context"] == (
        " Use the planning summary as advisory context for avoiding near-duplicate candidates."
    )


def test_build_planning_prompt_vars_rejects_empty_candidate_ids(
    querygen_spec: QueryGenSpec,
) -> None:
    with pytest.raises(ValueError, match="batch_candidate_ids must not be empty"):
        _build_planning_prompt_vars(
            spec=querygen_spec,
            batch_candidate_ids=[],
        )


def test_run_planning_stage_wires_planning_assets_and_settings_into_llm_builder(
    monkeypatch: pytest.MonkeyPatch,
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
    expected_prompt_vars: dict[str, object],
) -> None:
    mock_runnable = Mock()
    mock_runnable.invoke.return_value = QueryBlueprintList(candidates=[_make_blueprint()])

    build_llm_runnable_mock = Mock(return_value=mock_runnable)
    monkeypatch.setattr(
        "pragmata.core.querygen.planning.build_llm_runnable",
        build_llm_runnable_mock,
    )

    result = run_planning_stage(
        spec=querygen_spec,
        llm_settings=llm_settings,
        api_key="test-api-key",
        batch_candidate_ids=["C001"],
    )

    assert result == [_make_blueprint()]

    build_llm_runnable_mock.assert_called_once_with(
        system_text=SYSTEM_PROMPT_PLANNING,
        user_text=USER_PROMPT_PLANNING,
        model_provider=llm_settings.model_provider,
        model=llm_settings.planning_model,
        api_key="test-api-key",
        output_schema=QueryBlueprintList,
        requests_per_second=llm_settings.requests_per_second,
        check_every_n_seconds=llm_settings.check_every_n_seconds,
        max_bucket_size=llm_settings.max_bucket_size,
        base_url=llm_settings.base_url,
        model_kwargs=llm_settings.model_kwargs,
    )
    mock_runnable.invoke.assert_called_once_with(expected_prompt_vars)


def test_run_planning_stage_wires_optional_planning_summary_through_to_runnable(
    monkeypatch: pytest.MonkeyPatch,
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
    planning_summary_state: PlanningSummaryState,
) -> None:
    mock_runnable = Mock()
    mock_runnable.invoke.return_value = QueryBlueprintList(candidates=[_make_blueprint()])

    monkeypatch.setattr(
        "pragmata.core.querygen.planning.build_llm_runnable",
        Mock(return_value=mock_runnable),
    )

    run_planning_stage(
        spec=querygen_spec,
        llm_settings=llm_settings,
        api_key="test-api-key",
        batch_candidate_ids=["C001"],
        planning_summary=planning_summary_state,
    )

    mock_runnable.invoke.assert_called_once_with(
        {
            "candidate_ids": "\n    - C001",
            "domains": "healthcare (weight=0.7), education (weight=0.3)",
            "roles": "patient (weight=1)",
            "languages": "English (weight=0.5), German (weight=0.5)",
            "topics": "insurance coverage (weight=0.6), treatment options (weight=0.4)",
            "intents": "understand (weight=0.5), compare (weight=0.5)",
            "tasks": "summarize (weight=0.25), recommend (weight=0.75)",
            "difficulty": "advanced (weight=1)",
            "formats": "bullet list (weight=0.5), table (weight=0.5)",
            "disallowed_topics": "self-harm, hate speech",
            "n_queries": 1,
            "planning_summary": (
                "The following prior planning summary is provided as advisory context from earlier planning batches.\n\n"
                "- prior_planning_summary:\n"
                "  - redundancy_patterns: Repeated benefits-letter clarification scenarios for individual patients.\n"
                "  - diversification_targets: Add more comparison and decision-support scenarios across adjacent service contexts.\n"
                "  - coverage_notes: Basic insurance coverage lookups already well represented across English requests.\n\n"
            ),
            "planning_summary_task_context": (
                " Use the planning summary as advisory context for avoiding near-duplicate candidates."
            ),
        }
    )


def test_run_planning_stage_invokes_runnable_once_and_returns_candidates(
    monkeypatch: pytest.MonkeyPatch,
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
    expected_prompt_vars: dict[str, object],
) -> None:
    class FakeRunnable:
        def __init__(self) -> None:
            self.seen_payload: dict[str, object] | None = None
            self.invoke_calls = 0

        def invoke(self, payload: dict[str, object]) -> QueryBlueprintList:
            self.invoke_calls += 1
            self.seen_payload = payload
            return QueryBlueprintList(candidates=[_make_blueprint()])

    fake_runnable = FakeRunnable()

    monkeypatch.setattr(
        "pragmata.core.querygen.planning.build_llm_runnable",
        lambda **_: fake_runnable,
    )

    result = run_planning_stage(
        spec=querygen_spec,
        llm_settings=llm_settings,
        api_key="test-api-key",
        batch_candidate_ids=["C001"],
    )

    assert result == [_make_blueprint()]
    assert isinstance(result, list)
    assert all(isinstance(item, QueryBlueprint) for item in result)
    assert fake_runnable.invoke_calls == 1
    assert fake_runnable.seen_payload == expected_prompt_vars


def test_run_planning_stage_extracts_candidates_from_query_blueprint_list(
    monkeypatch: pytest.MonkeyPatch,
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
) -> None:
    expected_candidates = [
        _make_blueprint("C001"),
        QueryBlueprint(
            candidate_id="C002",
            domain="education",
            role="patient",
            language="German",
            topic="treatment options",
            intent="compare",
            task="recommend",
            difficulty="advanced",
            format="table",
            user_scenario="A user is comparing options before an appointment.",
            information_need="Compare treatment alternatives.",
        ),
    ]

    class FakeRunnable:
        def invoke(self, payload: dict[str, object]) -> QueryBlueprintList:
            return QueryBlueprintList(candidates=expected_candidates)

    monkeypatch.setattr(
        "pragmata.core.querygen.planning.build_llm_runnable",
        lambda **_: FakeRunnable(),
    )

    result = run_planning_stage(
        spec=querygen_spec,
        llm_settings=llm_settings,
        api_key="test-api-key",
        batch_candidate_ids=["C001", "C002"],
    )

    assert result == expected_candidates


def test_run_planning_stage_propagates_llm_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
) -> None:
    monkeypatch.setattr(
        "pragmata.core.querygen.planning.build_llm_runnable",
        Mock(side_effect=LlmInitializationError("bad config")),
    )

    with pytest.raises(LlmInitializationError, match="bad config"):
        run_planning_stage(
            spec=querygen_spec,
            llm_settings=llm_settings,
            api_key="test-api-key",
            batch_candidate_ids=["C001"],
        )


def test_run_planning_stage_wraps_invoke_failures(
    monkeypatch: pytest.MonkeyPatch,
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
) -> None:
    class FakeRunnable:
        def invoke(self, payload: dict[str, object]) -> QueryBlueprintList:
            raise RuntimeError("provider failure")

    monkeypatch.setattr(
        "pragmata.core.querygen.planning.build_llm_runnable",
        lambda **_: FakeRunnable(),
    )

    with pytest.raises(PlanningStageError, match="Planning stage invocation failed."):
        run_planning_stage(
            spec=querygen_spec,
            llm_settings=llm_settings,
            api_key="test-api-key",
            batch_candidate_ids=["C001"],
        )


def test_run_planning_stage_propagates_empty_batch_error(
    querygen_spec: QueryGenSpec,
    llm_settings: LlmSettings,
) -> None:
    with pytest.raises(ValueError, match="batch_candidate_ids must not be empty"):
        run_planning_stage(
            spec=querygen_spec,
            llm_settings=llm_settings,
            api_key="test-api-key",
            batch_candidate_ids=[],
        )


def test_build_planning_prompt_vars_returns_exact_placeholder_mapping(
    querygen_spec: QueryGenSpec,
) -> None:
    result = _build_planning_prompt_vars(
        spec=querygen_spec,
        batch_candidate_ids=["C001"],
    )

    assert set(result) == {
        "candidate_ids",
        "domains",
        "roles",
        "languages",
        "topics",
        "intents",
        "tasks",
        "difficulty",
        "formats",
        "disallowed_topics",
        "n_queries",
        "planning_summary",
        "planning_summary_task_context",
    }
