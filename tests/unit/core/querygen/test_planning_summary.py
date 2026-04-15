"""Tests for the synthetic query-generation stage-1 planning summary executor."""

import hashlib
import json
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from pragmata.core.querygen.llm import LlmInitializationError
from pragmata.core.querygen.planning_summary import (
    PlanningSummaryStageError,
    _build_planning_summary_prompt_vars,
    _format_prior_summary_state,
    _serialize_spec_content,
    fingerprint_querygen_spec,
    run_planning_summary,
)
from pragmata.core.querygen.prompts import (
    SYSTEM_PROMPT_PLANNING_SUMMARY,
    USER_PROMPT_PLANNING_SUMMARY,
)
from pragmata.core.schemas.querygen_input import QueryGenSpec
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_summary import PlanningSummaryState
from pragmata.core.settings.querygen_settings import LlmSettings


@pytest.fixture()
def make_spec() -> Callable[..., QueryGenSpec]:
    def _make_spec(
        *,
        domains: object = "education policy",
        roles: object = "policy analyst",
        languages: object = "en",
        topics: object = "teacher shortages",
        intents: object = "find evidence",
        tasks: object = "literature search",
        difficulty: object = "medium",
        formats: object = "bullet list",
        disallowed_topics: list[str] | None = None,
    ) -> QueryGenSpec:
        return QueryGenSpec.model_validate(
            {
                "domain_context": {
                    "domains": domains,
                    "roles": roles,
                    "languages": languages,
                },
                "knowledge_scope": {
                    "topics": topics,
                },
                "scenario": {
                    "intents": intents,
                    "tasks": tasks,
                    "difficulty": difficulty,
                },
                "format_requests": {
                    "formats": formats,
                },
                "safety": {
                    "disallowed_topics": disallowed_topics,
                },
            }
        )

    return _make_spec


@pytest.fixture()
def expected_default_payload() -> dict[str, object]:
    return {
        "domain_context": {
            "domains": [{"value": "education policy", "weight": 1.0}],
            "roles": [{"value": "policy analyst", "weight": 1.0}],
            "languages": [{"value": "en", "weight": 1.0}],
        },
        "knowledge_scope": {
            "topics": [{"value": "teacher shortages", "weight": 1.0}],
        },
        "scenario": {
            "intents": [{"value": "find evidence", "weight": 1.0}],
            "tasks": [{"value": "literature search", "weight": 1.0}],
            "difficulty": [{"value": "medium", "weight": 1.0}],
        },
        "format_requests": {
            "formats": [{"value": "bullet list", "weight": 1.0}],
        },
        "safety": {
            "disallowed_topics": None,
        },
    }


@pytest.fixture()
def llm_settings() -> LlmSettings:
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


@pytest.fixture()
def prior_summary_state() -> PlanningSummaryState:
    return PlanningSummaryState(
        redundancy_patterns="Coverage-letter clarification scenarios recur.",
        diversification_targets="Add more comparison and multilingual scenarios.",
        coverage_notes="Basic benefits lookup appears well covered.",
    )


def _make_blueprint(candidate_id: str = "C001") -> QueryBlueprint:
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain="education policy",
        role="policy analyst",
        language="en",
        topic="teacher shortages",
        intent="find evidence",
        task="literature search",
        difficulty="medium",
        format="bullet list",
        user_scenario=f"Scenario for {candidate_id}",
        information_need=f"Information need for {candidate_id}",
    )


def test_serialize_spec_content_returns_expected_canonical_json(
    make_spec: Callable[..., QueryGenSpec],
    expected_default_payload: dict[str, object],
) -> None:
    spec = make_spec()

    expected_serialized = json.dumps(
        expected_default_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert _serialize_spec_content(spec) == expected_serialized


def test_serialize_spec_content_round_trips_to_expected_payload(
    make_spec: Callable[..., QueryGenSpec],
    expected_default_payload: dict[str, object],
) -> None:
    spec = make_spec()

    serialized = _serialize_spec_content(spec)

    assert isinstance(serialized, str)
    assert json.loads(serialized) == expected_default_payload


def test_fingerprint_querygen_spec_is_stable_across_repeated_calls(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec(disallowed_topics=["medical advice"])

    fingerprint = fingerprint_querygen_spec(spec)

    assert fingerprint == fingerprint_querygen_spec(spec)
    assert fingerprint == fingerprint_querygen_spec(spec)
    assert len(fingerprint) == 64
    assert all(char in "0123456789abcdef" for char in fingerprint)


@pytest.mark.parametrize(
    ("field_name", "base_value", "changed_value"),
    [
        ("domains", "education policy", "health policy"),
        ("roles", "policy analyst", "school principal"),
        ("languages", "en", "de"),
        ("topics", "teacher shortages", "school meals"),
        ("intents", "find evidence", "compare options"),
        ("tasks", "literature search", "summarization"),
        ("difficulty", "medium", "hard"),
        ("formats", "bullet list", "table"),
        ("disallowed_topics", ["medical advice"], ["legal advice"]),
    ],
)
def test_fingerprint_querygen_spec_changes_when_any_field_value_changes(
    make_spec: Callable[..., QueryGenSpec],
    field_name: str,
    base_value: object,
    changed_value: object,
) -> None:
    kwargs_base = {field_name: base_value}
    kwargs_changed = {field_name: changed_value}

    spec_a = make_spec(**kwargs_base)
    spec_b = make_spec(**kwargs_changed)

    assert fingerprint_querygen_spec(spec_a) != fingerprint_querygen_spec(spec_b)


def test_fingerprint_querygen_spec_matches_for_equivalent_canonicalized_inputs() -> None:
    scalar_spec = QueryGenSpec.model_validate(
        {
            "domain_context": {
                "domains": "education policy",
                "roles": "policy analyst",
                "languages": "en",
            },
            "knowledge_scope": {
                "topics": "teacher shortages",
            },
            "scenario": {
                "intents": "find evidence",
                "tasks": "literature search",
                "difficulty": "medium",
            },
            "format_requests": {
                "formats": "bullet list",
            },
            "safety": {
                "disallowed_topics": ["medical advice"],
            },
        }
    )

    weighted_spec = QueryGenSpec.model_validate(
        {
            "domain_context": {
                "domains": [{"value": "education policy", "weight": 1.0}],
                "roles": [{"value": "policy analyst", "weight": 1.0}],
                "languages": [{"value": "en", "weight": 1.0}],
            },
            "knowledge_scope": {
                "topics": [{"value": "teacher shortages", "weight": 1.0}],
            },
            "scenario": {
                "intents": [{"value": "find evidence", "weight": 1.0}],
                "tasks": [{"value": "literature search", "weight": 1.0}],
                "difficulty": [{"value": "medium", "weight": 1.0}],
            },
            "format_requests": {
                "formats": [{"value": "bullet list", "weight": 1.0}],
            },
            "safety": {
                "disallowed_topics": ["medical advice"],
            },
        }
    )

    assert scalar_spec == weighted_spec
    assert fingerprint_querygen_spec(scalar_spec) == fingerprint_querygen_spec(weighted_spec)


def test_fingerprint_querygen_spec_matches_sha256_of_serialized_content(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec(disallowed_topics=["medical advice"])

    serialized = _serialize_spec_content(spec)
    expected_fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    assert fingerprint_querygen_spec(spec) == expected_fingerprint


def test_format_prior_summary_state_returns_deterministic_multiline_block(
    prior_summary_state: PlanningSummaryState,
) -> None:
    assert _format_prior_summary_state(prior_summary_state) == (
        "- redundancy_patterns:\n"
        "  Coverage-letter clarification scenarios recur.\n"
        "- diversification_targets:\n"
        "  Add more comparison and multilingual scenarios.\n"
        "- coverage_notes:\n"
        "  Basic benefits lookup appears well covered."
    )


def test_build_planning_summary_prompt_vars_formats_spec_candidates_and_prior_summary(
    make_spec: Callable[..., QueryGenSpec],
    prior_summary_state: PlanningSummaryState,
) -> None:
    spec = make_spec(
        domains=[
            {"value": "education policy", "weight": 0.7},
            {"value": "health policy", "weight": 0.3},
        ],
        roles="policy analyst",
        languages=["en", "de"],
        topics=[
            {"value": "teacher shortages", "weight": 0.6},
            {"value": "school funding", "weight": 0.4},
        ],
        intents=["find evidence", "compare options"],
        tasks=[
            {"value": "literature search", "weight": 0.25},
            {"value": "summarization", "weight": 0.75},
        ],
        difficulty="medium",
        formats=["bullet list", "table"],
        disallowed_topics=["medical advice", "legal advice"],
    )

    result = _build_planning_summary_prompt_vars(
        spec=spec,
        candidates=[_make_blueprint("C001"), _make_blueprint("C002")],
        prior_summary_state=prior_summary_state,
    )

    assert result == {
        "domains": "education policy (weight=0.7), health policy (weight=0.3)",
        "roles": "policy analyst (weight=1)",
        "languages": "en (weight=0.5), de (weight=0.5)",
        "topics": "teacher shortages (weight=0.6), school funding (weight=0.4)",
        "intents": "find evidence (weight=0.5), compare options (weight=0.5)",
        "tasks": "literature search (weight=0.25), summarization (weight=0.75)",
        "difficulty": "medium (weight=1)",
        "formats": "bullet list (weight=0.5), table (weight=0.5)",
        "disallowed_topics": "medical advice, legal advice",
        "prior_planning_summary": (
            "- redundancy_patterns:\n"
            "  Coverage-letter clarification scenarios recur.\n"
            "- diversification_targets:\n"
            "  Add more comparison and multilingual scenarios.\n"
            "- coverage_notes:\n"
            "  Basic benefits lookup appears well covered."
        ),
        "query_blueprints": (
            "- candidate_id: C001\n"
            "  domain: education policy\n"
            "  role: policy analyst\n"
            "  language: en\n"
            "  topic: teacher shortages\n"
            "  intent: find evidence\n"
            "  task: literature search\n"
            "  difficulty: medium\n"
            "  format: bullet list\n"
            "  user_scenario: Scenario for C001\n"
            "  information_need: Information need for C001\n\n"
            "- candidate_id: C002\n"
            "  domain: education policy\n"
            "  role: policy analyst\n"
            "  language: en\n"
            "  topic: teacher shortages\n"
            "  intent: find evidence\n"
            "  task: literature search\n"
            "  difficulty: medium\n"
            "  format: bullet list\n"
            "  user_scenario: Scenario for C002\n"
            "  information_need: Information need for C002"
        ),
    }


def test_build_planning_summary_prompt_vars_uses_fallback_when_prior_summary_absent(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec(
        difficulty=None,
        formats=None,
        disallowed_topics=None,
    )

    result = _build_planning_summary_prompt_vars(
        spec=spec,
        candidates=[_make_blueprint("C001")],
        prior_summary_state=None,
    )

    assert result["difficulty"] == "Not specified"
    assert result["formats"] == "Not specified"
    assert result["disallowed_topics"] == "Not specified"
    assert result["prior_planning_summary"] == "No prior planning summary available yet."


def test_build_planning_summary_prompt_vars_rejects_empty_candidates(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec()

    with pytest.raises(ValueError, match="candidates must not be empty"):
        _build_planning_summary_prompt_vars(
            spec=spec,
            candidates=[],
            prior_summary_state=None,
        )


def test_build_planning_summary_prompt_vars_returns_exact_placeholder_mapping(
    make_spec: Callable[..., QueryGenSpec],
) -> None:
    spec = make_spec()

    result = _build_planning_summary_prompt_vars(
        spec=spec,
        candidates=[_make_blueprint("C001")],
        prior_summary_state=None,
    )

    assert set(result) == {
        "domains",
        "roles",
        "languages",
        "topics",
        "intents",
        "tasks",
        "difficulty",
        "formats",
        "disallowed_topics",
        "prior_planning_summary",
        "query_blueprints",
    }


def test_run_planning_summary_wires_summary_prompt_assets_and_settings_into_llm_builder(
    monkeypatch: pytest.MonkeyPatch,
    make_spec: Callable[..., QueryGenSpec],
    llm_settings: LlmSettings,
    prior_summary_state: PlanningSummaryState,
) -> None:
    spec = make_spec()
    candidates = [_make_blueprint("C001")]

    expected_prompt_vars = _build_planning_summary_prompt_vars(
        spec=spec,
        candidates=candidates,
        prior_summary_state=prior_summary_state,
    )

    expected_summary = PlanningSummaryState(
        redundancy_patterns="Repeated evidence-seeking framing appears.",
        diversification_targets="Increase variation in scenario framing.",
        coverage_notes="Core teacher-shortage lookup is already covered.",
    )

    mock_runnable = Mock()
    mock_runnable.invoke.return_value = expected_summary

    build_llm_runnable_mock = Mock(return_value=mock_runnable)
    monkeypatch.setattr(
        "pragmata.core.querygen.planning_summary.build_llm_runnable",
        build_llm_runnable_mock,
    )

    result = run_planning_summary(
        spec=spec,
        candidates=candidates,
        llm_settings=llm_settings,
        api_key="test-api-key",
        prior_summary_state=prior_summary_state,
    )

    assert result == expected_summary

    build_llm_runnable_mock.assert_called_once_with(
        system_text=SYSTEM_PROMPT_PLANNING_SUMMARY,
        user_text=USER_PROMPT_PLANNING_SUMMARY,
        model_provider=llm_settings.model_provider,
        model=llm_settings.planning_model,
        api_key="test-api-key",
        output_schema=PlanningSummaryState,
        requests_per_second=llm_settings.requests_per_second,
        check_every_n_seconds=llm_settings.check_every_n_seconds,
        max_bucket_size=llm_settings.max_bucket_size,
        base_url=llm_settings.base_url,
        model_kwargs=llm_settings.model_kwargs,
    )
    mock_runnable.invoke.assert_called_once_with(expected_prompt_vars)


def test_run_planning_summary_invokes_runnable_once_and_returns_summary_state(
    monkeypatch: pytest.MonkeyPatch,
    make_spec: Callable[..., QueryGenSpec],
    llm_settings: LlmSettings,
) -> None:
    spec = make_spec()
    candidates = [_make_blueprint("C001"), _make_blueprint("C002")]
    expected_prompt_vars = _build_planning_summary_prompt_vars(
        spec=spec,
        candidates=candidates,
        prior_summary_state=None,
    )

    expected_summary = PlanningSummaryState(
        redundancy_patterns="Repeated baseline lookup patterns appear.",
        diversification_targets="Use more decision-oriented information needs.",
        coverage_notes="Basic factual retrieval seems adequately covered.",
    )

    class FakeRunnable:
        def __init__(self) -> None:
            self.seen_payload: dict[str, object] | None = None
            self.invoke_calls = 0

        def invoke(self, payload: dict[str, object]) -> PlanningSummaryState:
            self.invoke_calls += 1
            self.seen_payload = payload
            return expected_summary

    fake_runnable = FakeRunnable()

    monkeypatch.setattr(
        "pragmata.core.querygen.planning_summary.build_llm_runnable",
        lambda **_: fake_runnable,
    )

    result = run_planning_summary(
        spec=spec,
        candidates=candidates,
        llm_settings=llm_settings,
        api_key="test-api-key",
    )

    assert result == expected_summary
    assert isinstance(result, PlanningSummaryState)
    assert fake_runnable.invoke_calls == 1
    assert fake_runnable.seen_payload == expected_prompt_vars


def test_run_planning_summary_propagates_llm_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
    make_spec: Callable[..., QueryGenSpec],
    llm_settings: LlmSettings,
) -> None:
    spec = make_spec()

    monkeypatch.setattr(
        "pragmata.core.querygen.planning_summary.build_llm_runnable",
        Mock(side_effect=LlmInitializationError("bad config")),
    )

    with pytest.raises(LlmInitializationError, match="bad config"):
        run_planning_summary(
            spec=spec,
            candidates=[_make_blueprint("C001")],
            llm_settings=llm_settings,
            api_key="test-api-key",
        )


def test_run_planning_summary_wraps_invoke_failures(
    monkeypatch: pytest.MonkeyPatch,
    make_spec: Callable[..., QueryGenSpec],
    llm_settings: LlmSettings,
) -> None:
    spec = make_spec()

    class FakeRunnable:
        def invoke(self, payload: dict[str, object]) -> PlanningSummaryState:
            raise RuntimeError("provider failure")

    monkeypatch.setattr(
        "pragmata.core.querygen.planning_summary.build_llm_runnable",
        lambda **_: FakeRunnable(),
    )

    with pytest.raises(
        PlanningSummaryStageError,
        match="Planning stage invocation failed while updating the planning summary.",
    ):
        run_planning_summary(
            spec=spec,
            candidates=[_make_blueprint("C001")],
            llm_settings=llm_settings,
            api_key="test-api-key",
        )


def test_run_planning_summary_propagates_empty_candidates_error(
    make_spec: Callable[..., QueryGenSpec],
    llm_settings: LlmSettings,
) -> None:
    spec = make_spec()

    with pytest.raises(ValueError, match="candidates must not be empty"):
        run_planning_summary(
            spec=spec,
            candidates=[],
            llm_settings=llm_settings,
            api_key="test-api-key",
        )
