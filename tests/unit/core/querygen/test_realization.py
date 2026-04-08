"""Tests for the synthetic query-generation stage-2 realization executor."""

from unittest.mock import Mock

import pytest

from pragmata.core.querygen.llm import LlmInitializationError
from pragmata.core.querygen.prompts import SYSTEM_PROMPT_REALIZATION, USER_PROMPT_REALIZATION
from pragmata.core.querygen.realization import (
    RealizationStageError,
    _build_realization_prompt_vars,
    run_realization_stage,
)
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery, RealizedQueryList
from pragmata.core.settings.querygen_settings import LlmSettings


@pytest.fixture
def llm_settings() -> LlmSettings:
    """Return representative realization-stage LLM settings."""
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


def _make_blueprint_with_optional_fields(candidate_id: str = "C002") -> QueryBlueprint:
    """Build a valid query blueprint including optional fields."""
    return QueryBlueprint(
        candidate_id=candidate_id,
        domain="education",
        role="student",
        language="German",
        topic="scholarship eligibility",
        intent="compare",
        task="recommend",
        difficulty="advanced",
        format="table",
        user_scenario="A student compares financial aid options before applying.",
        information_need="Compare eligibility criteria for available scholarships.",
    )


def _make_realized_query(candidate_id: str = "C001") -> RealizedQuery:
    """Build a valid realized query for tests."""
    return RealizedQuery(
        candidate_id=candidate_id,
        query="Can you summarize what my insurance benefits letter says is covered?",
    )


@pytest.fixture
def expected_prompt_vars_single() -> dict[str, object]:
    """Return the expected formatted realization prompt payload for one blueprint."""
    return {
        "query_blueprints": (
            "- candidate_id: C001\n"
            "  domain: healthcare\n"
            "  role: patient\n"
            "  language: English\n"
            "  topic: insurance coverage\n"
            "  intent: understand\n"
            "  task: summarize\n"
            "  user_scenario: A patient reviews a benefits letter.\n"
            "  information_need: Clarify what is covered."
        )
    }


def test_build_realization_prompt_vars_formats_blueprints(
    expected_prompt_vars_single: dict[str, object],
) -> None:
    result = _build_realization_prompt_vars([_make_blueprint()])

    assert result == expected_prompt_vars_single


def test_build_realization_prompt_vars_includes_optional_fields_only_when_present() -> None:
    result = _build_realization_prompt_vars(
        [
            _make_blueprint(),
            _make_blueprint_with_optional_fields(),
        ]
    )

    assert result == {
        "query_blueprints": (
            "- candidate_id: C001\n"
            "  domain: healthcare\n"
            "  role: patient\n"
            "  language: English\n"
            "  topic: insurance coverage\n"
            "  intent: understand\n"
            "  task: summarize\n"
            "  user_scenario: A patient reviews a benefits letter.\n"
            "  information_need: Clarify what is covered.\n\n"
            "- candidate_id: C002\n"
            "  domain: education\n"
            "  role: student\n"
            "  language: German\n"
            "  topic: scholarship eligibility\n"
            "  intent: compare\n"
            "  task: recommend\n"
            "  difficulty: advanced\n"
            "  format: table\n"
            "  user_scenario: A student compares financial aid options before applying.\n"
            "  information_need: Compare eligibility criteria for available scholarships."
        )
    }


def test_build_realization_prompt_vars_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="candidates must not be empty"):
        _build_realization_prompt_vars([])


def test_build_realization_prompt_vars_returns_exact_placeholder_mapping() -> None:
    result = _build_realization_prompt_vars([_make_blueprint()])

    assert set(result) == {"query_blueprints"}


def test_run_realization_stage_wires_realization_assets_and_settings_into_llm_builder(
    monkeypatch: pytest.MonkeyPatch,
    llm_settings: LlmSettings,
    expected_prompt_vars_single: dict[str, object],
) -> None:
    mock_runnable = Mock()
    mock_runnable.invoke.return_value = RealizedQueryList(queries=[_make_realized_query()])

    build_llm_runnable_mock = Mock(return_value=mock_runnable)
    monkeypatch.setattr(
        "pragmata.core.querygen.realization.build_llm_runnable",
        build_llm_runnable_mock,
    )

    result = run_realization_stage(
        candidates=[_make_blueprint()],
        llm_settings=llm_settings,
        api_key="test-api-key",
    )

    assert result == [_make_realized_query()]

    build_llm_runnable_mock.assert_called_once_with(
        system_text=SYSTEM_PROMPT_REALIZATION,
        user_text=USER_PROMPT_REALIZATION,
        model_provider=llm_settings.model_provider,
        model=llm_settings.realization_model,
        api_key="test-api-key",
        output_schema=RealizedQueryList,
        requests_per_second=llm_settings.requests_per_second,
        check_every_n_seconds=llm_settings.check_every_n_seconds,
        max_bucket_size=llm_settings.max_bucket_size,
        base_url=llm_settings.base_url,
        model_kwargs=llm_settings.model_kwargs,
    )
    mock_runnable.invoke.assert_called_once_with(expected_prompt_vars_single)


def test_run_realization_stage_invokes_runnable_once_and_returns_queries(
    monkeypatch: pytest.MonkeyPatch,
    llm_settings: LlmSettings,
    expected_prompt_vars_single: dict[str, object],
) -> None:
    class FakeRunnable:
        def __init__(self) -> None:
            self.seen_payload: dict[str, object] | None = None
            self.invoke_calls = 0

        def invoke(self, payload: dict[str, object]) -> RealizedQueryList:
            self.invoke_calls += 1
            self.seen_payload = payload
            return RealizedQueryList(queries=[_make_realized_query()])

    fake_runnable = FakeRunnable()

    monkeypatch.setattr(
        "pragmata.core.querygen.realization.build_llm_runnable",
        lambda **_: fake_runnable,
    )

    result = run_realization_stage(
        candidates=[_make_blueprint()],
        llm_settings=llm_settings,
        api_key="test-api-key",
    )

    assert result == [_make_realized_query()]
    assert isinstance(result, list)
    assert all(isinstance(item, RealizedQuery) for item in result)
    assert fake_runnable.invoke_calls == 1
    assert fake_runnable.seen_payload == expected_prompt_vars_single


def test_run_realization_stage_extracts_queries_from_realized_query_list(
    monkeypatch: pytest.MonkeyPatch,
    llm_settings: LlmSettings,
) -> None:
    expected_queries = [
        _make_realized_query("C001"),
        RealizedQuery(
            candidate_id="C002",
            query="Welche Stipendien kommen für mich infrage, und wie unterscheiden sich die Voraussetzungen?",
        ),
    ]

    class FakeRunnable:
        def invoke(self, payload: dict[str, object]) -> RealizedQueryList:
            return RealizedQueryList(queries=expected_queries)

    monkeypatch.setattr(
        "pragmata.core.querygen.realization.build_llm_runnable",
        lambda **_: FakeRunnable(),
    )

    result = run_realization_stage(
        candidates=[_make_blueprint(), _make_blueprint_with_optional_fields()],
        llm_settings=llm_settings,
        api_key="test-api-key",
    )

    assert result == expected_queries


def test_run_realization_stage_propagates_llm_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
    llm_settings: LlmSettings,
) -> None:
    monkeypatch.setattr(
        "pragmata.core.querygen.realization.build_llm_runnable",
        Mock(side_effect=LlmInitializationError("bad config")),
    )

    with pytest.raises(LlmInitializationError, match="bad config"):
        run_realization_stage(
            candidates=[_make_blueprint()],
            llm_settings=llm_settings,
            api_key="test-api-key",
        )


def test_run_realization_stage_wraps_invoke_failures(
    monkeypatch: pytest.MonkeyPatch,
    llm_settings: LlmSettings,
) -> None:
    class FakeRunnable:
        def invoke(self, payload: dict[str, object]) -> RealizedQueryList:
            raise RuntimeError("provider failure")

    monkeypatch.setattr(
        "pragmata.core.querygen.realization.build_llm_runnable",
        lambda **_: FakeRunnable(),
    )

    with pytest.raises(RealizationStageError, match="Realization stage invocation failed."):
        run_realization_stage(
            candidates=[_make_blueprint()],
            llm_settings=llm_settings,
            api_key="test-api-key",
        )


def test_run_realization_stage_wraps_prompt_var_construction_failures(
    llm_settings: LlmSettings,
) -> None:
    with pytest.raises(ValueError, match="candidates must not be empty"):
        run_realization_stage(
            candidates=[],
            llm_settings=llm_settings,
            api_key="test-api-key",
        )
