"""Integration tests for the public synthetic query generation surface."""

import json
from pathlib import Path

import numpy as np
import pytest

from pragmata import querygen
from pragmata.core.csv_io import read_csv
from pragmata.core.schemas.querygen_output import SyntheticQueriesMeta, SyntheticQueryRow
from pragmata.core.schemas.querygen_plan import QueryBlueprint, QueryBlueprintList
from pragmata.core.schemas.querygen_realize import RealizedQuery, RealizedQueryList
from pragmata.core.settings.settings_base import MissingSecretError
import pragmata.core.querygen.deduplication as deduplication
import pragmata.core.querygen.planning as planning
import pragmata.core.querygen.realization as realization

pytestmark = [pytest.mark.integration, pytest.mark.querygen]


@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a provider API key is present unless a test removes it explicitly."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-api-key")


def _required_querygen_kwargs(base_dir: Path) -> dict[str, object]:
    """Return a minimal valid public-surface invocation."""
    return {
        "domains": "healthcare",
        "roles": ["patient", "caregiver"],
        "languages": "en",
        "topics": "insurance coverage",
        "intents": "understand benefits",
        "tasks": "factual lookup",
        "difficulty": "basic",
        "formats": "bullet list",
        "disallowed_topics": ["medical diagnosis"],
        "base_dir": base_dir,
        "model_provider": "mistralai",
    }


def _make_blueprint(
    candidate_id: str,
    *,
    domain: str = "healthcare",
    role: str = "patient",
    language: str = "en",
    topic: str = "insurance coverage",
    intent: str = "understand benefits",
    task: str = "factual lookup",
    difficulty: str | None = "basic",
    format: str | None = "bullet list",
    user_scenario: str | None = None,
    information_need: str | None = None,
) -> QueryBlueprint:
    """Build a valid planning-stage blueprint."""
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
        user_scenario=user_scenario or f"Scenario for {candidate_id}",
        information_need=information_need or f"Information need for {candidate_id}",
    )


def _parse_candidate_ids_block(block: str) -> list[str]:
    """Parse the planning prompt variable containing candidate IDs."""
    return [
        line.strip().removeprefix("- ").strip()
        for line in block.splitlines()
        if line.strip()
    ]


def _parse_realization_candidate_ids(block: str) -> list[str]:
    """Parse candidate IDs from the realization blueprint block."""
    candidate_ids: list[str] = []

    for line in block.splitlines():
        stripped = line.strip()
        prefix = "- candidate_id: "
        if stripped.startswith(prefix):
            candidate_ids.append(stripped.removeprefix(prefix).strip())

    return candidate_ids


class _SimilarityModelStub:
    """Minimal similarity-only model stub for deterministic deduplication tests."""

    def similarity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        del right
        return np.eye(len(left), dtype=np.float32)


class _PlanningRunnableStub:
    """Planning runnable stub keyed by candidate-id batches."""

    def __init__(self, planning_calls: list[list[str]]) -> None:
        self._planning_calls = planning_calls
        self._duplicate_blueprint_kwargs = {
            "topic": "coverage appeals",
            "user_scenario": (
                "My insurer denied a request and I need help understanding the appeal process."
            ),
            "information_need": (
                "I need the concrete steps for appealing a denied coverage request."
            ),
        }

    def invoke(self, prompt_vars: dict[str, object]) -> QueryBlueprintList:
        """Return structured planning output for the requested batch."""
        batch_candidate_ids = _parse_candidate_ids_block(str(prompt_vars["candidate_ids"]))
        self._planning_calls.append(batch_candidate_ids)

        if batch_candidate_ids == ["c001", "c002"]:
            return QueryBlueprintList(
                candidates=[
                    _make_blueprint("c001"),
                    _make_blueprint("c002"),
                ]
            )

        if batch_candidate_ids == ["c003", "c004"]:
            return QueryBlueprintList(
                candidates=[
                    _make_blueprint("c003", **self._duplicate_blueprint_kwargs),
                    _make_blueprint("wrong-c004"),
                ]
            )

        if batch_candidate_ids == ["c005"]:
            return QueryBlueprintList(
                candidates=[
                    _make_blueprint("c005", **self._duplicate_blueprint_kwargs),
                ]
            )

        pytest.fail(f"Unexpected planning batch: {batch_candidate_ids}")


class _RealizationRunnableStub:
    """Realization runnable stub keyed by blueprint batches."""

    def __init__(self, realization_calls: list[list[str]]) -> None:
        self._realization_calls = realization_calls

    def invoke(self, prompt_vars: dict[str, object]) -> RealizedQueryList:
        """Return structured realization output for the requested batch."""
        batch_candidate_ids = _parse_realization_candidate_ids(str(prompt_vars["query_blueprints"]))
        self._realization_calls.append(batch_candidate_ids)

        if batch_candidate_ids == ["c001", "c002"]:
            return RealizedQueryList(
                queries=[
                    RealizedQuery(
                        candidate_id="c001",
                        query="How can I check whether a treatment is covered by my insurance plan?",
                    ),
                    RealizedQuery(
                        candidate_id="wrong-c002",
                        query="This query should be removed by stage-2 filtering.",
                    ),
                ]
            )

        if batch_candidate_ids == ["c003"]:
            return RealizedQueryList(
                queries=[
                    RealizedQuery(
                        candidate_id="c003",
                        query="What steps do I need to follow to appeal a denied coverage request?",
                    )
                ]
            )

        pytest.fail(f"Unexpected realization batch: {batch_candidate_ids}")


class _TimeoutRunnableStub:
    """Runnable stub that always fails at invocation time."""

    def __init__(self, message: str) -> None:
        self._message = message

    def invoke(self, prompt_vars: dict[str, object]) -> QueryBlueprintList | RealizedQueryList:
        """Raise a timeout-style failure."""
        del prompt_vars
        raise TimeoutError(self._message)


@pytest.fixture
def happy_path_workflow_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[list[str]]]:
    """Install happy-path LLM and deduplication stubs for end-to-end API tests."""
    planning_calls: list[list[str]] = []
    realization_calls: list[list[str]] = []

    def fake_planning_build_llm_runnable(**kwargs: object) -> _PlanningRunnableStub:
        del kwargs
        return _PlanningRunnableStub(planning_calls)

    def fake_realization_build_llm_runnable(**kwargs: object) -> _RealizationRunnableStub:
        del kwargs
        return _RealizationRunnableStub(realization_calls)

    monkeypatch.setattr(planning, "build_llm_runnable", fake_planning_build_llm_runnable)
    monkeypatch.setattr(realization, "build_llm_runnable", fake_realization_build_llm_runnable)
    monkeypatch.setattr(
        deduplication,
        "_embed_blueprints",
        lambda candidates: np.eye(len(candidates), dtype=np.float32),
    )
    monkeypatch.setattr(
        deduplication,
        "_load_embedding_model",
        lambda checkpoint="all-MiniLM-L6-v2": _SimilarityModelStub(),
    )

    return {
        "planning_calls": planning_calls,
        "realization_calls": realization_calls,
    }


def test_gen_queries_executes_staged_workflow_and_writes_expected_artifacts(
    happy_path_workflow_stubs: dict[str, list[list[str]]],
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "workspace"
    run_id = "integration-run-staged-001"

    result = querygen.gen_queries(
        **_required_querygen_kwargs(base_dir),
        run_id=run_id,
        n_queries=5,
        batch_size=2,
    )

    assert isinstance(result, querygen.QueryGenRunResult)

    expected_run_dir = base_dir.resolve() / "querygen" / "runs" / run_id
    assert result.paths.run_dir == expected_run_dir
    assert result.paths.synthetic_queries_csv == expected_run_dir / "synthetic_queries.csv"
    assert result.paths.synthetic_queries_meta_json == expected_run_dir / "synthetic_queries.meta.json"

    assert happy_path_workflow_stubs["planning_calls"] == [
        ["c001", "c002"],
        ["c003", "c004"],
        ["c005"],
    ]
    assert happy_path_workflow_stubs["realization_calls"] == [
        ["c001", "c002"],
        ["c003"],
    ]

    assert result.paths.run_dir.exists()
    assert result.paths.synthetic_queries_csv.exists()
    assert result.paths.synthetic_queries_meta_json.exists()

    rows = read_csv(result.paths.synthetic_queries_csv, SyntheticQueryRow)
    assert rows == [
        SyntheticQueryRow(
            query_id=f"{run_id}_q1",
            query="How can I check whether a treatment is covered by my insurance plan?",
            domain="healthcare",
            role="patient",
            language="en",
            topic="insurance coverage",
            intent="understand benefits",
            task="factual lookup",
            difficulty="basic",
            format="bullet list",
        ),
        SyntheticQueryRow(
            query_id=f"{run_id}_q2",
            query="What steps do I need to follow to appeal a denied coverage request?",
            domain="healthcare",
            role="patient",
            language="en",
            topic="coverage appeals",
            intent="understand benefits",
            task="factual lookup",
            difficulty="basic",
            format="bullet list",
        ),
    ]

    meta = SyntheticQueriesMeta.model_validate(
        json.loads(result.paths.synthetic_queries_meta_json.read_text(encoding="utf-8"))
    )
    assert meta.run_id == run_id
    assert meta.n_requested_queries == 5
    assert meta.n_returned_queries == 2
    assert meta.model_provider == "mistralai"
    assert meta.planning_model == "magistral-medium-latest"
    assert meta.realization_model == "mistral-medium-latest"



def test_gen_queries_raises_when_provider_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail before creating a run directory when the provider key is absent."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    base_dir = tmp_path / "workspace"
    run_id = "integration-run-missing-secret"
    expected_run_dir = base_dir.resolve() / "querygen" / "runs" / run_id

    with pytest.raises(MissingSecretError, match="MISTRAL_API_KEY"):
        querygen.gen_queries(
            **_required_querygen_kwargs(base_dir),
            run_id=run_id,
            n_queries=2,
        )

    assert not expected_run_dir.exists()

def test_gen_queries_propagates_planning_timeout_and_writes_no_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A planning-stage provider timeout propagates and prevents export."""

    def fake_planning_build_llm_runnable(**kwargs: object) -> _TimeoutRunnableStub:
        del kwargs
        return _TimeoutRunnableStub("provider timed out")

    monkeypatch.setattr(planning, "build_llm_runnable", fake_planning_build_llm_runnable)

    base_dir = tmp_path / "workspace"
    run_id = "integration-run-planning-timeout"

    with pytest.raises(planning.PlanningStageError, match="Planning stage invocation failed."):
        querygen.gen_queries(
            **_required_querygen_kwargs(base_dir),
            run_id=run_id,
            n_queries=2,
        )

    expected_run_dir = base_dir.resolve() / "querygen" / "runs" / run_id
    assert expected_run_dir.exists()
    assert not (expected_run_dir / "synthetic_queries.csv").exists()
    assert not (expected_run_dir / "synthetic_queries.meta.json").exists()
