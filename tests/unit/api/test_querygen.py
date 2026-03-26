"""Tests for the query generation API orchestration."""

from pathlib import Path

import pytest

from pragmata.api.querygen import QueryGenRunResult, gen_queries


@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure an API key is always present for orchestration tests."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-secret")


def _required_querygen_kwargs(tmp_path: Path) -> dict[str, object]:
    """Return the minimal valid kwargs for gen_queries()."""
    return {
        "domains": "public administration",
        "roles": "policy analyst",
        "languages": "en",
        "topics": "digital services",
        "intents": "learn",
        "tasks": "summarize",
        "base_dir": tmp_path,
        "model_provider": "mistralai",
    }


def test_gen_queries_combines_user_args_config_and_defaults(tmp_path: Path) -> None:
    """gen_queries combines user args, config values, and model defaults."""
    config_path = tmp_path / "querygen.yml"
    config_path.write_text(
        (
            "llm:\n"
            "  model_provider: mistralai\n"
            "  planning_model: custom-planner\n"
            "n_queries: 10\n"
            "batch_size: 12\n"
            "run_id: original-id\n"
        ),
        encoding="utf-8",
    )

    result = gen_queries(
        **_required_querygen_kwargs(tmp_path),
        config_path=config_path,
        run_id="overridden-id",
    )

    assert result.settings.n_queries == 10
    assert result.settings.llm.planning_model == "custom-planner"
    assert result.settings.run_id == "overridden-id"
    assert result.paths.run_dir.name == result.settings.run_id
    assert result.settings.llm.realization_model == "mistral-medium-latest"
    assert result.settings.batch_size == 12


def test_gen_queries_orchestrates_run_paths(tmp_path: Path) -> None:
    """gen_queries resolves and creates the expected run path scaffold."""
    result = gen_queries(
        **_required_querygen_kwargs(tmp_path),
        run_id="run-123",
    )

    expected_run_dir = tmp_path.resolve() / "querygen" / "runs" / "run-123"

    assert result.paths.run_dir == expected_run_dir
    assert result.paths.synthetic_queries_csv == expected_run_dir / "synthetic_queries.csv"
    assert result.paths.synthetic_queries_meta_json == expected_run_dir / "synthetic_queries.meta.json"
    assert expected_run_dir.is_dir()


def test_gen_queries_returns_result_object(tmp_path: Path) -> None:
    """gen_queries returns the structured prepared run result."""
    result = gen_queries(
        **_required_querygen_kwargs(tmp_path),
        run_id="result-check",
    )

    assert isinstance(result, QueryGenRunResult)
    assert result.settings.run_id == "result-check"
    assert result.paths.run_dir.name == "result-check"


def test_gen_queries_accepts_batch_size_override(tmp_path: Path) -> None:
    result = gen_queries(
        **_required_querygen_kwargs(tmp_path),
        batch_size=7,
        run_id="batch-size-check",
    )

    assert result.settings.batch_size == 7