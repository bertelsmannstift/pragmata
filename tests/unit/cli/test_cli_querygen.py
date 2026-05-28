"""Tests CLI command for synthetic query generation."""

from pathlib import Path

from typer.testing import CliRunner

from pragmata.api import UNSET
from pragmata.cli.app import app

runner = CliRunner()


class _PreparedResult:
    class _Settings:
        run_id = "run-123"

    class _Paths:
        run_dir = Path("workspace/querygen/runs/run-123")
        synthetic_queries_csv = Path("workspace/querygen/runs/run-123/synthetic_queries.csv")
        synthetic_queries_meta_json = Path("workspace/querygen/runs/run-123/synthetic_queries.meta.json")

    settings = _Settings()
    paths = _Paths()


def test_querygen_command_registered() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "querygen" in result.output


def test_querygen_gen_queries_help_available() -> None:
    result = runner.invoke(app, ["querygen", "gen-queries", "--help"])

    assert result.exit_code == 0


def test_querygen_cli_delegates_to_public_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_gen_queries(**kwargs):
        captured.update(kwargs)
        return _PreparedResult()

    monkeypatch.setattr("pragmata.querygen.gen_queries", fake_gen_queries)

    result = runner.invoke(
        app,
        [
            "querygen",
            "gen-queries",
            "--domains",
            '["public administration"]',
            "--roles",
            "policy analyst",
            "--run-id",
            "custom-run",
            "--planning-model-kwargs",
            '{"reasoning": {"effort": "medium"}}',
            "--realization-model-kwargs",
            '{"reasoning": {"effort": "low"}}',
        ],
    )

    assert result.exit_code == 0
    assert captured["domains"] == ["public administration"]
    assert captured["roles"] == "policy analyst"
    assert captured["run_id"] == "custom-run"
    assert captured["planning_model_kwargs"] == {"reasoning": {"effort": "medium"}}
    assert captured["realization_model_kwargs"] == {"reasoning": {"effort": "low"}}


def test_querygen_cli_maps_omitted_options_to_unset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_gen_queries(**kwargs):
        captured.update(kwargs)
        return _PreparedResult()

    monkeypatch.setattr("pragmata.querygen.gen_queries", fake_gen_queries)

    result = runner.invoke(app, ["querygen", "gen-queries"])

    expected_keys = {
        "domains",
        "roles",
        "languages",
        "topics",
        "intents",
        "tasks",
        "disallowed_topics",
        "difficulty",
        "formats",
        "base_dir",
        "config_path",
        "n_queries",
        "run_id",
        "model_provider",
        "planning_model",
        "realization_model",
        "base_url",
        "planning_model_kwargs",
        "realization_model_kwargs",
        "requests_per_second",
        "check_every_n_seconds",
        "max_bucket_size",
        "batch_size",
        "near_duplicate_tolerance",
        "enable_planning_memory",
    }

    assert result.exit_code == 0
    # ``fresh`` is a direct bool control flag (default False), not a settings-resolved UNSET option.
    assert set(captured) == expected_keys | {"fresh"}
    assert captured["fresh"] is False

    for key in expected_keys:
        assert captured[key] is UNSET


def test_querygen_cli_forwards_runtime_and_throttle_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_gen_queries(**kwargs):
        captured.update(kwargs)
        return _PreparedResult()

    monkeypatch.setattr("pragmata.querygen.gen_queries", fake_gen_queries)

    result = runner.invoke(
        app,
        [
            "querygen",
            "gen-queries",
            "--requests-per-second",
            "5.5",
            "--check-every-n-seconds",
            "0.25",
            "--max-bucket-size",
            "10",
            "--batch-size",
            "8",
            "--near-duplicate-tolerance",
            "0.8",
            "--no-enable-planning-memory",
        ],
    )

    assert result.exit_code == 0
    assert captured["requests_per_second"] == 5.5
    assert captured["check_every_n_seconds"] == 0.25
    assert captured["max_bucket_size"] == 10
    assert captured["batch_size"] == 8
    assert captured["near_duplicate_tolerance"] == 0.8
    assert captured["enable_planning_memory"] is False


def test_querygen_cli_enable_planning_memory_flag_true(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_gen_queries(**kwargs):
        captured.update(kwargs)
        return _PreparedResult()

    monkeypatch.setattr("pragmata.querygen.gen_queries", fake_gen_queries)

    result = runner.invoke(app, ["querygen", "gen-queries", "--enable-planning-memory"])

    assert result.exit_code == 0
    assert captured["enable_planning_memory"] is True


def test_querygen_cli_prints_prepared_run_summary(monkeypatch) -> None:
    def fake_gen_queries(**kwargs):
        return _PreparedResult()

    monkeypatch.setattr("pragmata.querygen.gen_queries", fake_gen_queries)

    result = runner.invoke(app, ["querygen", "gen-queries"])

    assert result.exit_code == 0
    assert "Synthetic query generation run prepared." in result.output
    assert "run_id: run-123" in result.output
    assert "run_directory:" in result.output
    assert "synthetic_queries.csv" in result.output
    assert "synthetic_queries.meta.json" in result.output
