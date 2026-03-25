"""Tests CLI command for synthetic query generation."""

import re
from pathlib import Path

from typer.testing import CliRunner

from pragmata.cli.app import app
from pragmata.api.querygen import UNSET

runner = CliRunner()

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


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
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Prepare a synthetic query generation run." in output
    assert "--domains" in output
    assert "--run-id" in output


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
            "--model-kwargs",
            '{"temperature": 0.2}',
        ],
    )

    assert result.exit_code == 0
    assert captured["domains"] == ["public administration"]
    assert captured["roles"] == "policy analyst"
    assert captured["run_id"] == "custom-run"
    assert captured["model_kwargs"] == {"temperature": 0.2}


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
        "model_kwargs",
    }

    assert result.exit_code == 0
    assert set(captured) == expected_keys

    for key in expected_keys:
        assert captured[key] is UNSET


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
