"""Tests CLI commands for evaluation workflows."""

import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pragmata.api import UNSET
from pragmata.cli.app import app
from pragmata.cli.commands.eval import eval_app

runner = CliRunner()

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


class _TrainResult:
    class _Paths:
        run_id = "train-run-123"
        run_dir = Path("workspace/eval/train_outputs/train-run-123")

    paths = _Paths()


def test_eval_command_registered() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "eval" in result.output


class TestTrainEvaluatorCommand:
    """Tests for the eval train-evaluator CLI command."""

    def test_root_app_help_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("COLUMNS", "200")
        result = runner.invoke(app, ["eval", "train-evaluator", "--help"], color=False)
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "Train a supervised evaluator model." in output
        assert "--train-kwargs" in output

    def test_help_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("COLUMNS", "200")
        result = runner.invoke(eval_app, ["--help"], color=False)
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "Train a supervised evaluator model." in output
        assert "--labeled-data-path" in output
        assert "--export-id" in output
        assert "--task" in output
        assert "--target-name" in output
        assert "--checkpoint" in output
        assert "--proxy-checkpoint" in output
        assert "--scale-learning-rate" in output
        assert "--no-scale-learning-rate" in output
        assert "--sequence-length" in output
        assert "--trust-remote-code" in output
        assert "--no-trust-remote-code" in output
        assert "--train-kwargs" in output

    def test_maps_omitted_options_to_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            captured.update(kwargs)
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(eval_app, [])

        expected_keys = {
            "labeled_data_path",
            "export_id",
            "task",
            "base_dir",
            "config_path",
            "target_name",
            "checkpoint",
            "proxy_checkpoint",
            "scale_learning_rate",
            "sequence_length",
            "trust_remote_code",
            "train_kwargs",
        }

        assert result.exit_code == 0
        assert set(captured) == expected_keys

        for key in expected_keys:
            assert captured[key] is UNSET

    def test_delegates_to_public_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            captured.update(kwargs)
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(
            eval_app,
            [
                "--labeled-data-path",
                "exports/retrieval.csv",
                "--export-id",
                "ignored-export",
                "--task",
                "retrieval",
                "--base-dir",
                "workspace",
                "--config",
                "eval.yml",
                "--target-name",
                "Retrieval quality",
                "--checkpoint",
                "target/checkpoint",
                "--proxy-checkpoint",
                "proxy/checkpoint",
                "--sequence-length",
                "2048",
                "--train-kwargs",
                '{"run_id": "custom-run", "verbosity": "quiet"}',
            ],
        )

        assert result.exit_code == 0
        assert captured == {
            "labeled_data_path": "exports/retrieval.csv",
            "export_id": "ignored-export",
            "task": "retrieval",
            "base_dir": "workspace",
            "config_path": "eval.yml",
            "target_name": "Retrieval quality",
            "checkpoint": "target/checkpoint",
            "proxy_checkpoint": "proxy/checkpoint",
            "scale_learning_rate": UNSET,
            "sequence_length": 2048,
            "trust_remote_code": UNSET,
            "train_kwargs": {"run_id": "custom-run", "verbosity": "quiet"},
        }

    def test_forwards_boolean_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            captured.update(kwargs)
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(
            eval_app,
            [
                "--no-scale-learning-rate",
                "--trust-remote-code",
            ],
        )

        assert result.exit_code == 0
        assert captured["scale_learning_rate"] is False
        assert captured["trust_remote_code"] is True

    def test_scale_learning_rate_flag_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            captured.update(kwargs)
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(eval_app, ["--scale-learning-rate", "--no-trust-remote-code"])

        assert result.exit_code == 0
        assert captured["scale_learning_rate"] is True
        assert captured["trust_remote_code"] is False

    def test_prints_run_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(eval_app, [])

        assert result.exit_code == 0
        assert "Evaluator training run complete." in result.output
        assert "run_id: train-run-123" in result.output
        assert "run_directory:" in result.output
        assert "workspace/eval/train_outputs/train-run-123" in result.output
