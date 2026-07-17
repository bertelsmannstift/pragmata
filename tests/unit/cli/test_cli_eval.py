"""Tests CLI commands for evaluation workflows."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from pragmata.api import UNSET
from pragmata.cli.app import app
from pragmata.cli.commands.eval import eval_app
from pragmata.core.schemas.eval_output import MetricScore, RetrievalScoreReport, ScoreInputSource
from tests.unit.cli.conftest import strip_ansi

runner = CliRunner()


class _TrainResult:
    class _Paths:
        run_id = "train-run-123"
        run_dir = Path("workspace/eval/train_outputs/train-run-123")
        model_dir = Path("workspace/eval/train_outputs/train-run-123/model")

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
        result = runner.invoke(eval_app, ["train-evaluator", "--help"], color=False)
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
        assert "--trust-remote-code" not in output
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

        result = runner.invoke(eval_app, ["train-evaluator"])

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
                "train-evaluator",
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
                "train-evaluator",
                "--no-scale-learning-rate",
            ],
        )

        assert result.exit_code == 0
        assert captured["scale_learning_rate"] is False

    def test_scale_learning_rate_flag_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            captured.update(kwargs)
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(eval_app, ["train-evaluator", "--scale-learning-rate"])

        assert result.exit_code == 0
        assert captured["scale_learning_rate"] is True

    def test_prints_run_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_train_evaluator(**kwargs: Any) -> _TrainResult:
            return _TrainResult()

        monkeypatch.setattr("pragmata.eval.train_evaluator", fake_train_evaluator)

        result = runner.invoke(eval_app, ["train-evaluator"])

        assert result.exit_code == 0
        assert "Evaluator training run complete." in result.output
        assert "run_id: train-run-123" in result.output
        assert "run_directory:" in result.output
        assert "workspace/eval/train_outputs/train-run-123" in result.output
        assert "model_directory:" in result.output
        assert "workspace/eval/train_outputs/train-run-123/model" in result.output


def _fake_retrieval_report() -> RetrievalScoreReport:
    def metric(method: str) -> MetricScore:
        return MetricScore(point=0.5, ci_lower=0.4, ci_upper=0.6, method=method, n=2)

    return RetrievalScoreReport(
        source=ScoreInputSource(kind="direct_path", ref="d.csv", resolved_path="d.csv"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        n_examples=2,
        top_k=3,
        ci_level=0.95,
        topical_precision_at_k=metric("bootstrap"),
        sufficiency_hit_at_k=metric("wilson"),
        sufficiency_rate_at_k=metric("bootstrap"),
        misleading_context_rate_at_k=metric("bootstrap"),
        mean_reciprocal_rank_at_k=metric("bootstrap"),
        ndcg_at_k=metric("bootstrap"),
    )


def _retrieval_rows() -> list[dict]:
    return [
        {
            "record_uuid": "r1",
            "chunk_id": "c1",
            "chunk_rank": 1,
            "query": "q1",
            "chunk": "a",
            "topically_relevant": 1,
            "evidence_sufficient": 1,
            "misleading": 0,
        },
        {
            "record_uuid": "r1",
            "chunk_id": "c2",
            "chunk_rank": 2,
            "query": "q1",
            "chunk": "b",
            "topically_relevant": 1,
            "evidence_sufficient": 0,
            "misleading": 0,
        },
        {
            "record_uuid": "r2",
            "chunk_id": "c3",
            "chunk_rank": 1,
            "query": "q2",
            "chunk": "c",
            "topically_relevant": 0,
            "evidence_sufficient": 0,
            "misleading": 1,
        },
    ]


class TestScoreCommand:
    """Tests for the eval score CLI command."""

    def test_help_lists_score_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLUMNS", "200")
        result = runner.invoke(eval_app, ["score", "--help"], color=False)
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "Score labeled eval data" in output
        for option in (
            "--task",
            "--path",
            "--export-id",
            "--prediction-id",
            "--score-id",
            "--base-dir",
            "--n-resamples",
            "--ci",
            "--seed",
            "--config",
        ):
            assert option in output
        assert "--top-k" not in output  # K is inferred, never a parameter

    def test_maps_omitted_options_to_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_score(**kwargs: Any) -> RetrievalScoreReport:
            captured.update(kwargs)
            return _fake_retrieval_report()

        monkeypatch.setattr("pragmata.eval.score", fake_score)

        result = runner.invoke(eval_app, ["score"])

        assert result.exit_code == 0
        assert set(captured) == {
            "task",
            "path",
            "export_id",
            "prediction_id",
            "score_id",
            "base_dir",
            "n_resamples",
            "ci",
            "seed",
            "config_path",
        }
        assert all(value is UNSET for value in captured.values())

    def test_delegates_to_public_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_score(**kwargs: Any) -> RetrievalScoreReport:
            captured.update(kwargs)
            return _fake_retrieval_report()

        monkeypatch.setattr("pragmata.eval.score", fake_score)

        result = runner.invoke(
            eval_app,
            [
                "score",
                "--task",
                "retrieval",
                "--path",
                "data/labeled.csv",
                "--export-id",
                "e1",
                "--prediction-id",
                "p1",
                "--score-id",
                "score-1",
                "--base-dir",
                "workspace",
                "--n-resamples",
                "500",
                "--ci",
                "0.9",
                "--seed",
                "7",
                "--config",
                "eval.yml",
            ],
        )

        assert result.exit_code == 0
        assert captured == {
            "task": "retrieval",
            "path": "data/labeled.csv",
            "export_id": "e1",
            "prediction_id": "p1",
            "score_id": "score-1",
            "base_dir": "workspace",
            "n_resamples": 500,
            "ci": 0.9,
            "seed": 7,
            "config_path": "eval.yml",
        }

    def test_prints_score_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pragmata.eval.score", lambda **_: _fake_retrieval_report())

        result = runner.invoke(eval_app, ["score", "--task", "retrieval", "--path", "d.csv"])
        output = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "retrieval scores (n=2, 95% CI)" in output
        assert "ndcg_at_k: 0.500 [0.400, 0.600] (bootstrap, n=2)" in output
        assert "sufficiency_hit_at_k: 0.500 [0.400, 0.600] (wilson, n=2)" in output

    def test_end_to_end_writes_report(self, tmp_path: Path) -> None:
        csv = tmp_path / "ret.csv"
        pd.DataFrame(_retrieval_rows()).to_csv(csv, index=False)

        result = runner.invoke(
            eval_app,
            ["score", "--task", "retrieval", "--path", str(csv), "--base-dir", str(tmp_path), "--seed", "1"],
        )

        assert result.exit_code == 0
        assert "retrieval scores" in strip_ansi(result.output)
        assert next((tmp_path / "eval" / "scores").glob("*/retrieval_scores.json")).is_file()

    def test_prediction_id_exits_nonzero(self, tmp_path: Path) -> None:
        # Prediction-run scoring is not yet supported; it must fail rather than silently no-op.
        result = runner.invoke(
            eval_app, ["score", "--task", "retrieval", "--prediction-id", "p1", "--base-dir", str(tmp_path)]
        )

        assert result.exit_code != 0
