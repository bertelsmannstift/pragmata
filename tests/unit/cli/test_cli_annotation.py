"""Tests for the annotation CLI commands."""

from datetime import datetime, timezone
from unittest.mock import patch

from typer.testing import CliRunner

from pragmata.annotation import Task
from pragmata.api import UNSET
from pragmata.api.annotation_import import ImportResult
from pragmata.cli.app import app
from pragmata.core.schemas.iaa_report import (
    AnnotatorPair,
    IaaReport,
    LabelAgreement,
    TaskAgreement,
)

runner = CliRunner()


def _empty_import_result() -> ImportResult:
    return ImportResult(total_records=0, dataset_counts={})


def _make_report(*, alpha: float | None = 0.6, ci_lower: float | None = 0.3, ci_upper: float | None = 0.9) -> IaaReport:
    return IaaReport(
        export_id="test-export",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        tasks=[
            TaskAgreement(
                task=Task.RETRIEVAL,
                labels=[
                    LabelAgreement(
                        label="topically_relevant",
                        alpha=alpha,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        n_items=6,
                        n_annotators=2,
                        pct_agreement=0.83,
                    ),
                ],
                pairwise_kappa=[AnnotatorPair(annotator_a="alice", annotator_b="bob", kappa=0.5, n_shared_items=6)],
            ),
        ],
        n_bootstrap_resamples=100,
        ci_level=0.95,
    )


class TestIaaCommand:
    @patch("pragmata.annotation.compute_iaa")
    def test_displays_alpha_with_ci(self, mock_iaa):
        mock_iaa.return_value = _make_report(alpha=0.593, ci_lower=-0.1, ci_upper=1.0)
        result = runner.invoke(app, ["annotation", "iaa", "test-export"])
        assert result.exit_code == 0
        assert "alpha=0.593 [-0.100, 1.000]" in result.output

    @patch("pragmata.annotation.compute_iaa")
    def test_none_alpha_shows_insufficient_overlap(self, mock_iaa):
        mock_iaa.return_value = _make_report(alpha=None, ci_lower=None, ci_upper=None)
        result = runner.invoke(app, ["annotation", "iaa", "test-export"])
        assert result.exit_code == 0
        assert "n/a (insufficient overlap)" in result.output
        assert ".3f" not in result.output

    @patch("pragmata.annotation.compute_iaa")
    def test_empty_report_no_crash(self, mock_iaa):
        mock_iaa.return_value = IaaReport(
            export_id="empty",
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            tasks=[],
            n_bootstrap_resamples=100,
            ci_level=0.95,
        )
        result = runner.invoke(app, ["annotation", "iaa", "empty"])
        assert result.exit_code == 0


class TestImportCommandFlags:
    """CLI wiring for --no-calibration and --calibration-partition-seed."""

    @patch("pragmata.annotation.import_records")
    def test_no_calibration_sets_fraction_and_min_submitted(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(app, ["annotation", "import", str(records_file), "--no-calibration"])

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["calibration_fraction"] == 0.0
        assert kwargs["calibration_min_submitted"] is None

    @patch("pragmata.annotation.import_records")
    def test_no_calibration_conflicts_with_positive_fraction(self, mock_import, tmp_path):
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(
            app,
            ["annotation", "import", str(records_file), "--no-calibration", "--calibration-fraction", "0.2"],
        )

        assert result.exit_code == 2
        assert "cannot be combined" in result.output
        mock_import.assert_not_called()

    @patch("pragmata.annotation.import_records")
    def test_no_calibration_with_zero_fraction_is_allowed(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(
            app,
            ["annotation", "import", str(records_file), "--no-calibration", "--calibration-fraction", "0.0"],
        )

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["calibration_fraction"] == 0.0
        assert kwargs["calibration_min_submitted"] is None

    @patch("pragmata.annotation.import_records")
    def test_partition_seed_threaded_through(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(
            app,
            ["annotation", "import", str(records_file), "--calibration-partition-seed", "42"],
        )

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["calibration_partition_seed"] == 42

    @patch("pragmata.annotation.import_records")
    def test_partition_seed_default_is_unset(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(app, ["annotation", "import", str(records_file)])

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["calibration_partition_seed"] is UNSET
        assert kwargs["calibration_min_submitted"] is UNSET
        assert kwargs["calibration_fraction"] is UNSET
