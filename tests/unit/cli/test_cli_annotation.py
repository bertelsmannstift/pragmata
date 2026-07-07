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


class TestStatusCommand:
    def _stub_report(
        self,
        *,
        n_panels: int = 2,
        n_complete: int = 1,
        n_overlap_satisfied: int = 1,
        n_integrity_warnings: int = 0,
        n_orphans_skipped: int = 0,
    ):
        from pragmata.core.annotation.panel_status import (
            HeadlineTotals,
            ProgressReport,
            ProgressRow,
            StatusReport,
        )

        progress = ProgressReport(
            grand=HeadlineTotals(total=100, completed=20, pending=80),
            by_task=[
                ProgressRow(label="retrieval", task="retrieval", total=60, completed=10, pending=50),
                ProgressRow(label="grounding", task="grounding", total=20, completed=4, pending=16),
                ProgressRow(label="generation", task="generation", total=20, completed=6, pending=14),
            ],
            by_workspace=[ProgressRow(label="ws1_retrieval", task="retrieval", total=60, completed=10, pending=50)],
            by_dataset=[
                ProgressRow(
                    label="ws1_retrieval/retrieval_production", task="retrieval", total=60, completed=10, pending=50
                )
            ],
        )
        return StatusReport(
            panels={},
            headline=HeadlineTotals(total=60, completed=10, pending=50),
            n_panels=n_panels,
            n_complete=n_complete,
            n_overlap_satisfied=n_overlap_satisfied,
            n_integrity_warnings=n_integrity_warnings,
            n_orphans_skipped=n_orphans_skipped,
        ).with_progress(progress)

    @patch("pragmata.annotation.report_status")
    def test_status_shows_all_tasks_and_panels(self, mock_status):
        mock_status.return_value = self._stub_report()
        result = runner.invoke(app, ["annotation", "status"])
        assert result.exit_code == 0
        assert "records: 100 total" in result.output
        assert "20 completed" in result.output
        for task in ("retrieval", "grounding", "generation"):
            assert task in result.output  # all three tasks displayed
        assert "PANELS" in result.output
        assert "–" in result.output  # non-retrieval tasks show a dash for panels

    @patch("pragmata.annotation.report_status")
    def test_status_reports_integrity_warnings_when_present(self, mock_status):
        mock_status.return_value = self._stub_report(n_integrity_warnings=3)
        result = runner.invoke(app, ["annotation", "status"])
        assert result.exit_code == 0
        assert "integrity warnings: 3" in result.output

    @patch("pragmata.annotation.report_status")
    def test_status_by_workspace_and_by_dataset_flags(self, mock_status):
        mock_status.return_value = self._stub_report()
        out_ws = runner.invoke(app, ["annotation", "status", "--by-workspace"]).output
        assert "WORKSPACE" in out_ws and "ws1_retrieval" in out_ws
        out_ds = runner.invoke(app, ["annotation", "status", "--by-dataset"]).output
        assert "DATASET" in out_ds and "retrieval_production" in out_ds

    @patch("pragmata.annotation.report_status")
    def test_status_threads_defaults(self, mock_status):
        mock_status.return_value = self._stub_report()
        runner.invoke(app, ["annotation", "status"])
        kwargs = mock_status.call_args.kwargs
        assert kwargs["api_url"] is UNSET
        assert kwargs["workspace"] is None


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


class TestIaaCommandFilters:
    """CLI wiring for --after / --before / --exclude-annotators."""

    @patch("pragmata.annotation.compute_iaa")
    def test_after_and_before_parsed_to_datetime(self, mock_iaa):
        mock_iaa.return_value = _make_report()
        result = runner.invoke(
            app,
            [
                "annotation",
                "iaa",
                "test-export",
                "--after",
                "2026-05-01T00:00:00",
                "--before",
                "2026-06-01",
            ],
        )
        assert result.exit_code == 0
        kwargs = mock_iaa.call_args.kwargs
        assert kwargs["after"] == datetime(2026, 5, 1, 0, 0, 0)
        assert kwargs["before"] == datetime(2026, 6, 1)

    @patch("pragmata.annotation.compute_iaa")
    def test_exclude_annotators_parsed(self, mock_iaa):
        mock_iaa.return_value = _make_report()
        result = runner.invoke(
            app,
            ["annotation", "iaa", "test-export", "--exclude-annotators", "alice, bob ,carol"],
        )
        assert result.exit_code == 0
        kwargs = mock_iaa.call_args.kwargs
        assert kwargs["exclude_annotators"] == ["alice", "bob", "carol"]

    @patch("pragmata.annotation.compute_iaa")
    def test_filters_default_to_none(self, mock_iaa):
        mock_iaa.return_value = _make_report()
        result = runner.invoke(app, ["annotation", "iaa", "test-export"])
        assert result.exit_code == 0
        kwargs = mock_iaa.call_args.kwargs
        assert kwargs["after"] is None
        assert kwargs["before"] is None
        assert kwargs["exclude_annotators"] is None

    @patch("pragmata.annotation.compute_iaa")
    def test_invalid_after_exits_with_usage_error(self, mock_iaa):
        result = runner.invoke(
            app,
            ["annotation", "iaa", "test-export", "--after", "not-a-date"],
        )
        assert result.exit_code != 0
        mock_iaa.assert_not_called()


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
        assert kwargs["calibration_max_items"] is UNSET

    @patch("pragmata.annotation.import_records")
    def test_calibration_max_items_threaded_through(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(
            app,
            ["annotation", "import", str(records_file), "--calibration-max-items", "200"],
        )

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["calibration_max_items"] == 200

    @patch("pragmata.annotation.import_records")
    def test_calibration_max_items_conflicts_with_no_calibration(self, mock_import, tmp_path):
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(
            app,
            ["annotation", "import", str(records_file), "--no-calibration", "--calibration-max-items", "50"],
        )

        assert result.exit_code == 2
        assert "cannot be combined" in result.output
        mock_import.assert_not_called()

    @patch("pragmata.annotation.import_records")
    def test_locale_catalog_dir_threaded_through(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")
        catalog_dir = tmp_path / "locales"
        catalog_dir.mkdir()

        result = runner.invoke(
            app,
            ["annotation", "import", str(records_file), "--locale-catalog", str(catalog_dir)],
        )

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["locale_catalog_dir"] == str(catalog_dir)

    @patch("pragmata.annotation.import_records")
    def test_locale_catalog_dir_default_is_unset(self, mock_import, tmp_path):
        mock_import.return_value = _empty_import_result()
        records_file = tmp_path / "records.jsonl"
        records_file.write_text("", encoding="utf-8")

        result = runner.invoke(app, ["annotation", "import", str(records_file)])

        assert result.exit_code == 0
        kwargs = mock_import.call_args.kwargs
        assert kwargs["locale_catalog_dir"] is UNSET
