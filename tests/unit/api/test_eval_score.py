"""End-to-end tests for the eval scoring API (``pragmata.eval.score``)."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from pragmata.api.eval import score
from pragmata.core.schemas.annotation_export import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import EvalInputSchemaError
from pragmata.core.schemas.eval_output import (
    GenerationScoreReport,
    GroundingScoreReport,
    MetricScore,
    RetrievalScoreReport,
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _retrieval_rows() -> list[dict]:
    # rec-1: two chunks (one relevant+sufficient, one relevant); rec-2: one irrelevant chunk.
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


def _grounding_rows() -> list[dict]:
    return [
        {
            "record_uuid": "r1",
            "answer": "a",
            "context_set": "c",
            "support_present": 1,
            "unsupported_claim_present": 0,
            "contradicted_claim_present": 0,
            "source_cited": 1,
            "fabricated_source": 1,
        },
        {
            "record_uuid": "r2",
            "answer": "a",
            "context_set": "c",
            "support_present": 0,
            "unsupported_claim_present": 1,
            "contradicted_claim_present": 0,
            "source_cited": 0,
            "fabricated_source": 0,
        },
    ]


def _generation_rows() -> list[dict]:
    return [
        {
            "record_uuid": "r1",
            "query": "q",
            "answer": "a",
            "proper_action": 1,
            "response_on_topic": 1,
            "helpful": 1,
            "incomplete": 0,
            "unsafe_content": 0,
        },
        {
            "record_uuid": "r2",
            "query": "q",
            "answer": "a",
            "proper_action": 0,
            "response_on_topic": 1,
            "helpful": 0,
            "incomplete": 1,
            "unsafe_content": 0,
        },
    ]


def _write_retrieval_export(base_dir: Path, export_id: str) -> None:
    """Write a retrieval annotation export under ``base_dir`` for fallback/export tests."""
    export_dir = base_dir / "annotation" / "exports" / export_id
    _write(export_dir / "retrieval.csv", _retrieval_rows())
    meta = AnnotationExportMeta(
        export_id=export_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_id=None,
        tasks=[Task.RETRIEVAL],
        include_discarded=False,
        row_counts={Task.RETRIEVAL: 3},
        n_annotators={Task.RETRIEVAL: 1},
        calibration_enabled={},
        constraint_summary={},
    )
    (export_dir / "annotation_export.meta.json").write_text(meta.model_dump_json(), encoding="utf-8")


class TestScoreRetrieval:
    def test_returns_and_writes_report(self, tmp_path: Path):
        csv = _write(tmp_path / "ret.csv", _retrieval_rows())

        report = score(base_dir=tmp_path, path=csv, task="retrieval", seed=42, n_resamples=200)

        assert isinstance(report, RetrievalScoreReport)
        assert report.task == Task.RETRIEVAL
        assert report.n_examples == 2  # two queries
        assert report.top_k == 2  # inferred max chunk_rank
        assert report.ci_level == 0.95
        # topical precision: mean over queries of (per-query mean) = mean(1.0, 0.0) = 0.5
        assert report.topical_precision_at_k.point == pytest.approx(0.5)

        # the written JSON round-trips back to an equal report
        out = next((tmp_path / "eval" / "scores").glob("*/retrieval_scores.json"))
        assert RetrievalScoreReport.model_validate_json(out.read_text()) == report

    def test_method_per_metric(self, tmp_path: Path):
        csv = _write(tmp_path / "ret.csv", _retrieval_rows())
        report = score(base_dir=tmp_path, path=csv, task="retrieval", seed=1)

        assert report.sufficiency_hit_at_k.method == "wilson"
        for field in (
            report.topical_precision_at_k,
            report.sufficiency_rate_at_k,
            report.misleading_context_rate_at_k,
            report.mean_reciprocal_rank_at_k,
            report.ndcg_at_k,
        ):
            assert field.method == "bootstrap"

    def test_wilson_point_within_interval_bootstrap_only_ordered(self, tmp_path: Path):
        csv = _write(tmp_path / "ret.csv", _retrieval_rows())
        report = score(base_dir=tmp_path, path=csv, task="retrieval", seed=7)

        # Wilson: point lies within the interval by construction.
        hit = report.sufficiency_hit_at_k
        assert hit.ci_lower <= hit.point <= hit.ci_upper
        # Bootstrap at tiny n: only require ordered, bounded CI (point may sit outside).
        for m in (report.topical_precision_at_k, report.ndcg_at_k):
            assert 0.0 <= m.ci_lower <= m.ci_upper <= 1.0

    def test_seed_reproducible(self, tmp_path: Path):
        csv = _write(tmp_path / "ret.csv", _retrieval_rows())
        a = score(base_dir=tmp_path, path=csv, task="retrieval", seed=99, n_resamples=300)
        b = score(base_dir=tmp_path, path=csv, task="retrieval", seed=99, n_resamples=300)
        assert a.ndcg_at_k == b.ndcg_at_k
        assert a.topical_precision_at_k == b.topical_precision_at_k


class TestScoreGrounding:
    def test_returns_report_with_conditional(self, tmp_path: Path):
        csv = _write(tmp_path / "g.csv", _grounding_rows())
        report = score(base_dir=tmp_path, path=csv, task="grounding", seed=1)

        assert isinstance(report, GroundingScoreReport)
        assert report.grounding_presence_rate.method == "wilson"
        assert report.grounding_presence_rate.point == pytest.approx(0.5)
        # one cited query (r1) whose source is fabricated -> rate 1.0 over n=1
        assert report.conditional_fabrication_rate is not None
        assert report.conditional_fabrication_rate.point == pytest.approx(1.0)
        assert report.conditional_fabrication_rate.n == 1
        assert report.conditional_fabrication_rate.method == "wilson"

    def test_conditional_none_when_no_citations(self, tmp_path: Path):
        rows = _grounding_rows()
        for row in rows:
            row["source_cited"] = 0
        csv = _write(tmp_path / "g.csv", rows)

        report = score(base_dir=tmp_path, path=csv, task="grounding")

        assert report.conditional_fabrication_rate is None
        # the Optional None field survives the JSON round-trip
        out = next((tmp_path / "eval" / "scores").glob("*/grounding_scores.json"))
        assert GroundingScoreReport.model_validate_json(out.read_text()) == report


class TestScoreGeneration:
    def test_returns_report_all_wilson(self, tmp_path: Path):
        csv = _write(tmp_path / "gen.csv", _generation_rows())
        report = score(base_dir=tmp_path, path=csv, task="generation", seed=1)

        assert isinstance(report, GenerationScoreReport)
        for m in (
            report.proper_action_rate,
            report.on_topic_rate,
            report.helpfulness_rate,
            report.incompleteness_rate,
            report.unsafe_content_rate,
        ):
            assert isinstance(m, MetricScore)
            assert m.method == "wilson"
        assert report.on_topic_rate.point == pytest.approx(1.0)  # both on-topic


class TestScoreInput:
    def test_records_source_relative_to_workspace(self, tmp_path: Path):
        csv = _write(tmp_path / "ret.csv", _retrieval_rows())

        report = score(base_dir=tmp_path, path=csv, task="retrieval", seed=1)

        # Inside the workspace -> recorded relative to base_dir.
        assert report.source.kind == "direct_path"
        assert report.source.ref == str(csv)
        assert report.source.resolved_path == "ret.csv"

    def test_records_absolute_path_when_outside_workspace(self, tmp_path: Path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        csv = _write(tmp_path / "external" / "ret.csv", _retrieval_rows())  # outside the workspace

        report = score(base_dir=workspace, path=csv, task="retrieval", seed=1)

        # Outside the workspace -> can't be made relative, so recorded absolute.
        assert report.source.resolved_path == str(csv.resolve())

    def test_missing_path_errors(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Scoring input CSV does not exist"):
            score(base_dir=tmp_path, path=tmp_path / "nope.csv", task="retrieval")

    def test_scores_via_export_id(self, tmp_path: Path):
        _write_retrieval_export(tmp_path, "export-a")

        report = score(base_dir=tmp_path, export_id="export-a", task="retrieval", seed=1)

        assert isinstance(report, RetrievalScoreReport)
        assert report.n_examples == 2
        assert report.source.kind == "annotation_export"
        assert report.source.ref == "export-a"
        assert report.source.resolved_path == str(Path("annotation") / "exports" / "export-a" / "retrieval.csv")

    def test_no_selector_falls_back_to_latest_export(self, tmp_path: Path):
        _write_retrieval_export(tmp_path, "export-a")

        report = score(base_dir=tmp_path, task="retrieval", seed=1)  # no selector

        assert report.source.kind == "annotation_export"
        assert report.source.ref == "export-a"

    def test_no_selector_without_exports_errors(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            score(base_dir=tmp_path, task="retrieval")  # nothing to fall back to

    def test_rejects_multiple_selectors(self, tmp_path: Path):
        csv = _write(tmp_path / "ret.csv", _retrieval_rows())

        # path and export_id together is ambiguous - selectors are mutually exclusive, no precedence.
        with pytest.raises(ValueError, match="At most one input selector"):
            score(base_dir=tmp_path, path=csv, export_id="ignored", task="retrieval", seed=1)

    def test_prediction_id_not_supported(self, tmp_path: Path):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            score(base_dir=tmp_path, prediction_id="predict-001", task="retrieval")


class TestScoreConsolidationAndGuard:
    def test_consolidates_duplicate_chunk(self, tmp_path: Path):
        rows = _retrieval_rows()
        rows.append({**rows[0]})  # a second annotator row for (r1, c1) - collapsed by majority
        csv = _write(tmp_path / "ret.csv", rows)

        report = score(base_dir=tmp_path, path=csv, task="retrieval", seed=1)

        assert report.n_examples == 2  # two queries; the duplicate chunk was consolidated

    def test_rejects_non_collapsible_duplicate_chunk_rank(self, tmp_path: Path):
        rows = _retrieval_rows()
        # distinct chunk_id sharing (record_uuid, chunk_rank) -> not collapsible, guard still errors
        rows.append({**rows[0], "chunk_id": "c1b"})
        csv = _write(tmp_path / "ret.csv", rows)

        with pytest.raises(EvalInputSchemaError, match="chunk rank"):
            score(base_dir=tmp_path, path=csv, task="retrieval")

    def test_consolidates_duplicate_record_uuid_grounding(self, tmp_path: Path):
        rows = _grounding_rows()
        rows.append({**rows[0]})  # a second annotator row for r1 - collapsed by majority
        csv = _write(tmp_path / "g.csv", rows)

        report = score(base_dir=tmp_path, path=csv, task="grounding")

        assert report.n_examples == 2  # two records; the duplicate was consolidated
