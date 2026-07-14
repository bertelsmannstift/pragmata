"""Tests for eval score output schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import (
    EvalTrainMeta,
    GenerationScoreReport,
    GroundingScoreReport,
    MetricScore,
    RetrievalScoreReport,
)

NOW = datetime(2026, 5, 28, 13, 30, tzinfo=UTC)


def _metric(point: float = 0.5, *, method: str = "bootstrap", n: int = 5) -> dict[str, object]:
    """Build a valid ``MetricScore`` payload; the CI collapses to the point estimate."""
    return {"point": point, "ci_lower": point, "ci_upper": point, "method": method, "n": n}


@pytest.fixture()
def valid_retrieval_report():
    """Valid retrieval score report fields."""
    return {
        "annotation_export_id": "export-1",
        "created_at": NOW,
        "n_examples": 5,
        "top_k": 3,
        "ci_level": 0.95,
        "n_bootstrap_resamples": 1000,
        "topical_precision_at_k": _metric(0.6, method="bootstrap"),
        "sufficiency_hit_at_k": _metric(0.8, method="wilson"),
        "sufficiency_rate_at_k": _metric(0.4, method="bootstrap"),
        "misleading_context_rate_at_k": _metric(0.0, method="bootstrap"),
        "mean_reciprocal_rank_at_k": _metric(0.7, method="bootstrap"),
        "ndcg_at_k": _metric(1.0, method="bootstrap"),
    }


@pytest.fixture()
def valid_grounding_report():
    """Valid grounding score report fields."""
    return {
        "annotation_export_id": "export-1",
        "created_at": NOW,
        "n_examples": 5,
        "ci_level": 0.95,
        "n_bootstrap_resamples": 1000,
        "grounding_presence_rate": _metric(0.8, method="wilson"),
        "unsupported_claim_rate": _metric(0.2, method="wilson"),
        "contradiction_rate": _metric(0.0, method="wilson"),
        "citation_presence_rate": _metric(0.6, method="wilson"),
        "conditional_fabrication_rate": _metric(0.1, method="wilson", n=3),
    }


@pytest.fixture()
def valid_generation_report():
    """Valid generation score report fields."""
    return {
        "annotation_export_id": "export-1",
        "created_at": NOW,
        "n_examples": 5,
        "ci_level": 0.95,
        "n_bootstrap_resamples": 1000,
        "proper_action_rate": _metric(1.0, method="wilson"),
        "on_topic_rate": _metric(0.8, method="wilson"),
        "helpfulness_rate": _metric(0.6, method="wilson"),
        "incompleteness_rate": _metric(0.2, method="wilson"),
        "unsafe_content_rate": _metric(0.0, method="wilson"),
    }


class TestMetricScore:
    """Tests for the nested per-metric score model."""

    def test_constructs_and_exposes_fields(self) -> None:
        score = MetricScore(point=0.6, ci_lower=0.5, ci_upper=0.7, method="wilson", n=10)

        assert score.point == 0.6
        assert (score.ci_lower, score.ci_upper) == (0.5, 0.7)
        assert score.method == "wilson"
        assert score.n == 10

    @pytest.mark.parametrize("method", ["bootstrap", "wilson"])
    def test_accepts_both_methods(self, method: str) -> None:
        assert MetricScore(**_metric(0.5, method=method)).method == method

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValidationError):
            MetricScore(**_metric(0.5, method="jackknife"))

    @pytest.mark.parametrize("field", ["point", "ci_lower", "ci_upper"])
    @pytest.mark.parametrize("bad_value", [-0.01, 1.01])
    def test_rejects_out_of_range_rates(self, field: str, bad_value: float) -> None:
        payload = _metric(0.5)
        payload[field] = bad_value

        with pytest.raises(ValidationError):
            MetricScore(**payload)

    def test_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValidationError):
            MetricScore(**_metric(0.5, n=0))

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            MetricScore(**_metric(0.5), unexpected="value")

    def test_is_frozen(self) -> None:
        score = MetricScore(**_metric(0.5))
        with pytest.raises(ValidationError):
            score.point = 0.9  # type: ignore[misc]


def test_eval_train_meta_accepts_valid_payload() -> None:
    """EvalTrainMeta captures the Pragmata-owned run/task link."""
    meta = EvalTrainMeta(
        run_id="train-run-1",
        created_at=NOW,
        task=Task.RETRIEVAL,
        annotation_export_id="export-1",
    )

    assert meta.run_id == "train-run-1"
    assert meta.created_at == NOW
    assert meta.task == Task.RETRIEVAL
    assert meta.annotation_export_id == "export-1"


def test_eval_train_meta_defaults_created_at_and_export_id() -> None:
    """EvalTrainMeta supports standalone training with an internally stamped timestamp."""
    meta = EvalTrainMeta(run_id="train-run-1", task=Task.GROUNDING)

    assert meta.created_at.tzinfo is UTC
    assert meta.annotation_export_id is None


def test_eval_train_meta_rejects_extra_fields() -> None:
    """EvalTrainMeta rejects accidental artifact-shape drift."""
    with pytest.raises(ValidationError):
        EvalTrainMeta.model_validate(
            {
                "run_id": "train-run-1",
                "task": "generation",
                "label_names": ["helpful"],
            }
        )


def test_retrieval_report_constructs(valid_retrieval_report):
    """Retrieval score report constructs with retrieval task identity."""
    report = RetrievalScoreReport(**valid_retrieval_report)

    assert report.task == Task.RETRIEVAL
    assert report.top_k == 3
    assert report.ci_level == 0.95
    assert report.n_bootstrap_resamples == 1000
    assert report.ndcg_at_k.point == 1.0
    assert report.annotation_export_id == "export-1"


def test_grounding_report_constructs(valid_grounding_report):
    """Grounding score report constructs with grounding task identity."""
    report = GroundingScoreReport(**valid_grounding_report)

    assert report.task == Task.GROUNDING
    assert report.grounding_presence_rate.point == 0.8
    assert report.conditional_fabrication_rate.n == 3
    assert report.annotation_export_id == "export-1"


def test_generation_report_constructs(valid_generation_report):
    """Generation score report constructs with generation task identity."""
    report = GenerationScoreReport(**valid_generation_report)

    assert report.task == Task.GENERATION
    assert report.helpfulness_rate.point == 0.6
    assert report.annotation_export_id == "export-1"


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture", "wrong_task"),
    [
        (RetrievalScoreReport, "valid_retrieval_report", Task.GROUNDING),
        (GroundingScoreReport, "valid_grounding_report", Task.GENERATION),
        (GenerationScoreReport, "valid_generation_report", Task.RETRIEVAL),
    ],
)
def test_score_reports_reject_wrong_task(report_cls, fields_fixture, wrong_task, request):
    """Task-specific score reports reject mismatched task values."""
    fields = request.getfixturevalue(fields_fixture).copy()
    fields["task"] = wrong_task

    with pytest.raises(ValidationError):
        report_cls(**fields)


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture"),
    [
        (RetrievalScoreReport, "valid_retrieval_report"),
        (GroundingScoreReport, "valid_grounding_report"),
        (GenerationScoreReport, "valid_generation_report"),
    ],
)
def test_annotation_export_id_is_optional(report_cls, fields_fixture, request):
    """Score reports can be written without annotation-export provenance."""
    fields = request.getfixturevalue(fields_fixture).copy()
    fields.pop("annotation_export_id")

    report = report_cls(**fields)

    assert report.annotation_export_id is None


def test_grounding_allows_undefined_conditional_fabrication_rate(valid_grounding_report):
    """Conditional fabrication rate may be absent when no cited examples exist."""
    valid_grounding_report["conditional_fabrication_rate"] = None

    report = GroundingScoreReport(**valid_grounding_report)

    assert report.conditional_fabrication_rate is None


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture"),
    [
        (RetrievalScoreReport, "valid_retrieval_report"),
        (GroundingScoreReport, "valid_grounding_report"),
        (GenerationScoreReport, "valid_generation_report"),
    ],
)
def test_score_reports_reject_extra_fields(report_cls, fields_fixture, request):
    """Score report schemas reject accidental artifact-shape drift."""
    fields = request.getfixturevalue(fields_fixture).copy()
    fields["unexpected"] = "value"

    with pytest.raises(ValidationError):
        report_cls(**fields)


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture", "metric"),
    [
        (RetrievalScoreReport, "valid_retrieval_report", "topical_precision_at_k"),
        (RetrievalScoreReport, "valid_retrieval_report", "sufficiency_hit_at_k"),
        (RetrievalScoreReport, "valid_retrieval_report", "sufficiency_rate_at_k"),
        (RetrievalScoreReport, "valid_retrieval_report", "misleading_context_rate_at_k"),
        (RetrievalScoreReport, "valid_retrieval_report", "mean_reciprocal_rank_at_k"),
        (RetrievalScoreReport, "valid_retrieval_report", "ndcg_at_k"),
        (GroundingScoreReport, "valid_grounding_report", "grounding_presence_rate"),
        (GroundingScoreReport, "valid_grounding_report", "unsupported_claim_rate"),
        (GroundingScoreReport, "valid_grounding_report", "contradiction_rate"),
        (GroundingScoreReport, "valid_grounding_report", "citation_presence_rate"),
        (GroundingScoreReport, "valid_grounding_report", "conditional_fabrication_rate"),
        (GenerationScoreReport, "valid_generation_report", "proper_action_rate"),
        (GenerationScoreReport, "valid_generation_report", "on_topic_rate"),
        (GenerationScoreReport, "valid_generation_report", "helpfulness_rate"),
        (GenerationScoreReport, "valid_generation_report", "incompleteness_rate"),
        (GenerationScoreReport, "valid_generation_report", "unsafe_content_rate"),
    ],
)
def test_metric_fields_require_metric_score(report_cls, fields_fixture, metric, request):
    """Every metric field requires a MetricScore, not the bare float it used to be."""
    fields = request.getfixturevalue(fields_fixture).copy()

    fields[metric] = 0.5
    with pytest.raises(ValidationError):
        report_cls(**fields)


def test_retrieval_top_k_must_be_positive(valid_retrieval_report):
    """Retrieval score reports require a positive top-k value."""
    valid_retrieval_report["top_k"] = 0

    with pytest.raises(ValidationError):
        RetrievalScoreReport(**valid_retrieval_report)


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture"),
    [
        (RetrievalScoreReport, "valid_retrieval_report"),
        (GroundingScoreReport, "valid_grounding_report"),
        (GenerationScoreReport, "valid_generation_report"),
    ],
)
@pytest.mark.parametrize("bad_ci_level", [0.0, 1.0])
def test_ci_level_must_be_strictly_between_zero_and_one(report_cls, fields_fixture, bad_ci_level, request):
    """ci_level is a confidence level in the open interval (0, 1)."""
    fields = request.getfixturevalue(fields_fixture).copy()
    fields["ci_level"] = bad_ci_level

    with pytest.raises(ValidationError):
        report_cls(**fields)


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture"),
    [
        (RetrievalScoreReport, "valid_retrieval_report"),
        (GroundingScoreReport, "valid_grounding_report"),
        (GenerationScoreReport, "valid_generation_report"),
    ],
)
def test_n_bootstrap_resamples_must_be_positive(report_cls, fields_fixture, request):
    """The recorded bootstrap-resample count must be a positive integer."""
    fields = request.getfixturevalue(fields_fixture).copy()
    fields["n_bootstrap_resamples"] = 0

    with pytest.raises(ValidationError):
        report_cls(**fields)


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture", "run_field"),
    [
        (RetrievalScoreReport, "valid_retrieval_report", "ci_level"),
        (RetrievalScoreReport, "valid_retrieval_report", "n_bootstrap_resamples"),
        (GroundingScoreReport, "valid_grounding_report", "ci_level"),
        (GenerationScoreReport, "valid_generation_report", "n_bootstrap_resamples"),
    ],
)
def test_run_level_uncertainty_fields_are_required(report_cls, fields_fixture, run_field, request):
    """Reports must record the run-level uncertainty settings."""
    fields = request.getfixturevalue(fields_fixture).copy()
    fields.pop(run_field)

    with pytest.raises(ValidationError):
        report_cls(**fields)


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture"),
    [
        (RetrievalScoreReport, "valid_retrieval_report"),
        (GroundingScoreReport, "valid_grounding_report"),
        (GenerationScoreReport, "valid_generation_report"),
    ],
)
def test_score_reports_round_trip_via_json(report_cls, fields_fixture, request):
    """Score reports (with nested MetricScores) round-trip through JSON."""
    original = report_cls(**request.getfixturevalue(fields_fixture))

    restored = report_cls.model_validate(original.model_dump(mode="json"))

    assert restored == original


def test_score_report_serialises_iso_8601_datetime(valid_retrieval_report):
    """Score report datetimes serialize as ISO-8601 timestamps."""
    dumped = RetrievalScoreReport(**valid_retrieval_report).model_dump(mode="json")

    assert dumped["created_at"] == "2026-05-28T13:30:00Z"


@pytest.mark.parametrize(
    ("report_cls", "fields_fixture"),
    [
        (RetrievalScoreReport, "valid_retrieval_report"),
        (GroundingScoreReport, "valid_grounding_report"),
        (GenerationScoreReport, "valid_generation_report"),
    ],
)
def test_score_report_notes_default_empty(report_cls, fields_fixture, request):
    """Score report notes default to empty string."""
    report = report_cls(**request.getfixturevalue(fields_fixture))

    assert report.notes == ""
