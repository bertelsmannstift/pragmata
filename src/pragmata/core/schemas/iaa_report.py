"""Boundary schemas for IAA (inter-annotator agreement) reports."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from pragmata.core.schemas.annotation_task import Task


class LabelAgreement(BaseModel):
    """Agreement metrics for a single binary label.

    Metric fields are ``None`` when there is insufficient overlap to compute
    a value (e.g. fewer than two annotators on any item).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    alpha: float | None
    ci_lower: float | None
    ci_upper: float | None
    n_items: int
    n_annotators: int
    pct_agreement: float | None


class AnnotatorPair(BaseModel):
    """Pairwise Cohen's Kappa for one annotator pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotator_a: str
    annotator_b: str
    kappa: float
    n_shared_items: int


class TaskAgreement(BaseModel):
    """IAA results for one annotation task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: Task
    labels: list[LabelAgreement]
    pairwise_kappa: list[AnnotatorPair]


class IaaReport(BaseModel):
    """Full IAA report across all analysed tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_id: str
    created_at: datetime
    tasks: list[TaskAgreement]
    n_bootstrap_resamples: int
    ci_level: float
