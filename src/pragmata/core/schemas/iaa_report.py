"""Boundary schemas for IAA (inter-annotator agreement) reports."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from pragmata.core.schemas.annotation_task import Task

AgreementScore = Annotated[float, Field(ge=-1.0, le=1.0)]
Proportion = Annotated[float, Field(ge=0.0, le=1.0)]
CiLevel = Annotated[float, Field(gt=0.0, lt=1.0)]


class LabelAgreement(BaseModel):
    """Agreement metrics for a single binary label.

    Metric fields are ``None`` when there is insufficient overlap to compute
    a value (e.g. fewer than two annotators on any item).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    alpha: AgreementScore | None
    ci_lower: AgreementScore | None
    ci_upper: AgreementScore | None
    n_items: NonNegativeInt
    n_annotators: NonNegativeInt
    pct_agreement: Proportion | None


class AnnotatorPair(BaseModel):
    """Pairwise Cohen's Kappa for one annotator pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotator_a: str
    annotator_b: str
    kappa: AgreementScore
    n_shared_items: NonNegativeInt


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
    n_bootstrap_resamples: PositiveInt
    ci_level: CiLevel
