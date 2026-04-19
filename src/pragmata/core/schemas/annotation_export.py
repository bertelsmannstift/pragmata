"""Boundary schemas for annotation export records (one per task type)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from pragmata.core.schemas.annotation_task import Task

ResponseStatus = Literal["submitted", "discarded"]


class AnnotationBase(BaseModel):
    """Base fields shared across annotation exports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_uuid: str
    annotator_id: str
    task: Task
    language: str | None
    inserted_at: datetime
    created_at: datetime
    record_status: str
    response_status: ResponseStatus
    discard_reason: str | None = None
    discard_notes: str = ""


def _require_label_answers(model: "AnnotationBase", fields: tuple[str, ...]) -> "AnnotationBase":
    """Enforce: submitted responses must have non-None values for all label fields."""
    if model.response_status == "submitted":
        missing = [f for f in fields if getattr(model, f) is None]
        if missing:
            raise ValueError(f"submitted {model.task.value} annotation missing required label(s): {missing}")
    return model


class RetrievalAnnotation(AnnotationBase):
    """Exported annotation for a single retrieval judgement."""

    task: Literal[Task.RETRIEVAL] = Task.RETRIEVAL
    query: str
    chunk: str
    chunk_id: str
    doc_id: str
    chunk_rank: int
    topically_relevant: bool | None = None
    evidence_sufficient: bool | None = None
    misleading: bool | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check_submitted_has_labels(self) -> "RetrievalAnnotation":
        return _require_label_answers(self, ("topically_relevant", "evidence_sufficient", "misleading"))


class GroundingAnnotation(AnnotationBase):
    """Exported annotation for a single grounding judgement."""

    task: Literal[Task.GROUNDING] = Task.GROUNDING
    answer: str
    context_set: str
    support_present: bool | None = None
    unsupported_claim_present: bool | None = None
    contradicted_claim_present: bool | None = None
    source_cited: bool | None = None
    fabricated_source: bool | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check_submitted_has_labels(self) -> "GroundingAnnotation":
        return _require_label_answers(
            self,
            (
                "support_present",
                "unsupported_claim_present",
                "contradicted_claim_present",
                "source_cited",
                "fabricated_source",
            ),
        )


class GenerationAnnotation(AnnotationBase):
    """Exported annotation for a single generation judgement."""

    task: Literal[Task.GENERATION] = Task.GENERATION
    query: str
    answer: str
    proper_action: bool | None = None
    response_on_topic: bool | None = None
    helpful: bool | None = None
    incomplete: bool | None = None
    unsafe_content: bool | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check_submitted_has_labels(self) -> "GenerationAnnotation":
        return _require_label_answers(
            self,
            ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"),
        )


class RetrievalExportRow(RetrievalAnnotation):
    """Full on-disk CSV row for retrieval: extends RetrievalAnnotation with constraint metadata."""

    constraint_violated: bool
    constraint_details: str = ""


class GroundingExportRow(GroundingAnnotation):
    """Full on-disk CSV row for grounding: extends GroundingAnnotation with constraint metadata."""

    constraint_violated: bool
    constraint_details: str = ""


class GenerationExportRow(GenerationAnnotation):
    """Full on-disk CSV row for generation: extends GenerationAnnotation with constraint metadata."""

    constraint_violated: bool
    constraint_details: str = ""
