"""Boundary schemas for annotation export records (one per task type) and run provenance."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from pragmata.core.schemas.annotation_task import DiscardReason, Task

ResponseStatus = Literal["submitted", "discarded"]


class AnnotationBase(BaseModel):
    """Base fields shared across annotation exports.

    Subclasses declare task-specific label fields as ``bool | None``; a submitted
    response must carry non-None values for all of them (Argilla enforces this
    at the UI level). Discarded responses are allowed to leave labels unset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_uuid: str
    annotator_id: str
    task: Task
    language: str | None
    inserted_at: datetime
    created_at: datetime
    record_status: str
    response_status: ResponseStatus
    discard_reason: DiscardReason | None = None
    discard_notes: str | None = None

    @model_validator(mode="after")
    def _submitted_has_all_labels(self) -> "AnnotationBase":
        if self.response_status != "submitted":
            return self
        missing = [
            name
            for name, info in type(self).model_fields.items()
            if info.annotation == bool | None and getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"submitted {self.task.value} annotation missing required label(s): {missing}")
        return self


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


class AnnotationExportMeta(BaseModel):
    """Schema for annotation export run provenance (sidecar to the task CSVs)."""

    model_config = ConfigDict(extra="forbid")

    export_id: str
    created_at: datetime
    dataset_id: str | None
    tasks: list[Task]
    include_discarded: bool
    row_counts: dict[Task, NonNegativeInt]
    n_annotators: dict[Task, NonNegativeInt]
    constraint_summary: dict[str, NonNegativeInt]

    @model_validator(mode="after")
    def validate_keys_match_tasks(self) -> "AnnotationExportMeta":
        """Enforce per-task dicts cover exactly the exported tasks."""
        expected = set(self.tasks)
        if set(self.row_counts) != expected:
            raise ValueError("row_counts keys must match tasks")
        if set(self.n_annotators) != expected:
            raise ValueError("n_annotators keys must match tasks")
        return self
