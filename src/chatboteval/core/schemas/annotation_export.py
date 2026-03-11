"""Boundary schemas for annotation export records (one per task type)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from chatboteval.core.schemas.annotation_task import Task


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


class RetrievalAnnotation(AnnotationBase):
    """Exported annotation for a single retrieval judgement."""

    input_query: str
    chunk: str
    chunk_id: str
    doc_id: str
    chunk_rank: int
    topically_relevant: bool
    evidence_sufficient: bool
    misleading: bool
    notes: str = ""


class GroundingAnnotation(AnnotationBase):
    """Exported annotation for a single grounding judgement."""

    answer: str
    context_set: str
    support_present: bool
    unsupported_claim_present: bool
    contradicted_claim_present: bool
    source_cited: bool
    fabricated_source: bool
    notes: str = ""


class GenerationAnnotation(AnnotationBase):
    """Exported annotation for a single generation judgement."""

    query: str
    answer: str
    proper_action: bool
    response_on_topic: bool
    helpful: bool
    incomplete: bool
    unsafe_content: bool
    notes: str = ""
