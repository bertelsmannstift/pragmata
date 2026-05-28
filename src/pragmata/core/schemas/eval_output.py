"""Output schemas for eval score artifacts."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from pragmata.core.schemas.annotation_task import Task

type Rate = Annotated[float, Field(ge=0.0, le=1.0)]


class RetrievalScoreReport(BaseModel):
    """Schema for retrieval_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.RETRIEVAL] = Task.RETRIEVAL
    annotation_export_id: str | None = None
    created_at: datetime
    n_examples: PositiveInt
    top_k: PositiveInt
    topical_precision_at_k: Rate
    sufficiency_hit_at_k: Rate
    sufficiency_rate_at_k: Rate
    misleading_context_rate_at_k: Rate
    mean_reciprocal_rank_at_k: Rate
    ndcg_at_k: Rate


class GroundingScoreReport(BaseModel):
    """Schema for grounding_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.GROUNDING] = Task.GROUNDING
    annotation_export_id: str | None = None
    created_at: datetime
    n_examples: PositiveInt
    grounding_presence_rate: Rate
    unsupported_claim_rate: Rate
    contradiction_rate: Rate
    citation_presence_rate: Rate
    conditional_fabrication_rate: Rate | None = None


class GenerationScoreReport(BaseModel):
    """Schema for generation_scores.json."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[Task.GENERATION] = Task.GENERATION
    annotation_export_id: str | None = None
    created_at: datetime
    n_examples: PositiveInt
    proper_action_rate: Rate
    on_topic_rate: Rate
    helpfulness_rate: Rate
    incompleteness_rate: Rate
    unsafe_content_rate: Rate
