"""Output contract for annotation export provenance metadata."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from pragmata.core.schemas.annotation_task import Task


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
