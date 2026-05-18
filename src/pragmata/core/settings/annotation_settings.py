"""Operational settings for annotation (workspace topology, distribution)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.schemas.annotation_task import Locale, Task
from pragmata.core.settings.settings_base import ResolveSettings
from pragmata.core.types import SafePathSegment


class ArgillaSettings(BaseModel):
    """Argilla connection settings (URL only; API key is resolved from env)."""

    model_config = ConfigDict(extra="forbid")

    api_url: str | None = None


class TaskOverlap(BaseModel):
    """Per-task overlap topology: production and (optional) calibration thresholds.

    Calibration is declared here so future imports can route a subset of
    records to a separate calibration dataset for IAA.

    Attributes:
        production_min_submitted: Argilla ``min_submitted`` for the production
            dataset (typically 1; >1 enables full overlap on production).
        calibration_min_submitted: Argilla ``min_submitted`` for a future
            calibration dataset, or ``None`` to disable calibration for this
            task. Default 3 covers Krippendorff alpha plus pairwise Cohen
            kappa.
    """

    model_config = ConfigDict(extra="forbid")

    production_min_submitted: PositiveInt = 1
    calibration_min_submitted: PositiveInt | None = 3


class AnnotationSettings(ResolveSettings):
    """Configurable runtime settings for annotation (setup, import, export).

    Controls workspace topology and per-task overlap thresholds.
    Task definitions (Argilla rg.Settings per task) are hardcoded — see
    core/annotation/argilla_task_definitions.py.
    """

    argilla: ArgillaSettings = Field(default_factory=ArgillaSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    dataset_id: SafePathSegment = ""
    locale: Locale = Locale.EN
    workspace_dataset_map: dict[str, dict[Task, TaskOverlap]] = Field(
        default_factory=lambda: {
            "retrieval": {Task.RETRIEVAL: TaskOverlap()},
            "grounding": {Task.GROUNDING: TaskOverlap()},
            "generation": {Task.GENERATION: TaskOverlap()},
        }
    )
    calibration_fraction: float = Field(0.1, ge=0.0, le=1.0)
    calibration_partition_seed: NonNegativeInt = 0
    include_discarded: bool = False

    @model_validator(mode="after")
    def _check_calibration_topology(self) -> Self:
        if self.calibration_fraction <= 0.0:
            return self
        missing = sorted(
            {
                task.value
                for task_overlaps in self.workspace_dataset_map.values()
                for task, overlap in task_overlaps.items()
                if overlap.calibration_min_submitted is None
            }
        )
        if missing:
            raise ValueError(
                f"calibration_fraction={self.calibration_fraction} > 0 but topology has no "
                f"calibration dataset for tasks: {missing}. Either set calibration_fraction=0.0 "
                f"or enable calibration in workspace_dataset_map."
            )
        return self

    @model_validator(mode="after")
    def _validate_task_uniqueness(self) -> Self:
        seen: set[Task] = set()
        for task_overlaps in self.workspace_dataset_map.values():
            for task in task_overlaps:
                if task in seen:
                    raise ValueError(
                        f"task {task.value!r} appears in multiple workspace_dataset_map entries; "
                        "each task must belong to exactly one workspace."
                    )
                seen.add(task)
        return self


@dataclass
class UserSpec:
    """Specification for provisioning an Argilla user account.

    Attributes:
        username: Argilla login name.
        role: Account role — ``"owner"`` or ``"annotator"``.
        workspaces: Workspace base names to assign this user to.
        password: Explicit password. If None, one is auto-generated and
            returned in SetupResult.generated_passwords.
    """

    username: str
    role: Literal["owner", "annotator"]
    workspaces: list[str] = field(default_factory=list)
    password: str | None = None
