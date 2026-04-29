"""Operational settings for annotation (workspace topology, distribution)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.settings_base import ResolveSettings


class ArgillaSettings(BaseModel):
    """Argilla connection settings (URL only; API key is resolved from env)."""

    model_config = ConfigDict(extra="forbid")

    api_url: str | None = None


class TaskOverlap(BaseModel):
    """Per-task overlap topology: production and optional calibration thresholds.

    Attributes:
        production_min_submitted: Argilla ``min_submitted`` for the production
            dataset (typically 1; >1 enables full overlap on production).
        calibration_min_submitted: Argilla ``min_submitted`` for the
            calibration dataset. ``None`` disables calibration for this task.
            Default 3 covers Krippendorff alpha plus pairwise Cohen kappa.
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
    dataset_id: str = ""
    workspace_dataset_map: dict[str, dict[Task, TaskOverlap]] = Field(
        default_factory=lambda: {
            "retrieval": {Task.RETRIEVAL: TaskOverlap()},
            "grounding": {Task.GROUNDING: TaskOverlap()},
            "generation": {Task.GENERATION: TaskOverlap()},
        }
    )
    calibration_partition_seed: NonNegativeInt = 0
    include_discarded: bool = False


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
