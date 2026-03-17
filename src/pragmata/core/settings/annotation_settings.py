"""Operational settings for annotation (workspace topology, distribution)."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.settings_base import ResolveSettings


class AnnotationSettings(ResolveSettings):
    """Configurable runtime settings for annotation (setup, import, export).

    Controls workspace topology and task-distribution thresholds.
    Task definitions (Argilla rg.Settings per task) are hardcoded — see
    core/annotation/argilla_task_definitions.py.
    """

    workspace_prefix: str = ""
    workspace_dataset_map: dict[str, list[Task]] = Field(
        default_factory=lambda: {
            "retrieval": [Task.RETRIEVAL],
            "grounding": [Task.GROUNDING],
            "generation": [Task.GENERATION],
        }
    )
    min_submitted: int = 1


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
