"""Operational settings for annotation (workspace topology, distribution).

Settings are organised into three scopes — deployment (``AnnotationSettings``),
workspace (``WorkspaceSettings``), task (``TaskSettings``). Fields listed in
``AnnotationSettings._INHERITED_FIELDS`` may be set at any scope; child scopes
default to ``INHERIT`` and adopt the nearest non-inherited ancestor value at
materialisation. CSS-style inheritance: explicit override at any scope wins;
otherwise look upward.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.schemas.annotation_task import Locale, Task
from pragmata.core.settings.settings_base import INHERIT, Inherit, ResolveSettings, _InheritType
from pragmata.core.types import SafePathSegment


class ArgillaSettings(BaseModel):
    """Argilla connection settings (URL only; API key is resolved from env)."""

    model_config = ConfigDict(extra="forbid")

    api_url: str | None = None


def _inherit(child: BaseModel, parent: BaseModel, fields: tuple[str, ...]) -> None:
    """Replace ``INHERIT`` placeholders on ``child`` with the matching ``parent`` values."""
    for name in fields:
        if isinstance(getattr(child, name), _InheritType):
            setattr(child, name, getattr(parent, name))


class TaskSettings(BaseModel):
    """Per-task overrides; inherited fields default to ``INHERIT`` (use workspace value)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    locale: Locale | Inherit = INHERIT


class WorkspaceSettings(BaseModel):
    """Per-workspace overrides plus the workspace's ``tasks`` mapping.

    Inherited fields default to ``INHERIT`` (use deployment value). ``None`` on
    ``calibration_min_submitted`` explicitly disables calibration for any task
    that inherits from this workspace.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    locale: Locale | Inherit = INHERIT
    tasks: dict[Task, TaskSettings]


class AnnotationSettings(ResolveSettings):
    """Configurable runtime settings for annotation (setup, import, export).

    Controls workspace topology and per-task overlap thresholds. Deployment-level
    fields in ``_INHERITED_FIELDS`` are inherited by workspaces and tasks unless
    overridden — see module docstring for the inheritance model. Task definitions
    (Argilla ``rg.Settings`` per task) are hardcoded; see
    ``core/annotation/argilla_task_definitions.py``.
    """

    argilla: ArgillaSettings = Field(default_factory=ArgillaSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    dataset_id: SafePathSegment = ""
    production_min_submitted: PositiveInt = 1
    calibration_min_submitted: PositiveInt | None = 3
    locale: Locale = Locale.EN
    calibration_fraction: float = Field(0.1, ge=0.0, le=1.0)
    calibration_partition_seed: NonNegativeInt = 0
    include_discarded: bool = False
    workspaces: dict[str, WorkspaceSettings] = Field(
        default_factory=lambda: {
            "retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
            "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
        }
    )

    _INHERITED_FIELDS: ClassVar[tuple[str, ...]] = (
        "production_min_submitted",
        "calibration_min_submitted",
        "locale",
    )

    @model_validator(mode="after")
    def _propagate_inheritance(self) -> Self:
        for ws in self.workspaces.values():
            _inherit(ws, self, self._INHERITED_FIELDS)
            for task in ws.tasks.values():
                _inherit(task, ws, self._INHERITED_FIELDS)
        return self

    @model_validator(mode="after")
    def _validate_task_uniqueness(self) -> Self:
        seen: set[Task] = set()
        for ws in self.workspaces.values():
            for task in ws.tasks:
                if task in seen:
                    raise ValueError(
                        f"task {task.value!r} appears in multiple workspaces; "
                        "each task must belong to exactly one workspace."
                    )
                seen.add(task)
        return self

    @model_validator(mode="after")
    def _check_calibration_topology(self) -> Self:
        if self.calibration_fraction <= 0.0:
            return self
        missing = [
            f"{ws_name}/{task.value}"
            for ws_name, ws in self.workspaces.items()
            for task, ts in ws.tasks.items()
            if ts.calibration_min_submitted is None
        ]
        if missing:
            raise ValueError(
                f"calibration_fraction={self.calibration_fraction} > 0 but these "
                f"workspace/task pairs disable calibration: {', '.join(missing)}. "
                f"Either set calibration_fraction=0.0 or enable calibration at "
                f"the appropriate scope."
            )
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
