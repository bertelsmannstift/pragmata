"""Operational settings for annotation (workspace topology, distribution)."""

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


class TaskSettings(BaseModel):
    """Per-task overrides.

    Default-free for cascade fields: every cascade field defaults to ``INHERIT``,
    meaning "no override at this scope — use the parent (workspace) value."
    Concrete opinionated defaults live exclusively on ``AnnotationSettings``;
    this class only carries values an admin has explicitly overridden at task
    scope.

    Attributes:
        production_min_submitted: Argilla ``min_submitted`` for the production
            dataset, or ``INHERIT`` to use the workspace/deployment value.
        calibration_min_submitted: Argilla ``min_submitted`` for the calibration
            dataset, ``None`` to explicitly disable calibration for this task,
            or ``INHERIT`` to use the workspace/deployment value.
        locale: UI locale for Argilla dataset titles/questions/guidelines, or
            ``INHERIT`` to use the workspace/deployment value.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    locale: Locale | Inherit = INHERIT


class WorkspaceSettings(BaseModel):
    """Per-workspace overrides.

    Default-free for cascade fields: every cascade field defaults to ``INHERIT``,
    meaning "no override at this scope — use the parent (deployment) value."
    Concrete opinionated defaults live exclusively on ``AnnotationSettings``;
    this class only carries values an admin has explicitly overridden at
    workspace scope, plus the 1-to-N ``tasks`` mapping.

    Attributes:
        production_min_submitted: Workspace-level override for production
            ``min_submitted``, or ``INHERIT`` to use the deployment value.
        calibration_min_submitted: Workspace-level override for calibration
            ``min_submitted``, ``None`` to explicitly disable calibration for
            tasks that inherit, or ``INHERIT`` to use the deployment value.
        locale: Workspace-level override for UI locale, or ``INHERIT`` to use
            the deployment value.
        tasks: Mapping of tasks owned by this workspace to their per-task
            overrides.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    locale: Locale | Inherit = INHERIT
    tasks: dict[Task, TaskSettings]


class AnnotationSettings(ResolveSettings):
    """Configurable runtime settings for annotation (setup, import, export).

    Controls workspace topology and per-task overlap thresholds. Carries
    opinionated deployment-level defaults for cascade fields; per-workspace and
    per-task overrides live on ``WorkspaceSettings``/``TaskSettings`` and are
    propagated downward by ``_propagate_cascade`` at materialisation.

    Task definitions (Argilla rg.Settings per task) are hardcoded — see
    core/annotation/argilla_task_definitions.py.
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

    _CASCADE_FIELDS: ClassVar[tuple[str, ...]] = (
        "production_min_submitted",
        "calibration_min_submitted",
        "locale",
    )

    @model_validator(mode="after")
    def _propagate_cascade(self) -> Self:
        for ws in self.workspaces.values():
            for field_name in self._CASCADE_FIELDS:
                if isinstance(getattr(ws, field_name), _InheritType):
                    setattr(ws, field_name, getattr(self, field_name))
            for task_settings in ws.tasks.values():
                for field_name in self._CASCADE_FIELDS:
                    if isinstance(getattr(task_settings, field_name), _InheritType):
                        setattr(task_settings, field_name, getattr(ws, field_name))
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
        missing: list[tuple[str, str]] = []
        for ws_name, ws in self.workspaces.items():
            for task, task_settings in ws.tasks.items():
                if task_settings.calibration_min_submitted is None:
                    missing.append((ws_name, task.value))
        if missing:
            raise ValueError(
                f"calibration_fraction={self.calibration_fraction} > 0 but these "
                f"(workspace, task) pairs disable calibration: {missing}. "
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
