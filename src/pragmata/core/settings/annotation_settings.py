"""Operational settings for annotation (workspace topology, distribution).

Settings are organised into three scopes — deployment (``AnnotationSettings``),
workspace (``WorkspaceSettings``), task (``TaskSettings``). The inheritable
fields (``production_min_submitted``, ``calibration_min_submitted``, ``locale``)
may be set at any scope; child scopes default to ``INHERIT``. Models hold the **specified**
values exactly as given (``INHERIT`` survives validation, raw inputs round-trip
losslessly through ``model_dump()``). ``resolved_task(workspace_name, task)``
returns the **computed** values after walking task → workspace → deployment —
the CSS "computed value" analogy: first non-``INHERIT`` ancestor wins.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.schemas.annotation_task import Locale, Task
from pragmata.core.settings.settings_base import INHERIT, Inherit, ResolveSettings
from pragmata.core.types import SafePathSegment


class ArgillaSettings(BaseModel):
    """Argilla connection settings (URL only; API key is resolved from env)."""

    model_config = ConfigDict(extra="forbid")

    api_url: str | None = None


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


@dataclass(frozen=True)
class ResolvedTaskSettings:
    """Concrete per-task settings after inheritance resolution (CSS 'computed' values)."""

    production_min_submitted: int
    calibration_min_submitted: int | None
    locale: Locale


class AnnotationSettings(ResolveSettings):
    """Configurable runtime settings for annotation (setup, import, export).

    Controls workspace topology and per-task overlap thresholds. Task definitions
    (Argilla ``rg.Settings`` per task) are hardcoded; see
    ``core/annotation/argilla_task_definitions.py``.

    Attributes:
        production_min_submitted: Argilla ``min_submitted`` for the production
            dataset (typically 1; >1 enables full overlap on production).
            Inherited by workspaces/tasks.
        calibration_min_submitted: Argilla ``min_submitted`` for the calibration
            dataset, or ``None`` to disable calibration for that scope. Default
            3 covers Krippendorff alpha plus pairwise Cohen kappa for IAA.
            Inherited by workspaces/tasks.
        locale: Display locale for Argilla dataset strings (titles, guidelines,
            label option text). Inherited by workspaces/tasks. Identities and
            label values remain stable across locales so exports merge cleanly.
        locale_catalog_dir: Optional directory of user-provided locale YAML
            files. When set, any ``*.yaml`` in this directory is layered on
            top of the bundled catalogs (user wins on stem collision), so a
            deployment can add or override locales without modifying the
            installed package. Must exist if set.
        calibration_fraction: Fraction of records routed to a separate
            calibration dataset for IAA (0.0 disables; deployment-level only).
    """

    argilla: ArgillaSettings = Field(default_factory=ArgillaSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    dataset_id: SafePathSegment = ""
    production_min_submitted: PositiveInt = 1
    calibration_min_submitted: PositiveInt | None = 3
    locale: Locale = "en"
    locale_catalog_dir: Path | None = None
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

    def resolved_task(self, workspace_name: str, task: Task) -> ResolvedTaskSettings:
        """Return the computed task settings: task → workspace → deployment.

        First non-``INHERIT`` ancestor wins. Consumers should call this rather
        than reading inheritable fields directly off ``TaskSettings`` /
        ``WorkspaceSettings``, which hold raw specified values.
        """
        ws = self.workspaces[workspace_name]
        ts = ws.tasks[task]

        production = ts.production_min_submitted
        if isinstance(production, Inherit):
            production = ws.production_min_submitted
        if isinstance(production, Inherit):
            production = self.production_min_submitted

        calibration = ts.calibration_min_submitted
        if isinstance(calibration, Inherit):
            calibration = ws.calibration_min_submitted
        if isinstance(calibration, Inherit):
            calibration = self.calibration_min_submitted

        locale = ts.locale
        if isinstance(locale, Inherit):
            locale = ws.locale
        if isinstance(locale, Inherit):
            locale = self.locale

        return ResolvedTaskSettings(
            production_min_submitted=production,
            calibration_min_submitted=calibration,
            locale=locale,
        )

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
            for task in ws.tasks
            if self.resolved_task(ws_name, task).calibration_min_submitted is None
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
