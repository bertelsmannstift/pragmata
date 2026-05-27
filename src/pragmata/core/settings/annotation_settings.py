"""Operational settings for annotation (workspace topology, distribution).

Settings are organised into three scopes: deployment (``AnnotationSettings``),
workspace (``WorkspaceSettings``), task (``TaskSettings``). The inheritable
fields (``production_min_submitted``, ``calibration_min_submitted``) may be set
at any scope; child scopes default to ``INHERIT``. Models hold the **specified**
values exactly as given (``INHERIT`` survives validation, raw inputs round-trip
losslessly through ``model_dump()``). ``resolved_task(workspace_name, task)``
returns the **computed** values after walking task, workspace, deployment
(the CSS "computed value" analogy: first non-``INHERIT`` ancestor wins).

Multi-key maps (e.g. ``constraint_severity: dict[str, Severity]``,
``field_render_mode: dict[str, FieldRenderMode]``) use **sparse-dict overlay**
rather than the ``INHERIT`` sentinel: user-supplied keys override the deployment
default for that key only; omitted keys fall through. This is the natural
mechanism for dict-shaped fields; INHERIT is reserved for single-value primitives.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.annotation.logical_constraints import CONSTRAINT_BY_ID, Severity
from pragmata.core.schemas.annotation_task import TEXT_FIELDS, FieldRenderMode, Task
from pragmata.core.settings.settings_base import INHERIT, Inherit, ResolveSettings
from pragmata.core.types import SafePathSegment


class ArgillaSettings(BaseModel):
    """Argilla connection settings (URL only; API key is resolved from env)."""

    model_config = ConfigDict(extra="forbid")

    api_url: str | None = None


class TaskSettings(BaseModel):
    """Per-task overrides; inherited fields default to ``INHERIT`` (use workspace value).

    ``field_render_mode`` is a sparse override map: only listed field names
    override the workspace/deployment value; omitted keys fall through.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    field_render_mode: dict[str, FieldRenderMode] = Field(default_factory=dict)


class WorkspaceSettings(BaseModel):
    """Per-workspace overrides plus the workspace's ``tasks`` mapping.

    Inherited fields default to ``INHERIT`` (use deployment value). ``None`` on
    ``calibration_min_submitted`` explicitly disables calibration for any task
    that inherits from this workspace. ``constraint_severity`` and
    ``field_render_mode`` are sparse override maps: only listed keys override
    the deployment value; omitted keys fall through to the deployment map.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    constraint_severity: dict[str, Severity] = Field(default_factory=dict)
    field_render_mode: dict[str, FieldRenderMode] = Field(default_factory=dict)
    tasks: dict[Task, TaskSettings]


@dataclass(frozen=True)
class ResolvedTaskSettings:
    """Concrete per-task settings after inheritance resolution (CSS 'computed' values)."""

    production_min_submitted: int
    calibration_min_submitted: int | None


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
        calibration_fraction: Fraction of records routed to a separate
            calibration dataset for IAA (0.0 disables; deployment-level only).
        field_render_mode: Sparse override map of TextField name (e.g.
            ``"answer"``) to render mode (``"plain"`` or ``"markdown"``).
            Markdown rendering also handles inline raw HTML, useful when
            content comes from a pipeline that emits HTML. Per-workspace
            overrides live on ``WorkspaceSettings.field_render_mode``;
            ``resolved_render_mode(workspace_name, field_name)`` walks
            workspace then deployment. Unknown keys raise at validation
            time. Defaults populate every known field with ``"plain"``.
    """

    argilla: ArgillaSettings = Field(default_factory=ArgillaSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    dataset_id: SafePathSegment = ""
    production_min_submitted: PositiveInt = 1
    calibration_min_submitted: PositiveInt | None = 3
    calibration_fraction: float = Field(0.1, ge=0.0, le=1.0)
    calibration_partition_seed: NonNegativeInt = 0
    include_discarded: bool = False
    constraint_severity: dict[str, Severity] = Field(
        default_factory=lambda: {
            "evidence_requires_relevance": "block",
            "evidence_excludes_misleading": "warn",
            "contradiction_requires_unsupported": "block",
            "fabricated_requires_cited": "block",
        }
    )
    field_render_mode: dict[str, FieldRenderMode] = Field(
        default_factory=lambda: {name: "plain" for names in TEXT_FIELDS.values() for name in names}
    )
    workspaces: dict[str, WorkspaceSettings] = Field(
        default_factory=lambda: {
            "retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
            "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
            "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
        }
    )

    def task_to_workspace(self) -> dict[Task, str]:
        """Map each task to its owning workspace name (validated 1:1 by ``_validate_task_uniqueness``)."""
        return {task: ws_name for ws_name, ws in self.workspaces.items() for task in ws.tasks}

    def resolved_severity(self, workspace_name: str, constraint_id: str) -> Severity:
        """Return the computed severity for a logical constraint: workspace then deployment.

        Severity is per-constraint (not per-task), so this walks only the two
        scopes where it can be set. Deployment defaults are guaranteed complete
        by the merge validator on ``constraint_severity``; unknown ``constraint_id``
        raises ``KeyError``.
        """
        ws = self.workspaces[workspace_name]
        if constraint_id in ws.constraint_severity:
            return ws.constraint_severity[constraint_id]
        return self.constraint_severity[constraint_id]

    def resolved_render_mode(self, workspace_name: str, task: Task, field_name: str) -> FieldRenderMode:
        """Return the computed render mode for a TextField: task → workspace → deployment.

        Walks all three scopes via sparse overlay (per-key, not whole-dict):
        the first scope that lists ``field_name`` wins. Deployment defaults
        are guaranteed complete by the merge validator on ``field_render_mode``,
        so the workspace and task scopes only need to be sparse overrides.
        Unknown ``field_name`` raises ``KeyError``.
        """
        ws = self.workspaces[workspace_name]
        ts = ws.tasks[task]
        for scope in (ts.field_render_mode, ws.field_render_mode, self.field_render_mode):
            if field_name in scope:
                return scope[field_name]
        raise KeyError(field_name)

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

        return ResolvedTaskSettings(
            production_min_submitted=production,
            calibration_min_submitted=calibration,
        )

    @model_validator(mode="before")
    @classmethod
    def _merge_constraint_severity_defaults(cls, data: object) -> object:
        """Overlay user-provided deployment severities onto the built-in defaults.

        Users supply only the constraint_ids they want to override; the rest
        fall through to the field's default.
        """
        if isinstance(data, dict) and isinstance(data.get("constraint_severity"), dict):
            defaults = cls.model_fields["constraint_severity"].default_factory()
            data["constraint_severity"] = {**defaults, **data["constraint_severity"]}
        return data

    @model_validator(mode="before")
    @classmethod
    def _merge_field_render_mode_defaults(cls, data: object) -> object:
        """Overlay user-provided deployment render modes onto the built-in defaults.

        Users supply only the field names they want to override; the rest fall
        through to the field's default (all ``"plain"``).
        """
        if isinstance(data, dict) and isinstance(data.get("field_render_mode"), dict):
            defaults = cls.model_fields["field_render_mode"].default_factory()
            data["field_render_mode"] = {**defaults, **data["field_render_mode"]}
        return data

    @model_validator(mode="after")
    def _validate_constraint_severity_keys(self) -> Self:
        known = set(CONSTRAINT_BY_ID)
        for scope_name, overrides in [("deployment", self.constraint_severity)] + [
            (f"workspace {ws_name!r}", ws.constraint_severity) for ws_name, ws in self.workspaces.items()
        ]:
            unknown = set(overrides) - known
            if unknown:
                raise ValueError(
                    f"{scope_name} constraint_severity references unknown constraint_id(s): "
                    f"{sorted(unknown)}. Known constraint_ids: {sorted(known)}."
                )
        return self

    @model_validator(mode="after")
    def _validate_field_render_mode_keys(self) -> Self:
        """Reject ``field_render_mode`` keys not declared in ``TEXT_FIELDS`` for the scope.

        Deployment scope accepts any field name that exists in any task;
        workspace and task scopes are constrained to the field names that
        actually belong to that scope's tasks (so an ``"answer"`` override
        on a retrieval-only workspace is caught at validation time).
        """
        all_fields = {name for names in TEXT_FIELDS.values() for name in names}
        unknown = set(self.field_render_mode) - all_fields
        if unknown:
            raise ValueError(
                f"deployment field_render_mode references unknown field name(s): "
                f"{sorted(unknown)}. Known field names: {sorted(all_fields)}."
            )
        for ws_name, ws in self.workspaces.items():
            ws_fields = {name for task in ws.tasks for name in TEXT_FIELDS[task]}
            unknown = set(ws.field_render_mode) - ws_fields
            if unknown:
                raise ValueError(
                    f"workspace {ws_name!r} field_render_mode references field(s) "
                    f"not in any of its tasks: {sorted(unknown)}. "
                    f"Allowed: {sorted(ws_fields)}."
                )
            for task, ts in ws.tasks.items():
                task_fields = TEXT_FIELDS[task]
                unknown = set(ts.field_render_mode) - task_fields
                if unknown:
                    raise ValueError(
                        f"workspace {ws_name!r} task {task.value!r} field_render_mode "
                        f"references field(s) not in this task: {sorted(unknown)}. "
                        f"Allowed: {sorted(task_fields)}."
                    )
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
