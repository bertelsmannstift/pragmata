"""Operational settings for annotation (workspace topology, distribution).

Settings are organised into three scopes: deployment (``AnnotationSettings``),
workspace (``WorkspaceSettings``), task (``TaskSettings``). The inheritable
fields (``production_min_submitted``, ``calibration_min_submitted``, ``locale``,
``calibration_fraction``, ``calibration_max_records``) may be set at any scope;
child scopes default to ``INHERIT``. Models hold the **specified** values
exactly as given (``INHERIT`` survives validation, raw inputs round-trip
losslessly through ``model_dump()``). ``resolved_task(workspace_name, task)``
returns the **computed** values after walking task, workspace, deployment
(the CSS "computed value" analogy: first non-``INHERIT`` ancestor wins).

Multi-key maps (e.g. ``constraint_severity: dict[str, Severity]``) use
**sparse-dict overlay** rather than the ``INHERIT`` sentinel: user-supplied keys
override the deployment default for that key only; omitted keys fall through.
This is the natural mechanism for dict-shaped fields; INHERIT is reserved for
single-value primitives.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.annotation.logical_constraints import CONSTRAINT_BY_ID, Severity
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
    calibration_fraction: float | Inherit = INHERIT
    calibration_max_records: PositiveInt | None | Inherit = INHERIT


class WorkspaceSettings(BaseModel):
    """Per-workspace overrides plus the workspace's ``tasks`` mapping.

    Inherited fields default to ``INHERIT`` (use deployment value). ``None`` on
    ``calibration_min_submitted`` explicitly disables calibration for any task
    that inherits from this workspace. ``constraint_severity`` is a sparse
    constraint-id to severity map: only listed constraint_ids override the
    deployment value; omitted constraint_ids fall through to the deployment map.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    production_min_submitted: PositiveInt | Inherit = INHERIT
    calibration_min_submitted: PositiveInt | None | Inherit = INHERIT
    constraint_severity: dict[str, Severity] = Field(default_factory=dict)
    locale: Locale | Inherit = INHERIT
    calibration_fraction: float | Inherit = INHERIT
    calibration_max_records: PositiveInt | None | Inherit = INHERIT
    tasks: dict[Task, TaskSettings]


@dataclass(frozen=True)
class ResolvedTaskSettings:
    """Concrete per-task settings after inheritance resolution (CSS 'computed' values)."""

    production_min_submitted: int
    calibration_min_submitted: int | None
    locale: Locale
    calibration_fraction: float
    calibration_max_records: int | None


def _inherit(*candidates):
    """First non-``Inherit`` candidate. Walks task → workspace → deployment in caller order."""
    return next(c for c in candidates if not isinstance(c, Inherit))


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
        calibration_fraction: Fraction of annotation items routed to the
            calibration dataset for IAA (0.0 disables for that scope).
            Inherited by workspaces/tasks. Cap unit is the annotation item:
            for retrieval that's a chunk, for grounding/generation that's
            a record_uuid.
        calibration_max_records: Optional absolute cap on calibration
            annotation items per task. ``None`` is uncapped (just the
            fractional knob). Smaller of (fraction × N_items, cap) wins.
            Inherited by workspaces/tasks.
    """

    argilla: ArgillaSettings = Field(default_factory=ArgillaSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    dataset_id: SafePathSegment = ""
    production_min_submitted: PositiveInt = 1
    calibration_min_submitted: PositiveInt | None = 3
    locale: Locale = "en"
    locale_catalog_dir: Path | None = None
    calibration_fraction: float = Field(0.1, ge=0.0, le=1.0)
    calibration_max_records: PositiveInt | None = None
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

    def resolved_task(self, workspace_name: str, task: Task) -> ResolvedTaskSettings:
        """Return the computed task settings: task → workspace → deployment.

        First non-``INHERIT`` ancestor wins. Consumers should call this rather
        than reading inheritable fields directly off ``TaskSettings`` /
        ``WorkspaceSettings``, which hold raw specified values.
        """
        ws = self.workspaces[workspace_name]
        ts = ws.tasks[task]

        def at(field: str):
            return _inherit(getattr(ts, field), getattr(ws, field), getattr(self, field))

        return ResolvedTaskSettings(
            production_min_submitted=at("production_min_submitted"),
            calibration_min_submitted=at("calibration_min_submitted"),
            locale=at("locale"),
            calibration_fraction=at("calibration_fraction"),
            calibration_max_records=at("calibration_max_records"),
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
        """Reject configs where a task wants calibration but its overlap is disabled.

        Walks per-(workspace, task) using ``resolved_task`` so per-scope overrides
        on ``calibration_fraction`` and ``calibration_min_submitted`` are honoured.
        A task with resolved ``calibration_fraction > 0`` and resolved
        ``calibration_min_submitted is None`` is incoherent: calibration items
        would be routed but no overlap threshold gates them.
        """
        missing = []
        for ws_name, ws in self.workspaces.items():
            for task in ws.tasks:
                resolved = self.resolved_task(ws_name, task)
                if resolved.calibration_fraction > 0 and resolved.calibration_min_submitted is None:
                    missing.append(f"{ws_name}/{task.value}")
        if missing:
            raise ValueError(
                f"calibration_fraction > 0 but these workspace/task pairs disable "
                f"calibration_min_submitted: {', '.join(missing)}. "
                f"Either set calibration_fraction=0.0 for those scopes or enable "
                f"calibration_min_submitted at the appropriate scope."
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
