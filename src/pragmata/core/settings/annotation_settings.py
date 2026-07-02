"""Operational settings for annotation (workspace topology, distribution).

Settings are organised into three scopes: deployment (``AnnotationSettings``),
workspace (``WorkspaceSettings``), task (``TaskSettings``). The inheritable
fields (``production_min_submitted``, ``calibration_min_submitted``, ``locale``,
``calibration_fraction``) may be set at any scope; child scopes default to
``INHERIT``. Models hold the **specified** values exactly as given
(``INHERIT`` survives validation, raw inputs round-trip losslessly through
``model_dump()``). ``resolved_task(workspace_name, task)`` returns the
**computed** values after walking task → workspace → deployment — the CSS
"computed value" analogy: first non-``INHERIT`` ancestor wins.

Multi-key maps (e.g. ``constraint_severity: dict[str, Severity]``) use
**sparse-dict overlay** rather than the ``INHERIT`` sentinel: user-supplied keys
override the deployment default for that key only; omitted keys fall through.
This is the natural mechanism for dict-shaped fields; INHERIT is reserved for
single-value primitives.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from pragmata.core.annotation.logical_constraints import CONSTRAINT_BY_ID, Severity
from pragmata.core.schemas.annotation_task import Locale, Task
from pragmata.core.settings.settings_base import INHERIT, Inherit, ResolveSettings
from pragmata.core.types import SafePathSegment

T = TypeVar("T")

# Inheritable calibration_fraction override: bounded [0, 1] or the INHERIT sentinel.
CalibrationFractionOverride = Annotated[float, Field(ge=0.0, le=1.0)] | Inherit

# Severity is a deployment concern (not a property of the rule itself); it's user
# configurable. The same LogicalConstraint may b "block" in one deployment and
# "warn" in another. so it lives here rather than as a field on LogicalConstraint.
_DEFAULT_CONSTRAINT_SEVERITY: dict[str, Severity] = {
    "evidence_requires_relevance": Severity.BLOCK,
    "evidence_excludes_misleading": Severity.WARN,
    "contradiction_requires_unsupported": Severity.BLOCK,
    "fabricated_requires_cited": Severity.BLOCK,
}


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
    calibration_fraction: CalibrationFractionOverride = INHERIT


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
    locale: Locale | Inherit = INHERIT
    calibration_fraction: CalibrationFractionOverride = INHERIT
    constraint_severity: dict[str, Severity] = Field(default_factory=dict)
    tasks: dict[Task, TaskSettings]


@dataclass(frozen=True)
class ResolvedTaskSettings:
    """Concrete per-task settings after inheritance resolution (CSS 'computed' values)."""

    production_min_submitted: int
    calibration_min_submitted: int | None
    locale: Locale
    calibration_fraction: float


def _inherit(*candidates: T | Inherit) -> T:
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
            Inherited by workspaces/tasks. The annotation item is a chunk
            for retrieval and a record_uuid for grounding / generation.
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
    constraint_severity: dict[str, Severity] = Field(default_factory=lambda: dict(_DEFAULT_CONSTRAINT_SEVERITY))
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

        return ResolvedTaskSettings(
            production_min_submitted=_inherit(
                ts.production_min_submitted, ws.production_min_submitted, self.production_min_submitted
            ),
            calibration_min_submitted=_inherit(
                ts.calibration_min_submitted, ws.calibration_min_submitted, self.calibration_min_submitted
            ),
            locale=_inherit(ts.locale, ws.locale, self.locale),
            calibration_fraction=_inherit(ts.calibration_fraction, ws.calibration_fraction, self.calibration_fraction),
        )

    @model_validator(mode="before")
    @classmethod
    def _merge_constraint_severity_defaults(cls, data: object) -> object:
        """Overlay user-provided deployment severities onto the built-in defaults.

        Users supply only the constraint_ids they want to override; the rest
        fall through to the field's default.
        """
        if isinstance(data, dict) and isinstance(data.get("constraint_severity"), dict):
            return {**data, "constraint_severity": {**_DEFAULT_CONSTRAINT_SEVERITY, **data["constraint_severity"]}}
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
    def _validate_constraint_severity_complete(self) -> Self:
        """Every known constraint_id must resolve to a deployment-scope severity.

        ``resolved_severity()`` falls through to ``self.constraint_severity`` for
        any constraint_id not overridden at workspace scope. A gap here would
        otherwise surface only as a bare ``KeyError`` deep in severity resolution
        (e.g. mid-widget-render) rather than at construction time - typically
        because a new ``LogicalConstraint`` was added without a matching entry
        in ``_DEFAULT_CONSTRAINT_SEVERITY``.
        """
        missing = set(CONSTRAINT_BY_ID) - set(self.constraint_severity)
        if missing:
            raise ValueError(
                f"deployment constraint_severity is missing entries for known constraint_id(s): "
                f"{sorted(missing)}. Every constraint must have a deployment-scope severity "
                f"(add it to _DEFAULT_CONSTRAINT_SEVERITY)."
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
