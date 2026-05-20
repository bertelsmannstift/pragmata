"""Unit tests for AnnotationSettings and UserSpec."""

import pytest
import yaml
from pydantic import ValidationError

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import (
    AnnotationSettings,
    ArgillaSettings,
    TaskSettings,
    UserSpec,
    WorkspaceSettings,
)
from pragmata.core.settings.settings_base import INHERIT


class TestAnnotationSettingsDefaults:
    def test_workspaces_default(self):
        s = AnnotationSettings()
        # Default factory builds three single-task workspaces.
        assert set(s.workspaces) == {"retrieval", "grounding", "generation"}
        assert set(s.workspaces["retrieval"].tasks) == {Task.RETRIEVAL}
        assert set(s.workspaces["grounding"].tasks) == {Task.GROUNDING}
        assert set(s.workspaces["generation"].tasks) == {Task.GENERATION}

    def test_dataset_id_default(self):
        s = AnnotationSettings()
        assert s.dataset_id == ""

    def test_calibration_partition_seed_default(self):
        s = AnnotationSettings()
        assert s.calibration_partition_seed == 0

    def test_production_min_submitted_deployment_default(self):
        s = AnnotationSettings()
        assert s.production_min_submitted == 1

    def test_calibration_min_submitted_deployment_default(self):
        s = AnnotationSettings()
        assert s.calibration_min_submitted == 3

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(nonexistent_field="value")


class TestTaskSettingsDefaults:
    """``TaskSettings`` is default-free for cascade fields (INHERIT)."""

    def test_production_default_inherit(self):
        t = TaskSettings()
        assert t.production_min_submitted is INHERIT

    def test_calibration_default_inherit(self):
        t = TaskSettings()
        assert t.calibration_min_submitted is INHERIT

    def test_calibration_can_be_disabled(self):
        t = TaskSettings(calibration_min_submitted=None)
        assert t.calibration_min_submitted is None

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            TaskSettings(nonexistent_field=1)  # type: ignore[call-arg]

    def test_production_must_be_positive(self):
        with pytest.raises(ValidationError):
            TaskSettings(production_min_submitted=0)


class TestWorkspaceSettingsDefaults:
    """``WorkspaceSettings`` is default-free for cascade fields (INHERIT)."""

    def test_production_default_inherit(self):
        w = WorkspaceSettings(tasks={})
        assert w.production_min_submitted is INHERIT

    def test_calibration_default_inherit(self):
        w = WorkspaceSettings(tasks={})
        assert w.calibration_min_submitted is INHERIT

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            WorkspaceSettings(tasks={}, nonexistent_field=1)  # type: ignore[call-arg]

    def test_calibration_can_be_disabled(self):
        w = WorkspaceSettings(tasks={}, calibration_min_submitted=None)
        assert w.calibration_min_submitted is None


class TestAnnotationSettingsResolve:
    def test_resolve_with_no_args_returns_defaults(self):
        s = AnnotationSettings.resolve()
        assert s.calibration_partition_seed == 0
        assert s.dataset_id == ""

    def test_resolve_overrides_dataset_id(self):
        s = AnnotationSettings.resolve(overrides={"dataset_id": "run1"})
        assert s.dataset_id == "run1"

    def test_resolve_overrides_partition_seed(self):
        s = AnnotationSettings.resolve(overrides={"calibration_partition_seed": 42})
        assert s.calibration_partition_seed == 42

    @pytest.mark.parametrize("bad", ["a/b", "..", "foo..bar", " run", "run\\sub"])
    def test_unsafe_dataset_id_rejected(self, bad):
        with pytest.raises(ValidationError):
            AnnotationSettings(dataset_id=bad)


def _disabled_topology() -> dict[str, WorkspaceSettings]:
    """Workspaces where every task explicitly disables calibration."""
    return {
        "retrieval": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings(calibration_min_submitted=None)}),
        "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings(calibration_min_submitted=None)}),
        "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings(calibration_min_submitted=None)}),
    }


class TestCalibrationTopologyValidator:
    def test_zero_fraction_with_disabled_calibration_ok(self):
        AnnotationSettings(calibration_fraction=0.0, workspaces=_disabled_topology())

    def test_positive_fraction_with_disabled_calibration_rejected(self):
        with pytest.raises(ValidationError, match=r"\(workspace, task\) pairs"):
            AnnotationSettings(calibration_fraction=0.1, workspaces=_disabled_topology())

    def test_positive_fraction_with_one_enabled_task_still_rejected(self):
        topology = _disabled_topology()
        topology["retrieval"] = WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings(calibration_min_submitted=3)})
        # Mixed: retrieval calibrated, grounding/generation not - still rejected
        # because the validator checks every task, not "any task".
        with pytest.raises(ValidationError, match=r"\(workspace, task\) pairs"):
            AnnotationSettings(calibration_fraction=0.1, workspaces=topology)

    def test_error_message_reports_workspace_task_pairs(self):
        with pytest.raises(ValidationError) as excinfo:
            AnnotationSettings(calibration_fraction=0.1, workspaces=_disabled_topology())
        msg = str(excinfo.value)
        assert "('retrieval', 'retrieval')" in msg
        assert "('grounding', 'grounding')" in msg
        assert "('generation', 'generation')" in msg

    def test_per_workspace_task_cascade(self):
        """Workspace-level calibration_min_submitted=None cascades to its inheriting tasks.

        With calibration_fraction>0, the topology validator (running after the
        cascade) sees the propagated None on each task and reports them by
        (workspace, task) pair.
        """
        workspaces = {
            "retrieval": WorkspaceSettings(
                calibration_min_submitted=None,
                tasks={Task.RETRIEVAL: TaskSettings()},  # inherits None
            ),
            "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
            "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
        }
        with pytest.raises(ValidationError, match=r"\('retrieval', 'retrieval'\)"):
            AnnotationSettings(calibration_fraction=0.1, workspaces=workspaces)

    def test_fraction_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_fraction=1.5)
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_fraction=-0.1)


class TestCascadePropagation:
    """``_propagate_cascade`` replaces INHERIT with the parent-scope value."""

    def test_workspace_inherits_from_annotation(self):
        s = AnnotationSettings(
            production_min_submitted=5,
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.workspaces["r"].production_min_submitted == 5

    def test_task_inherits_from_workspace(self):
        s = AnnotationSettings(
            workspaces={
                "r": WorkspaceSettings(
                    production_min_submitted=7,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.workspaces["r"].tasks[Task.RETRIEVAL].production_min_submitted == 7

    def test_respects_explicit_workspace_override(self):
        s = AnnotationSettings(
            production_min_submitted=1,
            workspaces={
                "r": WorkspaceSettings(
                    production_min_submitted=4,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.workspaces["r"].production_min_submitted == 4

    def test_respects_explicit_task_override(self):
        s = AnnotationSettings(
            production_min_submitted=1,
            workspaces={
                "r": WorkspaceSettings(
                    production_min_submitted=2,
                    tasks={Task.RETRIEVAL: TaskSettings(production_min_submitted=9)},
                ),
            },
        )
        assert s.workspaces["r"].tasks[Task.RETRIEVAL].production_min_submitted == 9

    def test_preserves_explicit_disable_at_task(self):
        s = AnnotationSettings(
            calibration_fraction=0.0,
            workspaces={
                "r": WorkspaceSettings(
                    tasks={Task.RETRIEVAL: TaskSettings(calibration_min_submitted=None)},
                ),
            },
        )
        assert s.workspaces["r"].tasks[Task.RETRIEVAL].calibration_min_submitted is None

    def test_workspace_disable_cascades_to_inheriting_tasks(self):
        s = AnnotationSettings(
            calibration_fraction=0.0,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_min_submitted=None,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.workspaces["r"].tasks[Task.RETRIEVAL].calibration_min_submitted is None

    def test_empty_tasks_dict_no_op(self):
        # An empty tasks dict must not crash the cascade walk.
        s = AnnotationSettings(
            workspaces={"empty": WorkspaceSettings(tasks={})},
        )
        assert s.workspaces["empty"].tasks == {}
        # Workspace-level field still inherits from annotation.
        assert s.workspaces["empty"].production_min_submitted == 1


class TestTaskUniquenessValidator:
    def test_same_task_in_two_workspaces_rejected(self):
        with pytest.raises(ValidationError, match=r"task 'retrieval' appears in multiple workspaces"):
            AnnotationSettings(
                workspaces={
                    "a": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
                    "b": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
                },
            )


class TestValidatorOrdering:
    """``_propagate_cascade`` must run before ``_check_calibration_topology``.

    Construct settings where the topology check would FAIL if it saw the
    pre-cascade ``INHERIT`` placeholder on task fields. Under correct
    ordering, the cascade resolves the workspace-level ``None`` into each
    task's ``calibration_min_submitted`` first, then the topology check
    sees the concrete ``None`` and raises with a (workspace, task) pair.

    If the topology validator ran first (pre-cascade), it would see
    ``INHERIT`` on tasks - not ``None`` - and silently accept the config,
    which would be wrong.
    """

    def test_propagation_runs_before_topology_check(self):
        with pytest.raises(ValidationError, match=r"\('r', 'retrieval'\)"):
            AnnotationSettings(
                calibration_fraction=0.1,
                workspaces={
                    "r": WorkspaceSettings(
                        calibration_min_submitted=None,
                        tasks={Task.RETRIEVAL: TaskSettings()},
                    ),
                    "g": WorkspaceSettings(
                        calibration_min_submitted=None,
                        tasks={Task.GROUNDING: TaskSettings()},
                    ),
                    "x": WorkspaceSettings(
                        calibration_min_submitted=None,
                        tasks={Task.GENERATION: TaskSettings()},
                    ),
                },
            )


class TestYamlRoundtrip:
    """Documented non-goal: sparse YAML inputs become concrete in ``model_dump()``."""

    def test_model_dump_is_post_cascade_lossy(self):
        data = yaml.safe_load(
            "production_min_submitted: 4\ncalibration_fraction: 0.0\nworkspaces: {r: {tasks: {retrieval: {}}}}\n"
        )
        dumped = AnnotationSettings(**data).model_dump()
        assert dumped["workspaces"]["r"]["production_min_submitted"] == 4
        assert dumped["workspaces"]["r"]["tasks"]["retrieval"]["production_min_submitted"] == 4


class TestArgillaSettings:
    def test_defaults_to_none_api_url(self):
        s = AnnotationSettings()
        assert isinstance(s.argilla, ArgillaSettings)
        assert s.argilla.api_url is None

    def test_resolve_accepts_api_url_override(self):
        s = AnnotationSettings.resolve(overrides={"argilla": {"api_url": "http://localhost:6900"}})
        assert s.argilla.api_url == "http://localhost:6900"

    def test_resolve_loads_api_url_from_config(self):
        s = AnnotationSettings.resolve(config={"argilla": {"api_url": "http://cfg:6900"}})
        assert s.argilla.api_url == "http://cfg:6900"

    def test_overrides_win_over_config_for_api_url(self):
        s = AnnotationSettings.resolve(
            config={"argilla": {"api_url": "http://cfg:6900"}},
            overrides={"argilla": {"api_url": "http://override:6900"}},
        )
        assert s.argilla.api_url == "http://override:6900"

    def test_argilla_settings_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            ArgillaSettings(api_key="sekret")  # type: ignore[call-arg]


class TestUserSpec:
    def test_required_fields(self):
        u = UserSpec(username="alice", role="annotator", workspaces=["retrieval_grounding"])
        assert u.username == "alice"
        assert u.role == "annotator"
        assert u.workspaces == ["retrieval_grounding"]
        assert u.password is None

    def test_with_password(self):
        u = UserSpec(username="bob", role="owner", workspaces=[], password="secret")
        assert u.password == "secret"

    def test_role_owner(self):
        u = UserSpec(username="admin", role="owner", workspaces=[])
        assert u.role == "owner"
