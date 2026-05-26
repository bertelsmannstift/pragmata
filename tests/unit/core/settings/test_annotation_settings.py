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
    """``TaskSettings`` is default-free for inherited fields (INHERIT)."""

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
    """``WorkspaceSettings`` is default-free for inherited fields (INHERIT)."""

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


class TestLocaleCatalogDirField:
    def test_none_is_default(self):
        assert AnnotationSettings().locale_catalog_dir is None

    def test_path_accepted_without_existence_check(self, tmp_path):
        # Existence is validated by the consumer (register_catalog_dir), not the field.
        s = AnnotationSettings(locale_catalog_dir=tmp_path / "does-not-exist")
        assert s.locale_catalog_dir == tmp_path / "does-not-exist"


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
        with pytest.raises(ValidationError, match=r"workspace/task pairs"):
            AnnotationSettings(calibration_fraction=0.1, workspaces=_disabled_topology())

    def test_positive_fraction_with_one_enabled_task_still_rejected(self):
        topology = _disabled_topology()
        topology["retrieval"] = WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings(calibration_min_submitted=3)})
        # Mixed: retrieval calibrated, grounding/generation not - still rejected
        # because the validator checks every task, not "any task".
        with pytest.raises(ValidationError, match=r"workspace/task pairs"):
            AnnotationSettings(calibration_fraction=0.1, workspaces=topology)

    def test_error_message_reports_workspace_task_pairs(self):
        with pytest.raises(ValidationError) as excinfo:
            AnnotationSettings(calibration_fraction=0.1, workspaces=_disabled_topology())
        msg = str(excinfo.value)
        assert "retrieval/retrieval" in msg
        assert "grounding/grounding" in msg
        assert "generation/generation" in msg

    def test_per_workspace_task_inheritance(self):
        """Workspace-level calibration_min_submitted=None is inherited by its tasks.

        With calibration_fraction>0, the topology validator resolves each
        task's calibration value through the inheritance walk and reports
        any None resolutions by workspace/task pair.
        """
        workspaces = {
            "retrieval": WorkspaceSettings(
                calibration_min_submitted=None,
                tasks={Task.RETRIEVAL: TaskSettings()},  # inherits None
            ),
            "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
            "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
        }
        with pytest.raises(ValidationError, match=r"retrieval/retrieval"):
            AnnotationSettings(calibration_fraction=0.1, workspaces=workspaces)

    def test_fraction_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_fraction=1.5)
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_fraction=-0.1)


class TestResolvedTask:
    """``resolved_task()`` walks task → workspace → deployment for inherited fields."""

    def test_task_resolves_from_deployment_when_unset(self):
        s = AnnotationSettings(
            production_min_submitted=5,
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.resolved_task("r", Task.RETRIEVAL).production_min_submitted == 5

    def test_task_resolves_from_workspace_when_unset(self):
        s = AnnotationSettings(
            workspaces={
                "r": WorkspaceSettings(
                    production_min_submitted=7,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).production_min_submitted == 7

    def test_workspace_override_wins_over_deployment(self):
        s = AnnotationSettings(
            production_min_submitted=1,
            workspaces={
                "r": WorkspaceSettings(
                    production_min_submitted=4,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).production_min_submitted == 4

    def test_task_override_wins_over_workspace_and_deployment(self):
        s = AnnotationSettings(
            production_min_submitted=1,
            workspaces={
                "r": WorkspaceSettings(
                    production_min_submitted=2,
                    tasks={Task.RETRIEVAL: TaskSettings(production_min_submitted=9)},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).production_min_submitted == 9

    def test_resolves_explicit_disable_at_task(self):
        s = AnnotationSettings(
            calibration_fraction=0.0,
            workspaces={
                "r": WorkspaceSettings(
                    tasks={Task.RETRIEVAL: TaskSettings(calibration_min_submitted=None)},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_min_submitted is None

    def test_resolves_workspace_disable_for_task(self):
        s = AnnotationSettings(
            calibration_fraction=0.0,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_min_submitted=None,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_min_submitted is None

    def test_empty_tasks_dict_no_op(self):
        # An empty tasks dict must not crash settings construction.
        s = AnnotationSettings(
            workspaces={"empty": WorkspaceSettings(tasks={})},
        )
        assert s.workspaces["empty"].tasks == {}

    def test_raw_models_preserve_inherit_sentinel(self):
        """The model holds specified values; ``INHERIT`` is preserved untouched."""
        s = AnnotationSettings(
            production_min_submitted=5,
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.workspaces["r"].production_min_submitted is INHERIT
        assert s.workspaces["r"].tasks[Task.RETRIEVAL].production_min_submitted is INHERIT

    def test_shared_workspace_settings_never_mutated(self):
        """Reused WorkspaceSettings instances must not carry stale resolved values."""
        ws = WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})
        _ = AnnotationSettings(production_min_submitted=1, workspaces={"retrieval": ws})
        s2 = AnnotationSettings(production_min_submitted=42, workspaces={"retrieval": ws})
        assert s2.resolved_task("retrieval", Task.RETRIEVAL).production_min_submitted == 42
        # Raw instance untouched — INHERIT sentinel preserved across both constructions
        assert ws.production_min_submitted is INHERIT
        assert ws.tasks[Task.RETRIEVAL].production_min_submitted is INHERIT

    def test_locale_resolves_from_deployment_when_unset(self):
        s = AnnotationSettings(
            locale="de",
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.resolved_task("r", Task.RETRIEVAL).locale == "de"

    def test_locale_resolves_from_workspace_when_task_unset(self):
        s = AnnotationSettings(
            workspaces={
                "r": WorkspaceSettings(
                    locale="de",
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).locale == "de"

    def test_locale_task_override_wins_over_workspace_and_deployment(self):
        s = AnnotationSettings(
            locale="en",
            workspaces={
                "r": WorkspaceSettings(
                    locale="de",
                    tasks={Task.RETRIEVAL: TaskSettings(locale="en")},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).locale == "en"

    def test_locale_default_resolves_to_en(self):
        s = AnnotationSettings(
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.resolved_task("r", Task.RETRIEVAL).locale == "en"


class TestTaskUniquenessValidator:
    def test_same_task_in_two_workspaces_rejected(self):
        with pytest.raises(ValidationError, match=r"task 'retrieval' appears in multiple workspaces"):
            AnnotationSettings(
                workspaces={
                    "a": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
                    "b": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()}),
                },
            )


class TestCalibrationTopologyResolution:
    """``_check_calibration_topology`` resolves inheritance for each task.

    A workspace-level ``calibration_min_submitted=None`` must surface in the
    topology check via the inheritance walk, even though the raw task models
    still hold ``INHERIT``.
    """

    def test_workspace_disable_surfaces_via_resolution(self):
        with pytest.raises(ValidationError, match=r"r/retrieval"):
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


class TestCalibrationFractionInheritance:
    """``calibration_fraction`` is inheritable across deployment / workspace / task."""

    def test_deployment_default_flows_to_task(self):
        s = AnnotationSettings(
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_fraction == 0.1

    def test_workspace_override_wins_over_deployment(self):
        s = AnnotationSettings(
            calibration_fraction=0.1,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_fraction=0.05,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_fraction == 0.05

    def test_task_override_wins_over_workspace_and_deployment(self):
        s = AnnotationSettings(
            calibration_fraction=0.1,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_fraction=0.2,
                    tasks={Task.RETRIEVAL: TaskSettings(calibration_fraction=0.5)},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_fraction == 0.5

    def test_zero_at_task_disables_just_that_task(self):
        """A task can opt out of calibration without disabling other tasks."""
        s = AnnotationSettings(
            calibration_fraction=0.1,
            workspaces={
                "r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings(calibration_fraction=0.0)}),
                "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_fraction == 0.0
        assert s.resolved_task("g", Task.GROUNDING).calibration_fraction == 0.1


class TestCalibrationMaxRecordsInheritance:
    """``calibration_max_records`` is inheritable across deployment / workspace / task."""

    def test_deployment_default_is_uncapped(self):
        s = AnnotationSettings(
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_max_records is None

    def test_deployment_cap_flows_to_task(self):
        s = AnnotationSettings(
            calibration_max_records=200,
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_max_records == 200

    def test_workspace_cap_overrides_deployment(self):
        s = AnnotationSettings(
            calibration_max_records=200,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_max_records=50,
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_max_records == 50

    def test_task_cap_overrides_workspace_and_deployment(self):
        s = AnnotationSettings(
            calibration_max_records=200,
            workspaces={
                "r": WorkspaceSettings(
                    calibration_max_records=100,
                    tasks={Task.RETRIEVAL: TaskSettings(calibration_max_records=20)},
                ),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_max_records == 20

    def test_zero_cap_rejected(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_max_records=0)

    def test_negative_cap_rejected(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_max_records=-1)


class TestPerTaskTopologyValidator:
    """``_check_calibration_topology`` walks per-(workspace, task) using resolved values."""

    def test_task_with_zero_fraction_does_not_require_min_submitted(self):
        """A task with calibration_fraction=0 may have calibration_min_submitted=None."""
        s = AnnotationSettings(
            calibration_fraction=0.1,
            workspaces={
                "r": WorkspaceSettings(
                    tasks={
                        Task.RETRIEVAL: TaskSettings(
                            calibration_fraction=0.0,
                            calibration_min_submitted=None,
                        )
                    },
                ),
                "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
            },
        )
        assert s.resolved_task("r", Task.RETRIEVAL).calibration_min_submitted is None

    def test_task_with_positive_fraction_requires_min_submitted(self):
        with pytest.raises(ValidationError, match=r"r/retrieval"):
            AnnotationSettings(
                calibration_fraction=0.0,  # deployment off
                workspaces={
                    "r": WorkspaceSettings(
                        tasks={
                            Task.RETRIEVAL: TaskSettings(
                                calibration_fraction=0.1,  # task overrides to on
                                calibration_min_submitted=None,  # but no overlap
                            )
                        },
                    ),
                    "g": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                    "x": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
                },
            )


class TestYamlRoundtrip:
    """Specified values round-trip losslessly; ``resolved_task()`` gives computed values."""

    def test_model_dump_preserves_inherit_sentinel(self):
        """``model_dump()`` returns specified values: child scopes stay ``INHERIT``."""
        s = AnnotationSettings(
            production_min_submitted=5,
            workspaces={"r": WorkspaceSettings(tasks={Task.RETRIEVAL: TaskSettings()})},
        )
        dumped = s.model_dump()
        assert dumped["production_min_submitted"] == 5
        assert dumped["workspaces"]["r"]["production_min_submitted"] is INHERIT
        assert dumped["workspaces"]["r"]["tasks"][Task.RETRIEVAL]["production_min_submitted"] is INHERIT
        # But resolved_task gives the computed value.
        assert s.resolved_task("r", Task.RETRIEVAL).production_min_submitted == 5

    def test_sparse_yaml_input_roundtrips_to_resolved_value(self):
        data = yaml.safe_load(
            "production_min_submitted: 4\ncalibration_fraction: 0.0\nworkspaces: {r: {tasks: {retrieval: {}}}}\n"
        )
        s = AnnotationSettings(**data)
        assert s.resolved_task("r", Task.RETRIEVAL).production_min_submitted == 4


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
