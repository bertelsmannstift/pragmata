"""Unit tests for AnnotationSettings and UserSpec."""

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import (
    AnnotationSettings,
    ArgillaSettings,
    TaskOverlap,
    UserSpec,
)


class TestAnnotationSettingsDefaults:
    def test_workspace_dataset_map_default(self):
        s = AnnotationSettings()
        assert s.workspace_dataset_map == {
            "retrieval": {Task.RETRIEVAL: TaskOverlap()},
            "grounding": {Task.GROUNDING: TaskOverlap()},
            "generation": {Task.GENERATION: TaskOverlap()},
        }

    def test_dataset_id_default(self):
        s = AnnotationSettings()
        assert s.dataset_id == ""

    def test_calibration_partition_seed_default(self):
        s = AnnotationSettings()
        assert s.calibration_partition_seed == 0

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(nonexistent_field="value")


class TestTaskOverlapDefaults:
    def test_production_default_one(self):
        o = TaskOverlap()
        assert o.production_min_submitted == 1

    def test_calibration_default_three(self):
        o = TaskOverlap()
        assert o.calibration_min_submitted == 3

    def test_calibration_can_be_disabled(self):
        o = TaskOverlap(calibration_min_submitted=None)
        assert o.calibration_min_submitted is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            TaskOverlap(nonexistent_field=1)  # type: ignore[call-arg]

    def test_production_must_be_positive(self):
        with pytest.raises(ValidationError):
            TaskOverlap(production_min_submitted=0)


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


class TestCalibrationTopologyValidator:
    def _disabled_topology(self) -> dict:
        from pragmata.core.schemas.annotation_task import Task

        return {
            "retrieval": {Task.RETRIEVAL: TaskOverlap(calibration_min_submitted=None)},
            "grounding": {Task.GROUNDING: TaskOverlap(calibration_min_submitted=None)},
            "generation": {Task.GENERATION: TaskOverlap(calibration_min_submitted=None)},
        }

    def test_zero_fraction_with_disabled_calibration_ok(self):
        AnnotationSettings(calibration_fraction=0.0, workspace_dataset_map=self._disabled_topology())

    def test_positive_fraction_with_disabled_calibration_rejected(self):
        with pytest.raises(ValidationError, match="topology has no calibration dataset"):
            AnnotationSettings(calibration_fraction=0.1, workspace_dataset_map=self._disabled_topology())

    def test_positive_fraction_with_at_least_one_enabled_task_ok(self):
        from pragmata.core.schemas.annotation_task import Task

        topology = self._disabled_topology()
        topology["retrieval"][Task.RETRIEVAL] = TaskOverlap(calibration_min_submitted=3)
        # Mixed: retrieval calibrated, grounding/generation not -> still rejected
        # because the validator checks every task, not "any task".
        with pytest.raises(ValidationError, match="topology has no calibration dataset"):
            AnnotationSettings(calibration_fraction=0.1, workspace_dataset_map=topology)

    def test_fraction_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_fraction=1.5)
        with pytest.raises(ValidationError):
            AnnotationSettings(calibration_fraction=-0.1)


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
