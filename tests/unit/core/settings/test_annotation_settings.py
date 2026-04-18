"""Unit tests for AnnotationSettings and UserSpec."""

import pytest
from pydantic import ValidationError

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings, ArgillaSettings, UserSpec


class TestAnnotationSettingsDefaults:
    def test_workspace_dataset_map_default(self):
        s = AnnotationSettings()
        assert s.workspace_dataset_map == {
            "retrieval": [Task.RETRIEVAL],
            "grounding": [Task.GROUNDING],
            "generation": [Task.GENERATION],
        }

    def test_workspace_prefix_default(self):
        s = AnnotationSettings()
        assert s.workspace_prefix == ""

    def test_min_submitted_default(self):
        s = AnnotationSettings()
        assert s.min_submitted == 1

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AnnotationSettings(nonexistent_field="value")


class TestAnnotationSettingsResolve:
    def test_resolve_with_no_args_returns_defaults(self):
        s = AnnotationSettings.resolve()
        assert s.min_submitted == 1
        assert s.workspace_prefix == ""

    def test_resolve_overrides_min_submitted(self):
        s = AnnotationSettings.resolve(overrides={"min_submitted": 3})
        assert s.min_submitted == 3

    def test_resolve_overrides_prefix(self):
        s = AnnotationSettings.resolve(overrides={"workspace_prefix": "pb"})
        assert s.workspace_prefix == "pb"

    def test_resolve_config_layer(self):
        s = AnnotationSettings.resolve(config={"min_submitted": 2})
        assert s.min_submitted == 2

    def test_resolve_overrides_win_over_config(self):
        s = AnnotationSettings.resolve(
            config={"min_submitted": 2},
            overrides={"min_submitted": 5},
        )
        assert s.min_submitted == 5


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
