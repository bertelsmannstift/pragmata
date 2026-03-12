"""Unit tests for AnnotationSetupSettings and UserSpec."""

import pytest
from chatboteval.annotation.settings import AnnotationSetupSettings, UserSpec
from pydantic import ValidationError

from chatboteval.core.schemas.annotation_task import Task


class TestAnnotationSetupSettingsDefaults:
    def test_workspace_dataset_map_default(self):
        s = AnnotationSetupSettings()
        assert s.workspace_dataset_map == {
            "retrieval_grounding": [Task.RETRIEVAL, Task.GROUNDING],
            "generation": [Task.GENERATION],
        }

    def test_workspace_prefix_default(self):
        s = AnnotationSetupSettings()
        assert s.workspace_prefix == ""

    def test_min_submitted_default(self):
        s = AnnotationSetupSettings()
        assert s.min_submitted == 1

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AnnotationSetupSettings(nonexistent_field="value")


class TestAnnotationSetupSettingsResolve:
    def test_resolve_with_no_args_returns_defaults(self):
        s = AnnotationSetupSettings.resolve()
        assert s.min_submitted == 1
        assert s.workspace_prefix == ""

    def test_resolve_overrides_min_submitted(self):
        s = AnnotationSetupSettings.resolve(overrides={"min_submitted": 3})
        assert s.min_submitted == 3

    def test_resolve_overrides_prefix(self):
        s = AnnotationSetupSettings.resolve(overrides={"workspace_prefix": "pb"})
        assert s.workspace_prefix == "pb"

    def test_resolve_config_layer(self):
        s = AnnotationSetupSettings.resolve(config={"min_submitted": 2})
        assert s.min_submitted == 2

    def test_resolve_overrides_win_over_config(self):
        s = AnnotationSetupSettings.resolve(
            config={"min_submitted": 2},
            overrides={"min_submitted": 5},
        )
        assert s.min_submitted == 5


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
