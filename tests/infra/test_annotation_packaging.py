"""Tests for annotation packaging extras and pytest integration infrastructure."""

import pathlib

import pytest

PYPROJECT_PATH = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"


def _read_pyproject() -> dict:
    """Parse pyproject.toml via tomllib (3.11+)."""
    import tomllib

    return tomllib.loads(PYPROJECT_PATH.read_text())


def test_annotation_extra_defined() -> None:
    """Project metadata defines an 'annotation' optional extra with argilla."""
    extras = _read_pyproject()["project"]["optional-dependencies"]
    assert "annotation" in extras
    assert any("argilla" in dep for dep in extras["annotation"])


def test_dev_extra_includes_annotation() -> None:
    """The [dev] extra pulls in [annotation]."""
    extras = _read_pyproject()["project"]["optional-dependencies"]
    assert "dev" in extras
    assert any("annotation" in dep for dep in extras["dev"])


def test_integration_marker_registered(pytestconfig: pytest.Config) -> None:
    """The 'integration' marker is registered without warnings."""
    markers = {m.split(":")[0] for m in pytestconfig.getini("markers")}
    assert "integration" in markers
