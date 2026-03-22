"""Tests for the shared workspace path resolver."""

from pathlib import Path

import pytest

from pragmata.core.paths import WorkspacePaths


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    """Canonical workspace path bundle rooted in a temporary directory."""
    return WorkspacePaths.from_base_dir(tmp_path / "my_workspace")


def test_from_base_dir_resolves_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory resolves relative base directories to absolute paths."""
    monkeypatch.chdir(tmp_path)

    workspace_paths = WorkspacePaths.from_base_dir(Path("relative_dir"))

    assert workspace_paths.base_dir == (tmp_path / "relative_dir").resolve()
    assert workspace_paths.base_dir.is_absolute()


def test_tool_root_returns_tool_directory(workspace: WorkspacePaths) -> None:
    """Bundle returns the tool root under the workspace base directory."""
    assert workspace.tool_root("my_tool") == workspace.base_dir / "my_tool"


def test_run_root_returns_run_directory(workspace: WorkspacePaths) -> None:
    """Bundle returns the default run root under the tool runs directory."""
    assert workspace.run_root(tool="my_tool", run_id="id-1") == (workspace.base_dir / "my_tool" / "runs" / "id-1")
