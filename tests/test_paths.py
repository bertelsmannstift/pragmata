"""Tests for the shared workspace path resolver."""

from pathlib import Path
from typing import Any

import pytest

from chatboteval.core.paths.paths import WorkspacePaths, coerce_path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    """Canonical workspace path bundle rooted in a temporary directory."""
    return WorkspacePaths.from_base_dir(tmp_path / "my_workspace")


@pytest.mark.parametrize(
    ("input_path", "expected"),
    [
        ("workspace/data", Path("workspace/data")),
        (Path("workspace/data"), Path("workspace/data")),
    ],
)
def test_coerce_path_returns_path_for_valid_inputs(
    input_path: str | Path,
    expected: Path,
) -> None:
    """Helper returns a Path for valid string and Path inputs."""
    assert coerce_path(input_path) == expected


def test_coerce_path_returns_same_path_instance() -> None:
    """Helper returns the original Path instance unchanged."""
    input_path = Path("workspace/data")

    assert coerce_path(input_path) is input_path


def test_coerce_path_raises_type_error_for_invalid_input() -> None:
    """Helper rejects inputs that are neither str nor Path."""
    invalid_input: Any = 123

    with pytest.raises(TypeError, match="Expected path as str or Path"):
        coerce_path(invalid_input)


def test_from_base_dir_resolves_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory resolves relative base directories to absolute paths."""
    monkeypatch.chdir(tmp_path)

    workspace_paths = WorkspacePaths.from_base_dir("relative_dir")

    assert workspace_paths.base_dir == (tmp_path / "relative_dir").resolve()
    assert workspace_paths.base_dir.is_absolute()


def test_tool_root_returns_tool_directory(workspace: WorkspacePaths) -> None:
    """Bundle returns the tool root under the workspace base directory."""
    assert workspace.tool_root("my_tool") == workspace.base_dir / "my_tool"


def test_run_root_returns_run_directory(workspace: WorkspacePaths) -> None:
    """Bundle returns the default run root under the tool runs directory."""
    assert workspace.run_root(tool="my_tool", run_id="id-1") == (workspace.base_dir / "my_tool" / "runs" / "id-1")


def test_under_base_resolves_relative_path_under_workspace(
    workspace: WorkspacePaths,
) -> None:
    """Bundle resolves relative paths under the workspace base directory."""
    assert workspace.under_base("sub/file.txt") == (workspace.base_dir / "sub/file.txt").resolve()


def test_under_base_returns_absolute_path_as_resolved(
    workspace: WorkspacePaths,
    tmp_path: Path,
) -> None:
    """Bundle preserves absolute paths instead of rebasing them."""
    absolute_path = tmp_path / "outside.txt"

    assert workspace.under_base(absolute_path) == absolute_path.resolve()
