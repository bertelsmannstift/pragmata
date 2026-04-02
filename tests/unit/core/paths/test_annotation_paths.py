"""Unit tests for annotation path bundles."""

from pathlib import Path

import pytest

from pragmata.core.paths.annotation_paths import resolve_annotation_paths, resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
    return WorkspacePaths.from_base_dir(tmp_path)


# ---------------------------------------------------------------------------
# AnnotationPaths
# ---------------------------------------------------------------------------


class TestAnnotationPaths:
    def test_tool_root(self, workspace: WorkspacePaths) -> None:
        paths = resolve_annotation_paths(workspace=workspace)
        assert paths.tool_root == workspace.tool_root("annotation")

    def test_ensure_dirs_creates_tool_root(self, workspace: WorkspacePaths) -> None:
        paths = resolve_annotation_paths(workspace=workspace)
        assert not paths.tool_root.exists()
        paths.ensure_dirs()
        assert paths.tool_root.exists()

    def test_ensure_dirs_returns_self(self, workspace: WorkspacePaths) -> None:
        paths = resolve_annotation_paths(workspace=workspace)
        assert paths.ensure_dirs() is paths

    def test_frozen(self, workspace: WorkspacePaths, tmp_path: Path) -> None:
        paths = resolve_annotation_paths(workspace=workspace)
        with pytest.raises((AttributeError, TypeError)):
            paths.tool_root = tmp_path  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AnnotationExportPaths
# ---------------------------------------------------------------------------


class TestAnnotationExportPaths:
    def test_resolve_export_paths_structure(self, workspace: WorkspacePaths) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        expected_dir = workspace.tool_root("annotation") / "exports" / "run1"
        assert paths.export_dir == expected_dir

    def test_csv_paths_under_export_dir(self, workspace: WorkspacePaths) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        assert paths.retrieval_annotation_csv == paths.export_dir / "retrieval.csv"
        assert paths.grounding_annotation_csv == paths.export_dir / "grounding.csv"
        assert paths.generation_annotation_csv == paths.export_dir / "generation.csv"

    def test_ensure_dirs_creates_export_dir(self, workspace: WorkspacePaths) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        assert not paths.export_dir.exists()
        paths.ensure_dirs()
        assert paths.export_dir.exists()

    def test_ensure_dirs_returns_self(self, workspace: WorkspacePaths) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        result = paths.ensure_dirs()
        assert result is paths

    def test_frozen(self, workspace: WorkspacePaths, tmp_path: Path) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        with pytest.raises((AttributeError, TypeError)):
            paths.export_dir = tmp_path  # type: ignore[misc]
