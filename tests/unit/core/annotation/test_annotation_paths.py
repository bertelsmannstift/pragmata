"""Unit tests for annotation export/import path bundles."""

from pathlib import Path

import pytest

from pragmata.core.paths.annotation_paths import (
    AnnotationExportPaths,
    AnnotationImportPaths,
    resolve_export_paths,
    resolve_import_paths,
)
from pragmata.core.paths.paths import WorkspacePaths


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
    return WorkspacePaths.from_base_dir(tmp_path)


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
        assert paths.retrieval_csv == paths.export_dir / "retrieval.csv"
        assert paths.grounding_csv == paths.export_dir / "grounding.csv"
        assert paths.generation_csv == paths.export_dir / "generation.csv"

    def test_ensure_dirs_creates_export_dir(self, workspace: WorkspacePaths) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        assert not paths.export_dir.exists()
        paths.ensure_dirs()
        assert paths.export_dir.exists()

    def test_ensure_dirs_returns_self(self, workspace: WorkspacePaths) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        result = paths.ensure_dirs()
        assert result is paths

    def test_from_dir_without_workspace(self, tmp_path: Path) -> None:
        d = tmp_path / "some" / "arbitrary" / "dir"
        paths = AnnotationExportPaths.from_dir(d)
        assert paths.export_dir == d
        assert paths.retrieval_csv == d / "retrieval.csv"
        assert paths.grounding_csv == d / "grounding.csv"
        assert paths.generation_csv == d / "generation.csv"

    def test_frozen(self, workspace: WorkspacePaths, tmp_path: Path) -> None:
        paths = resolve_export_paths(workspace=workspace, export_id="run1")
        with pytest.raises((AttributeError, TypeError)):
            paths.export_dir = tmp_path  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AnnotationImportPaths
# ---------------------------------------------------------------------------


class TestAnnotationImportPaths:
    def test_resolve_import_paths_structure(self, workspace: WorkspacePaths) -> None:
        paths = resolve_import_paths(workspace=workspace, import_id="imp1")
        expected_dir = workspace.tool_root("annotation") / "imports" / "imp1"
        assert paths.import_dir == expected_dir

    def test_result_json_path(self, workspace: WorkspacePaths) -> None:
        paths = resolve_import_paths(workspace=workspace, import_id="imp1")
        assert paths.result_json == paths.import_dir / "import_result.json"

    def test_ensure_dirs_creates_import_dir(self, workspace: WorkspacePaths) -> None:
        paths = resolve_import_paths(workspace=workspace, import_id="imp1")
        assert not paths.import_dir.exists()
        paths.ensure_dirs()
        assert paths.import_dir.exists()

    def test_ensure_dirs_returns_self(self, workspace: WorkspacePaths) -> None:
        paths = resolve_import_paths(workspace=workspace, import_id="imp1")
        result = paths.ensure_dirs()
        assert result is paths

    def test_from_dir_without_workspace(self, tmp_path: Path) -> None:
        d = tmp_path / "imports" / "imp1"
        paths = AnnotationImportPaths.from_dir(d)
        assert paths.import_dir == d
        assert paths.result_json == d / "import_result.json"

    def test_frozen(self, workspace: WorkspacePaths, tmp_path: Path) -> None:
        paths = resolve_import_paths(workspace=workspace, import_id="imp1")
        with pytest.raises((AttributeError, TypeError)):
            paths.import_dir = tmp_path  # type: ignore[misc]
