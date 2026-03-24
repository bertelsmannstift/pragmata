"""Path bundles for annotation export and import runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pragmata.core.paths.paths import WorkspacePaths


@dataclass(frozen=True, slots=True)
class AnnotationExportPaths:
    """Path bundle for an annotation export run.

    Attributes:
        export_dir: Root directory for the export.
        retrieval_csv: Output path for retrieval task annotations.
        grounding_csv: Output path for grounding task annotations.
        generation_csv: Output path for generation task annotations.
    """

    export_dir: Path
    retrieval_csv: Path
    grounding_csv: Path
    generation_csv: Path

    @classmethod
    def from_dir(cls, path: Path) -> Self:
        """Build from an arbitrary directory without a WorkspacePaths."""
        return cls(
            export_dir=path,
            retrieval_csv=path / "retrieval.csv",
            grounding_csv=path / "grounding.csv",
            generation_csv=path / "generation.csv",
        )

    def ensure_dirs(self) -> Self:
        """Create the export directory scaffold."""
        self.export_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class AnnotationImportPaths:
    """Path bundle for an annotation import run.

    Attributes:
        import_dir: Root directory for the import.
        result_json: Output path for the import result JSON.
    """

    import_dir: Path
    result_json: Path

    @classmethod
    def from_dir(cls, path: Path) -> Self:
        """Build from an arbitrary directory without a WorkspacePaths."""
        return cls(import_dir=path, result_json=path / "import_result.json")

    def ensure_dirs(self) -> Self:
        """Create the import directory scaffold."""
        self.import_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_export_paths(*, workspace: WorkspacePaths, export_id: str) -> AnnotationExportPaths:
    """Build the path bundle for an annotation export run.

    Args:
        workspace: Workspace path bundle.
        export_id: Unique export identifier.

    Returns:
        Path bundle for the export run.
    """
    return AnnotationExportPaths.from_dir(workspace.tool_root("annotation") / "exports" / export_id)


def resolve_import_paths(*, workspace: WorkspacePaths, import_id: str) -> AnnotationImportPaths:
    """Build the path bundle for an annotation import run.

    Args:
        workspace: Workspace path bundle.
        import_id: Unique import identifier.

    Returns:
        Path bundle for the import run.
    """
    return AnnotationImportPaths.from_dir(workspace.tool_root("annotation") / "imports" / import_id)
