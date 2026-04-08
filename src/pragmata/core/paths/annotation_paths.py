"""Path bundles for annotation tool and export runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pragmata.core.paths.paths import WorkspacePaths


@dataclass(frozen=True, slots=True)
class AnnotationPaths:
    """Path bundle for the annotation tool root.

    Attributes:
        tool_root: Root directory for annotation artifacts.
    """

    tool_root: Path

    def ensure_dirs(self) -> Self:
        """Create the annotation tool root directory."""
        self.tool_root.mkdir(parents=True, exist_ok=True)
        return self


def resolve_annotation_paths(*, workspace: WorkspacePaths) -> AnnotationPaths:
    """Build the path bundle for the annotation tool.

    Args:
        workspace: Workspace path bundle.

    Returns:
        Path bundle for the annotation tool root.
    """
    return AnnotationPaths(tool_root=workspace.tool_root("annotation"))


@dataclass(frozen=True, slots=True)
class AnnotationExportPaths:
    """Path bundle for an annotation export run.

    Attributes:
        export_dir: Root directory for the export.
        tool_root: Root directory for the tool data.
        retrieval_annotation_csv: Output path for retrieval task annotations.
        grounding_annotation_csv: Output path for grounding task annotations.
        generation_annotation_csv: Output path for generation task annotations.
    """

    export_dir: Path
    tool_root: Path
    retrieval_annotation_csv: Path
    grounding_annotation_csv: Path
    generation_annotation_csv: Path

    def ensure_dirs(self) -> Self:
        """Create the export directory scaffold."""
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.tool_root.mkdir(parents=True, exist_ok=True)
        return self


def resolve_export_paths(*, workspace: WorkspacePaths, export_id: str) -> AnnotationExportPaths:
    """Build the path bundle for an annotation export run.

    Args:
        workspace: Workspace path bundle.
        export_id: Unique export identifier.

    Returns:
        Path bundle for the export run.
    """
    export_dir = workspace.tool_root("annotation") / "exports" / export_id
    return AnnotationExportPaths(
        export_dir=export_dir,
        tool_root=workspace.tool_root("annotation"),
        retrieval_annotation_csv=export_dir / "retrieval.csv",
        grounding_annotation_csv=export_dir / "grounding.csv",
        generation_annotation_csv=export_dir / "generation.csv",
    )
