"""Path bundles for annotation tool and export runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pragmata.core.paths.paths import WorkspacePaths


@dataclass(frozen=True, slots=True)
class AnnotationExportPaths:
    """Path bundle for an annotation export run.

    Attributes:
        export_dir: Root directory for the export.
        retrieval_annotation_csv: Output path for retrieval task annotations.
        grounding_annotation_csv: Output path for grounding task annotations.
        generation_annotation_csv: Output path for generation task annotations.
    """

    export_dir: Path
    retrieval_annotation_csv: Path
    grounding_annotation_csv: Path
    generation_annotation_csv: Path

    def ensure_dirs(self) -> Self:
        """Create the export directory scaffold."""
        self.export_dir.mkdir(parents=True, exist_ok=True)
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
        retrieval_annotation_csv=export_dir / "retrieval.csv",
        grounding_annotation_csv=export_dir / "grounding.csv",
        generation_annotation_csv=export_dir / "generation.csv",
    )
