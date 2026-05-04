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
class AnnotationImportPaths:
    """Path bundle for an annotation import scope (one per ``dataset_id``).

    Imports accumulate state across calls into the same logical scope, so the
    bundle is keyed by ``dataset_id`` (persistent topology scope) rather than
    a per-call identifier. The partition manifest sidecar tracks calibration
    vs production assignment for every record imported into this scope.

    Attributes:
        import_dir: Per-``dataset_id`` import directory.
        partition_manifest: Partition manifest sidecar path.
    """

    import_dir: Path
    partition_manifest: Path

    def ensure_dirs(self) -> Self:
        """Create the import directory scaffold."""
        self.import_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_import_paths(*, workspace: WorkspacePaths, dataset_id: str) -> AnnotationImportPaths:
    """Build the path bundle for an annotation import scope.

    Args:
        workspace: Workspace path bundle.
        dataset_id: Topology scope. Empty string maps to the ``"default"``
            directory so the layout is uniform across scopes.

    Returns:
        Path bundle for the import scope.
    """
    scope = dataset_id or "default"
    import_dir = workspace.tool_root("annotation") / "imports" / scope
    return AnnotationImportPaths(
        import_dir=import_dir,
        partition_manifest=import_dir / "partition.meta.json",
    )


@dataclass(frozen=True, slots=True)
class AnnotationExportPaths:
    """Path bundle for an annotation export run.

    Attributes:
        export_dir: Root directory for the export.
        tool_root: Root directory for the tool data.
        retrieval_annotation_csv: Output path for retrieval task annotations.
        grounding_annotation_csv: Output path for grounding task annotations.
        generation_annotation_csv: Output path for generation task annotations.
        export_meta_json: Output path for the export provenance sidecar.
    """

    export_dir: Path
    tool_root: Path
    retrieval_annotation_csv: Path
    grounding_annotation_csv: Path
    generation_annotation_csv: Path
    export_meta_json: Path

    def ensure_dirs(self) -> Self:
        """Create the export directory scaffold."""
        self.export_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class IaaPaths:
    """Path bundle for an IAA analysis run scoped to an export.

    Attributes:
        iaa_dir: Directory for IAA outputs.
        report: Path to the JSON report file.
    """

    iaa_dir: Path
    report: Path

    def ensure_dirs(self) -> Self:
        """Create the IAA output directory."""
        self.iaa_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_iaa_paths(*, export_paths: AnnotationExportPaths) -> IaaPaths:
    """Build the path bundle for an IAA run within an export.

    Args:
        export_paths: Export path bundle that this IAA run analyses.

    Returns:
        Path bundle for IAA outputs.
    """
    iaa_dir = export_paths.export_dir / "iaa"
    return IaaPaths(iaa_dir=iaa_dir, report=iaa_dir / "report.json")


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
        export_meta_json=export_dir / "annotation_export.meta.json",
    )
