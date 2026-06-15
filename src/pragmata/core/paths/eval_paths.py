"""Path bundles for eval workflows."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from pragmata.core.paths.annotation_paths import resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_export import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task


@dataclass(frozen=True, slots=True)
class EvalTrainPaths:
    """Resolved labeled input path and source provenance for eval train.

    Attributes:
        training_input_csv: Concrete labeled CSV path consumed by eval train.
        annotation_export_id: Resolved annotation export ID when the input comes
            from the annotation tool. ``None`` for direct user-provided CSV input.
    """

    training_input_csv: Path
    annotation_export_id: str | None = None


def find_latest_annotation_export_id(
    *,
    workspace: WorkspacePaths,
    task: Task,
) -> str:
    """Find the most recent annotation export containing the requested task CSV."""
    exports_dir = workspace.tool_root("annotation") / "exports"
    if not exports_dir.is_dir():
        raise FileNotFoundError(
            f"No annotation exports found for eval training. Expected annotation exports directory at {exports_dir}."
        )
    completed_exports: list[tuple[datetime, str]] = []

    for meta_path in exports_dir.glob("*/annotation_export.meta.json"):
        meta = AnnotationExportMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

        if task not in meta.tasks:
            continue

        export_paths = resolve_export_paths(
            workspace=workspace,
            export_id=meta.export_id,
        )

        match task:
            case Task.RETRIEVAL:
                task_csv = export_paths.retrieval_annotation_csv
            case Task.GROUNDING:
                task_csv = export_paths.grounding_annotation_csv
            case Task.GENERATION:
                task_csv = export_paths.generation_annotation_csv

        if not task_csv.is_file():
            continue

        completed_exports.append((meta.created_at, meta.export_id))

    if not completed_exports:
        raise FileNotFoundError(
            "No annotation exports found for eval training. "
            f"Expected at least one annotation_export.meta.json with task={task.value!r} "
            f"and a matching task CSV under {exports_dir}."
        )

    _, latest_export_id = max(
        completed_exports,
        key=lambda item: (item[0], item[1]),
    )
    return latest_export_id


def resolve_eval_train_paths(
    *,
    workspace: WorkspacePaths,
    task: Task,
    labeled_data_path: Path | None = None,
    export_id: str | None = None,
) -> EvalTrainPaths:
    """Resolve the labeled input CSV consumed by eval train.

    Direct labeled CSV input takes precedence over annotation export selection.
    If no direct input path is provided, ``export_id`` selects an annotation
    export. If neither is provided, the latest valid annotation export containing
    the requested task CSV is selected.

    Args:
        workspace: Workspace path bundle.
        task: Annotation task that determines which exported task CSV is required.
        labeled_data_path: Optional direct labeled CSV path for standalone eval use.
        export_id: Optional annotation export identifier.

    Returns:
        Resolved train-input paths and annotation export provenance.

    Raises:
        FileNotFoundError: If the selected input CSV or annotation export cannot
            be found.
    """
    if labeled_data_path is not None:
        training_input_csv = labeled_data_path.expanduser().resolve()
        if not training_input_csv.is_file():
            raise FileNotFoundError(f"Labeled eval training input CSV does not exist: {training_input_csv}")

        return EvalTrainPaths(
            training_input_csv=training_input_csv,
            annotation_export_id=None,
        )

    resolved_export_id = export_id or find_latest_annotation_export_id(
        workspace=workspace,
        task=task,
    )
    export_paths = resolve_export_paths(
        workspace=workspace,
        export_id=resolved_export_id,
    )

    if not export_paths.export_meta_json.is_file():
        raise FileNotFoundError(
            "Annotation export metadata does not exist. "
            f"Expected annotation_export.meta.json at {export_paths.export_meta_json}."
        )

    match task:
        case Task.RETRIEVAL:
            training_input_csv = export_paths.retrieval_annotation_csv
        case Task.GROUNDING:
            training_input_csv = export_paths.grounding_annotation_csv
        case Task.GENERATION:
            training_input_csv = export_paths.generation_annotation_csv

    if not training_input_csv.is_file():
        raise FileNotFoundError(
            f"Annotation export {resolved_export_id!r} does not contain the requested {task.value} CSV. "
            f"Expected {training_input_csv}."
        )

    return EvalTrainPaths(
        training_input_csv=training_input_csv,
        annotation_export_id=resolved_export_id,
    )


@dataclass(frozen=True, slots=True)
class EvalScorePaths:
    """Path bundle for an eval score run.

    Attributes:
        tool_root: Root directory for the eval tool.
        score_dir: Root directory for a specific eval score run.
        retrieval_scores_json: Output path for retrieval score metrics.
        grounding_scores_json: Output path for grounding score metrics.
        generation_scores_json: Output path for generation score metrics.
        scores_meta_json: Output path for score run metadata.
    """

    tool_root: Path
    score_dir: Path
    retrieval_scores_json: Path
    grounding_scores_json: Path
    generation_scores_json: Path
    scores_meta_json: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create the eval score directory scaffold."""
        self.score_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_eval_score_paths(
    *,
    workspace: WorkspacePaths,
    score_id: str,
) -> EvalScorePaths:
    """Build the path bundle for an eval score run.

    Args:
        workspace: Workspace path bundle.
        score_id: Unique score run identifier.

    Returns:
        Path bundle for the eval score run.
    """
    tool_root = workspace.tool_root("eval")
    score_dir = tool_root / "scores" / score_id

    return EvalScorePaths(
        tool_root=tool_root,
        score_dir=score_dir,
        retrieval_scores_json=score_dir / "retrieval_scores.json",
        grounding_scores_json=score_dir / "grounding_scores.json",
        generation_scores_json=score_dir / "generation_scores.json",
        scores_meta_json=score_dir / "scores.meta.json",
    )
