"""Path bundles for eval workflows."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from pragmata.core.paths.annotation_paths import AnnotationExportPaths, resolve_export_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_export import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import EvalPredictMeta, EvalTrainMeta, ScoreInputSource


@dataclass(frozen=True, slots=True)
class EvalTrainPaths:
    """Resolved labeled input path and source provenance for eval train.

    Attributes:
        tool_root: Root directory for eval artifacts. Passed to tlmtc as
            ``work_dir`` for training.
        training_input_csv: Concrete labeled CSV path consumed by eval train.
        annotation_export_id: Resolved annotation export ID when the input comes
            from the annotation tool. ``None`` for direct user-provided CSV input.
    """

    tool_root: Path
    training_input_csv: Path
    annotation_export_id: str | None = None

    def ensure_dirs(
        self,
    ) -> Self:
        """Create the eval tool directory scaffold."""
        self.tool_root.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class EvalPredictPaths:
    """Resolved input paths for eval prediction.

    Attributes:
        tool_root: Root directory for eval artifacts. Passed to tlmtc as
            ``work_dir`` for prediction.
        prediction_input_csv: Concrete unlabeled CSV path consumed by eval
            prediction.
    """

    tool_root: Path
    prediction_input_csv: Path

    def ensure_dirs(
        self,
    ) -> Self:
        """Create the eval tool root."""
        self.tool_root.mkdir(parents=True, exist_ok=True)
        return self


def _task_annotation_csv(export_paths: AnnotationExportPaths, task: Task) -> Path:
    """Select the per-task annotation CSV from a resolved annotation export."""
    match task:
        case Task.RETRIEVAL:
            return export_paths.retrieval_annotation_csv
        case Task.GROUNDING:
            return export_paths.grounding_annotation_csv
        case Task.GENERATION:
            return export_paths.generation_annotation_csv
        case _:
            raise ValueError(f"Unsupported eval task: {task!r}")


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

        task_csv = _task_annotation_csv(export_paths, task)

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


def _require_at_most_one_selector(**selectors: object | None) -> None:
    """Reject ambiguous input selection: at most one selector may be set.

    Eval input selectors (a direct path, an export id, a prediction id) are
    mutually exclusive - there is no precedence between them. Passing more than
    one raises; passing none is allowed (callers fall back to the latest export).
    """
    given = [name for name, value in selectors.items() if value is not None]
    if len(given) > 1:
        raise ValueError(
            f"At most one input selector may be given, got {', '.join(given)}. "
            "Pass a single selector, or none to use the latest annotation export."
        )


def resolve_eval_train_paths(
    *,
    workspace: WorkspacePaths,
    task: Task,
    labeled_data_path: Path | None = None,
    export_id: str | None = None,
) -> EvalTrainPaths:
    """Resolve the labeled input CSV consumed by eval train.

    At most one input selector may be given: a direct ``labeled_data_path`` or an
    ``export_id``. If neither is provided, the latest valid annotation export
    containing the requested task CSV is selected.

    Args:
        workspace: Workspace path bundle.
        task: Annotation task that determines which exported task CSV is required.
        labeled_data_path: Optional direct labeled CSV path for standalone eval use.
        export_id: Optional annotation export identifier.

    Returns:
        Resolved train-input paths and annotation export provenance.

    Raises:
        ValueError: If both ``labeled_data_path`` and ``export_id`` are given.
        FileNotFoundError: If the selected input CSV or annotation export cannot
            be found.
    """
    _require_at_most_one_selector(labeled_data_path=labeled_data_path, export_id=export_id)

    if labeled_data_path is not None:
        training_input_csv = labeled_data_path.expanduser().resolve()
        if not training_input_csv.is_file():
            raise FileNotFoundError(f"Labeled eval training input CSV does not exist: {training_input_csv}")

        return EvalTrainPaths(
            tool_root=workspace.tool_root("eval"),
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

    training_input_csv = _task_annotation_csv(export_paths, task)

    if not training_input_csv.is_file():
        raise FileNotFoundError(
            f"Annotation export {resolved_export_id!r} does not contain the requested {task.value} CSV. "
            f"Expected {training_input_csv}."
        )

    return EvalTrainPaths(
        tool_root=workspace.tool_root("eval"),
        training_input_csv=training_input_csv,
        annotation_export_id=resolved_export_id,
    )


def resolve_eval_train_meta_path(
    *,
    workspace: WorkspacePaths,
    run_id: str,
) -> Path:
    """Build the Pragmata-owned metadata path for a completed eval train run.

    Args:
        workspace: Workspace path bundle.
        run_id: tlmtc evaluator training run identifier.

    Returns:
        Path to the Pragmata train metadata sidecar under the tlmtc train-run
        directory.
    """
    return workspace.tool_root("eval") / "train_outputs" / run_id / "pragmata_train.meta.json"


def resolve_eval_predict_meta_path(
    *,
    workspace: WorkspacePaths,
    run_id: str,
) -> Path:
    """Build the Pragmata-owned metadata path for a completed eval prediction run.

    Args:
        workspace: Workspace path bundle.
        run_id: tlmtc prediction run identifier (the evaluator run id, which
            tlmtc also uses to name the prediction-output directory).

    Returns:
        Path to the Pragmata predict metadata sidecar under the tlmtc
        prediction-run directory.
    """
    return workspace.tool_root("eval") / "prediction_outputs" / run_id / "pragmata_predict.meta.json"


def resolve_eval_predict_paths(
    *,
    workspace: WorkspacePaths,
    unlabeled_data_path: Path,
) -> EvalPredictPaths:
    """Resolve the explicit unlabeled input CSV consumed by eval predict.

    Args:
        workspace: Workspace path bundle.
        unlabeled_data_path: Direct unlabeled CSV path.

    Returns:
        Resolved prediction-input paths.

    Raises:
        FileNotFoundError: If the input path is not an existing file.
    """
    prediction_input_csv = unlabeled_data_path.expanduser().resolve()
    if not prediction_input_csv.is_file():
        raise FileNotFoundError(f"Unlabeled eval prediction input CSV does not exist: {prediction_input_csv}")

    return EvalPredictPaths(
        tool_root=workspace.tool_root("eval"),
        prediction_input_csv=prediction_input_csv,
    )


def find_latest_eval_train_run_id(
    *,
    workspace: WorkspacePaths,
    task: Task,
) -> str:
    """Find the latest completed evaluator training run for a task.

    Args:
        workspace: Workspace path bundle.
        task: Pragmata task the evaluator must have been trained for.

    Returns:
        Run ID selected by creation time with run ID as a deterministic
        tie-breaker.

    Raises:
        FileNotFoundError: If no evaluator metadata matches the requested task.
        pydantic.ValidationError: If persisted evaluator metadata is invalid.
    """
    train_outputs_dir = workspace.tool_root("eval") / "train_outputs"
    compatible_runs: list[tuple[datetime, str]] = []

    for meta_path in train_outputs_dir.glob("*/pragmata_train.meta.json"):
        meta = EvalTrainMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.task == task:
            compatible_runs.append((meta.created_at, meta.run_id))

    if not compatible_runs:
        raise FileNotFoundError(
            f"No completed eval training runs found for task={task.value!r}. "
            f"Expected matching pragmata_train.meta.json under {train_outputs_dir}."
        )

    return max(compatible_runs, key=lambda item: (item[0], item[1]))[1]


def resolve_eval_train_run_id(
    *,
    workspace: WorkspacePaths,
    task: Task,
    evaluator_run_id: str | None = None,
) -> str:
    """Resolve a concrete evaluator training run ID compatible with a task.

    Args:
        workspace: Workspace path bundle.
        task: Pragmata task the evaluator must have been trained for.
        evaluator_run_id: Optional explicit evaluator run ID. When omitted, the
            latest task-compatible evaluator is selected.

    Returns:
        Concrete task-compatible evaluator run ID.

    Raises:
        FileNotFoundError: If explicit evaluator metadata is missing or no
            compatible evaluator exists.
        ValueError: If explicit metadata identifies a different task.
        pydantic.ValidationError: If persisted evaluator metadata is invalid.
    """
    if evaluator_run_id is None:
        return find_latest_eval_train_run_id(workspace=workspace, task=task)

    meta_path = resolve_eval_train_meta_path(
        workspace=workspace,
        run_id=evaluator_run_id,
    )
    if not meta_path.is_file():
        raise FileNotFoundError(f"Eval train metadata does not exist: {meta_path}")

    meta = EvalTrainMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    if meta.task != task:
        raise ValueError(
            f"Evaluator {evaluator_run_id!r} was trained for task={meta.task.value!r}, "
            f"not requested task={task.value!r}."
        )

    return evaluator_run_id


@dataclass(frozen=True, slots=True)
class EvalScorePaths:
    """Path bundle for an eval score run.

    Attributes:
        tool_root: Root directory for the eval tool.
        score_dir: Root directory for a specific eval score run.
        retrieval_scores_json: Output path for retrieval score metrics.
        grounding_scores_json: Output path for grounding score metrics.
        generation_scores_json: Output path for generation score metrics.
    """

    tool_root: Path
    score_dir: Path
    retrieval_scores_json: Path
    grounding_scores_json: Path
    generation_scores_json: Path

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
    )


@dataclass(frozen=True, slots=True)
class EvalScoreInput:
    """Resolved score input CSV and its provenance.

    Attributes:
        input_csv: Concrete labeled CSV path to read and score.
        source: Provenance of the input (selector kind, its value, and the
            resolved path), recorded on the score report.
    """

    input_csv: Path
    source: ScoreInputSource


def provenance_path(*, input_csv: Path, base_dir: Path) -> str:
    """Record a scored input relative to the workspace, or absolute if outside it."""
    resolved_base = base_dir.expanduser().resolve()
    try:
        return str(input_csv.relative_to(resolved_base))
    except ValueError:
        return str(input_csv)


def resolve_eval_score_input(
    *,
    workspace: WorkspacePaths,
    task: Task,
    path: Path | None = None,
    export_id: str | None = None,
    prediction_id: str | None = None,
) -> EvalScoreInput:
    """Resolve the labeled CSV consumed by eval score and its provenance.

    At most one input selector may be given: a direct ``path``, an ``export_id``,
    or a ``prediction_id``. With no selector, the latest valid annotation export
    for the task is used, mirroring ``resolve_eval_train_paths``.

    Direct paths and annotation exports are already Pragmata-shaped and score
    without further preparation. A ``prediction_id`` selects a completed
    prediction run's ``predictions.csv`` (tlmtc-shaped; the importer restores the
    task text columns); the run's persisted ``pragmata_predict.meta.json`` is
    checked to confirm it was produced for the requested task. With no selector,
    the latest annotation export is used.

    Args:
        workspace: Workspace path bundle.
        task: Annotation task that determines which exported task CSV is required.
        path: Direct labeled CSV path.
        export_id: Annotation export identifier.
        prediction_id: Pragmata prediction run identifier (the evaluator run id
            that produced the run; see ``EvalPredictMeta``).

    Returns:
        The resolved input CSV and its ``ScoreInputSource`` provenance. The
        provenance's ``kind`` and ``ref`` let the caller (and the report) record
        how the input was selected without re-running selection.

    Raises:
        ValueError: If more than one of ``path`` / ``export_id`` / ``prediction_id``
            is given, or if the prediction run was produced for a different task.
        FileNotFoundError: If the selected input CSV, annotation export, or
            prediction run is missing.
    """
    _require_at_most_one_selector(path=path, export_id=export_id, prediction_id=prediction_id)

    if path is not None:
        input_csv = path.expanduser().resolve()
        if not input_csv.is_file():
            raise FileNotFoundError(f"Scoring input CSV does not exist: {input_csv}")
        return EvalScoreInput(
            input_csv=input_csv,
            source=ScoreInputSource(
                kind="direct_path",
                ref=str(path),
                resolved_path=provenance_path(input_csv=input_csv, base_dir=workspace.base_dir),
            ),
        )

    if prediction_id is not None:
        meta_path = resolve_eval_predict_meta_path(workspace=workspace, run_id=prediction_id)
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Prediction run {prediction_id!r} metadata does not exist. "
                f"Expected pragmata_predict.meta.json at {meta_path}."
            )
        meta = EvalPredictMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.task != task:
            raise ValueError(
                f"Prediction run {prediction_id!r} was produced for task={meta.task.value!r}, "
                f"not requested task={task.value!r}."
            )
        predictions_csv = meta_path.parent / "predictions.csv"
        if not predictions_csv.is_file():
            raise FileNotFoundError(
                f"Prediction run {prediction_id!r} does not contain predictions.csv. Expected {predictions_csv}."
            )
        return EvalScoreInput(
            input_csv=predictions_csv,
            source=ScoreInputSource(
                kind="model_prediction",
                ref=prediction_id,
                resolved_path=provenance_path(input_csv=predictions_csv, base_dir=workspace.base_dir),
            ),
        )

    if export_id is not None:
        resolved_export_id = export_id
    else:
        resolved_export_id = find_latest_annotation_export_id(workspace=workspace, task=task)

    export_paths = resolve_export_paths(workspace=workspace, export_id=resolved_export_id)
    if not export_paths.export_meta_json.is_file():
        raise FileNotFoundError(
            "Annotation export metadata does not exist. "
            f"Expected annotation_export.meta.json at {export_paths.export_meta_json}."
        )

    input_csv = _task_annotation_csv(export_paths, task)

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Annotation export {resolved_export_id!r} does not contain the requested {task.value} CSV. "
            f"Expected {input_csv}."
        )
    return EvalScoreInput(
        input_csv=input_csv,
        source=ScoreInputSource(
            kind="annotation_export",
            ref=resolved_export_id,
            resolved_path=provenance_path(input_csv=input_csv, base_dir=workspace.base_dir),
        ),
    )
