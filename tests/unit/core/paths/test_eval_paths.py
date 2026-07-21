"""Tests for eval workflow path bundles."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pragmata.core.paths.eval_paths import (
    EvalPredictPaths,
    EvalScorePaths,
    EvalTrainPaths,
    find_latest_annotation_export_id,
    find_latest_eval_train_run_id,
    provenance_path,
    resolve_eval_predict_paths,
    resolve_eval_score_input,
    resolve_eval_score_paths,
    resolve_eval_train_meta_path,
    resolve_eval_train_paths,
    resolve_eval_train_run_id,
)
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_export import AnnotationExportMeta
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import EvalTrainMeta


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
    """Workspace path bundle rooted in a temporary directory."""
    return WorkspacePaths.from_base_dir(tmp_path)


def _write_annotation_export(
    *,
    workspace: WorkspacePaths,
    export_id: str,
    created_at: datetime,
    tasks: list[Task],
    csv_tasks: list[Task] | None = None,
) -> Path:
    """Write a minimal annotation export directory for path-resolution tests."""
    export_dir = workspace.base_dir / "annotation" / "exports" / export_id
    export_dir.mkdir(parents=True)

    for task in csv_tasks if csv_tasks is not None else tasks:
        match task:
            case Task.RETRIEVAL:
                csv_path = export_dir / "retrieval.csv"
            case Task.GROUNDING:
                csv_path = export_dir / "grounding.csv"
            case Task.GENERATION:
                csv_path = export_dir / "generation.csv"

        csv_path.write_text("example_id\nexample-1\n", encoding="utf-8")

    meta = AnnotationExportMeta(
        export_id=export_id,
        created_at=created_at,
        dataset_id=None,
        tasks=tasks,
        include_discarded=False,
        row_counts={task: 1 for task in tasks},
        n_annotators={task: 1 for task in tasks},
        calibration_enabled={},
        constraint_summary={},
    )
    (export_dir / "annotation_export.meta.json").write_text(
        meta.model_dump_json(),
        encoding="utf-8",
    )

    return export_dir


def _write_eval_train_meta(
    *,
    workspace: WorkspacePaths,
    run_id: str,
    created_at: datetime,
    task: Task,
) -> None:
    """Write a minimal eval train metadata sidecar for path tests."""
    meta_path = resolve_eval_train_meta_path(
        workspace=workspace,
        run_id=run_id,
    )
    meta_path.parent.mkdir(parents=True)
    meta = EvalTrainMeta(
        run_id=run_id,
        created_at=created_at,
        task=task,
    )
    meta_path.write_text(meta.model_dump_json(), encoding="utf-8")


def test_resolve_eval_train_paths_uses_direct_labeled_data_path(
    workspace: WorkspacePaths,
) -> None:
    """Direct labeled CSV input is used as-is and does not set an export ID."""
    input_csv = workspace.base_dir / "standalone" / "labeled.csv"
    input_csv.parent.mkdir()
    input_csv.write_text("example_id\nexample-1\n", encoding="utf-8")

    paths = resolve_eval_train_paths(
        workspace=workspace,
        task=Task.RETRIEVAL,
        labeled_data_path=input_csv,
    )

    assert paths == EvalTrainPaths(
        tool_root=workspace.base_dir / "eval",
        training_input_csv=input_csv.resolve(),
        annotation_export_id=None,
    )


def test_eval_train_paths_ensure_dirs_creates_eval_tool_root(
    workspace: WorkspacePaths,
) -> None:
    """EvalTrainPaths.ensure_dirs creates the eval tool root and returns self."""
    input_csv = workspace.base_dir / "standalone" / "labeled.csv"
    input_csv.parent.mkdir()
    input_csv.write_text("example_id\nexample-1\n", encoding="utf-8")
    paths = resolve_eval_train_paths(
        workspace=workspace,
        task=Task.RETRIEVAL,
        labeled_data_path=input_csv,
    )

    assert not paths.tool_root.exists()

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.tool_root.is_dir()


def test_resolve_eval_train_paths_rejects_missing_direct_labeled_data_path(
    workspace: WorkspacePaths,
) -> None:
    """Direct labeled CSV input must point to an existing file."""
    missing_csv = workspace.base_dir / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Labeled eval training input CSV does not exist"):
        resolve_eval_train_paths(
            workspace=workspace,
            task=Task.RETRIEVAL,
            labeled_data_path=missing_csv,
        )


def test_resolve_eval_train_paths_uses_explicit_annotation_export_id(
    workspace: WorkspacePaths,
) -> None:
    """Explicit export IDs resolve to the requested task CSV."""
    _write_annotation_export(
        workspace=workspace,
        export_id="export-a",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )

    paths = resolve_eval_train_paths(
        workspace=workspace,
        task=Task.RETRIEVAL,
        export_id="export-a",
    )

    expected_csv = workspace.base_dir / "annotation" / "exports" / "export-a" / "retrieval.csv"
    assert paths == EvalTrainPaths(
        tool_root=workspace.base_dir / "eval",
        training_input_csv=expected_csv,
        annotation_export_id="export-a",
    )


def test_resolve_eval_train_paths_rejects_missing_annotation_export_metadata(
    workspace: WorkspacePaths,
) -> None:
    """Explicit annotation export selection requires export metadata."""
    export_dir = workspace.base_dir / "annotation" / "exports" / "broken-export"
    export_dir.mkdir(parents=True)
    (export_dir / "retrieval.csv").write_text("example_id\nexample-1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Annotation export metadata does not exist"):
        resolve_eval_train_paths(
            workspace=workspace,
            task=Task.RETRIEVAL,
            export_id="broken-export",
        )


def test_resolve_eval_train_paths_rejects_missing_requested_task_csv(
    workspace: WorkspacePaths,
) -> None:
    """Explicit annotation export selection requires the requested task CSV."""
    _write_annotation_export(
        workspace=workspace,
        export_id="missing-task-csv",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
        csv_tasks=[],
    )

    with pytest.raises(FileNotFoundError, match="does not contain the requested"):
        resolve_eval_train_paths(
            workspace=workspace,
            task=Task.RETRIEVAL,
            export_id="missing-task-csv",
        )


def test_find_latest_annotation_export_id_filters_by_task(
    workspace: WorkspacePaths,
) -> None:
    """Latest annotation export lookup ignores exports for other tasks."""
    _write_annotation_export(
        workspace=workspace,
        export_id="newer-wrong-task",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        tasks=[Task.GENERATION],
    )
    _write_annotation_export(
        workspace=workspace,
        export_id="older-right-task",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )

    latest_export_id = find_latest_annotation_export_id(
        workspace=workspace,
        task=Task.RETRIEVAL,
    )

    assert latest_export_id == "older-right-task"


def test_find_latest_annotation_export_id_ignores_exports_missing_task_csv(
    workspace: WorkspacePaths,
) -> None:
    """Latest annotation export lookup requires both metadata membership and task CSV presence."""
    _write_annotation_export(
        workspace=workspace,
        export_id="newer-incomplete",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
        csv_tasks=[],
    )
    _write_annotation_export(
        workspace=workspace,
        export_id="older-complete",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )

    latest_export_id = find_latest_annotation_export_id(
        workspace=workspace,
        task=Task.RETRIEVAL,
    )

    assert latest_export_id == "older-complete"


def test_find_latest_annotation_export_id_tie_breaks_by_export_id(
    workspace: WorkspacePaths,
) -> None:
    """Latest annotation export lookup uses export ID as deterministic timestamp tie-breaker."""
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    _write_annotation_export(
        workspace=workspace,
        export_id="export-a",
        created_at=created_at,
        tasks=[Task.RETRIEVAL],
    )
    _write_annotation_export(
        workspace=workspace,
        export_id="export-b",
        created_at=created_at,
        tasks=[Task.RETRIEVAL],
    )

    latest_export_id = find_latest_annotation_export_id(
        workspace=workspace,
        task=Task.RETRIEVAL,
    )

    assert latest_export_id == "export-b"


def test_find_latest_annotation_export_id_rejects_missing_exports_dir(
    workspace: WorkspacePaths,
) -> None:
    """Latest annotation export lookup fails clearly when no exports directory exists."""
    with pytest.raises(FileNotFoundError, match="Expected annotation exports directory"):
        find_latest_annotation_export_id(
            workspace=workspace,
            task=Task.RETRIEVAL,
        )


def test_find_latest_annotation_export_id_rejects_no_matching_exports(
    workspace: WorkspacePaths,
) -> None:
    """Latest annotation export lookup fails when no complete export exists for the task."""
    _write_annotation_export(
        workspace=workspace,
        export_id="generation-only",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.GENERATION],
    )

    with pytest.raises(FileNotFoundError, match="Expected at least one annotation_export.meta.json"):
        find_latest_annotation_export_id(
            workspace=workspace,
            task=Task.RETRIEVAL,
        )


def test_resolve_eval_train_paths_uses_latest_annotation_export_when_selector_is_omitted(
    workspace: WorkspacePaths,
) -> None:
    """Train path resolution falls back to the latest valid annotation export."""
    _write_annotation_export(
        workspace=workspace,
        export_id="older",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )
    _write_annotation_export(
        workspace=workspace,
        export_id="newer",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )

    paths = resolve_eval_train_paths(
        workspace=workspace,
        task=Task.RETRIEVAL,
    )

    expected_csv = workspace.base_dir / "annotation" / "exports" / "newer" / "retrieval.csv"
    assert paths == EvalTrainPaths(
        tool_root=workspace.base_dir / "eval",
        training_input_csv=expected_csv,
        annotation_export_id="newer",
    )


def test_resolve_eval_train_meta_path_returns_train_run_sidecar_path(
    workspace: WorkspacePaths,
) -> None:
    """Pragmata train metadata is stored beside the tlmtc train run artifacts."""
    path = resolve_eval_train_meta_path(
        workspace=workspace,
        run_id="train-run-26",
    )

    assert path == workspace.base_dir / "eval" / "train_outputs" / "train-run-26" / "pragmata_train.meta.json"


def test_resolve_eval_predict_paths_uses_direct_unlabeled_data_path(
    workspace: WorkspacePaths,
) -> None:
    """Direct unlabeled CSV input resolves to an eval prediction path bundle."""
    input_csv = workspace.base_dir / "standalone" / "unlabeled.csv"
    input_csv.parent.mkdir()
    input_csv.write_text("example_id\nexample-1\n", encoding="utf-8")

    paths = resolve_eval_predict_paths(
        workspace=workspace,
        unlabeled_data_path=input_csv,
    )

    assert paths == EvalPredictPaths(
        tool_root=workspace.base_dir / "eval",
        prediction_input_csv=input_csv.resolve(),
    )


def test_resolve_eval_predict_paths_rejects_missing_input(
    workspace: WorkspacePaths,
) -> None:
    """Prediction input must point to an existing file."""
    with pytest.raises(FileNotFoundError, match="Unlabeled eval prediction input CSV does not exist"):
        resolve_eval_predict_paths(
            workspace=workspace,
            unlabeled_data_path=workspace.base_dir / "missing.csv",
        )


def test_eval_predict_paths_ensure_dirs_creates_eval_tool_root(
    workspace: WorkspacePaths,
) -> None:
    """EvalPredictPaths.ensure_dirs creates the eval tool root and returns self."""
    paths = EvalPredictPaths(
        tool_root=workspace.base_dir / "eval",
        prediction_input_csv=workspace.base_dir / "unlabeled.csv",
    )

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.tool_root.is_dir()


def test_resolve_eval_train_run_id_uses_explicit_task_compatible_evaluator(
    workspace: WorkspacePaths,
) -> None:
    """Explicit evaluator selection accepts metadata for the requested task."""
    _write_eval_train_meta(
        workspace=workspace,
        run_id="retrieval-run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        task=Task.RETRIEVAL,
    )

    run_id = resolve_eval_train_run_id(
        workspace=workspace,
        task=Task.RETRIEVAL,
        evaluator_run_id="retrieval-run",
    )

    assert run_id == "retrieval-run"


def test_resolve_eval_train_run_id_rejects_missing_metadata(
    workspace: WorkspacePaths,
) -> None:
    """Explicit evaluator selection requires its Pragmata metadata sidecar."""
    with pytest.raises(FileNotFoundError, match="pragmata_train.meta.json"):
        resolve_eval_train_run_id(
            workspace=workspace,
            task=Task.RETRIEVAL,
            evaluator_run_id="missing-run",
        )


def test_resolve_eval_train_run_id_rejects_task_mismatch(
    workspace: WorkspacePaths,
) -> None:
    """An explicit evaluator cannot be selected for a different task."""
    _write_eval_train_meta(
        workspace=workspace,
        run_id="generation-run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        task=Task.GENERATION,
    )

    with pytest.raises(ValueError, match="generation.*retrieval"):
        resolve_eval_train_run_id(
            workspace=workspace,
            task=Task.RETRIEVAL,
            evaluator_run_id="generation-run",
        )


def test_find_latest_eval_train_run_id_selects_latest_matching_task(
    workspace: WorkspacePaths,
) -> None:
    """Latest evaluator selection filters by task and creation time."""
    _write_eval_train_meta(
        workspace=workspace,
        run_id="older-retrieval",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        task=Task.RETRIEVAL,
    )
    _write_eval_train_meta(
        workspace=workspace,
        run_id="newer-retrieval",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        task=Task.RETRIEVAL,
    )
    _write_eval_train_meta(
        workspace=workspace,
        run_id="newest-generation",
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
        task=Task.GENERATION,
    )

    run_id = find_latest_eval_train_run_id(
        workspace=workspace,
        task=Task.RETRIEVAL,
    )

    assert run_id == "newer-retrieval"


def test_find_latest_eval_train_run_id_tie_breaks_by_run_id(
    workspace: WorkspacePaths,
) -> None:
    """Latest evaluator selection uses run ID as deterministic timestamp tie-breaker."""
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    _write_eval_train_meta(
        workspace=workspace,
        run_id="run-a",
        created_at=created_at,
        task=Task.RETRIEVAL,
    )
    _write_eval_train_meta(
        workspace=workspace,
        run_id="run-b",
        created_at=created_at,
        task=Task.RETRIEVAL,
    )

    run_id = find_latest_eval_train_run_id(
        workspace=workspace,
        task=Task.RETRIEVAL,
    )

    assert run_id == "run-b"


def test_find_latest_eval_train_run_id_rejects_no_matching_evaluator(
    workspace: WorkspacePaths,
) -> None:
    """Latest evaluator selection fails when no metadata matches the task."""
    _write_eval_train_meta(
        workspace=workspace,
        run_id="generation-run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        task=Task.GENERATION,
    )

    with pytest.raises(FileNotFoundError, match="task='retrieval'"):
        find_latest_eval_train_run_id(
            workspace=workspace,
            task=Task.RETRIEVAL,
        )


def test_resolve_eval_score_paths_returns_expected_bundle(
    workspace: WorkspacePaths,
) -> None:
    """Resolver returns the expected score run and artifact paths."""
    score_id = "score-26"

    paths = resolve_eval_score_paths(
        workspace=workspace,
        score_id=score_id,
    )
    expected_tool_root = workspace.base_dir / "eval"
    expected_score_dir = workspace.base_dir / "eval" / "scores" / score_id

    assert paths == EvalScorePaths(
        tool_root=expected_tool_root,
        score_dir=expected_score_dir,
        retrieval_scores_json=expected_score_dir / "retrieval_scores.json",
        grounding_scores_json=expected_score_dir / "grounding_scores.json",
        generation_scores_json=expected_score_dir / "generation_scores.json",
    )


def test_ensure_dirs_creates_score_directory_and_returns_self(
    workspace: WorkspacePaths,
) -> None:
    """ensure_dirs creates the score directory scaffold and returns self."""
    paths = resolve_eval_score_paths(
        workspace=workspace,
        score_id="new-score",
    )

    assert not paths.score_dir.exists()
    assert not paths.tool_root.exists()

    returned = paths.ensure_dirs()

    assert returned is paths
    assert paths.tool_root.is_dir()
    assert paths.score_dir.is_dir()


def test_score_artifact_paths_are_score_id_specific(
    workspace: WorkspacePaths,
) -> None:
    """Score artifact paths depend on the score ID."""
    first = resolve_eval_score_paths(
        workspace=workspace,
        score_id="score-a",
    )
    second = resolve_eval_score_paths(
        workspace=workspace,
        score_id="score-b",
    )

    assert first.score_dir != second.score_dir
    assert first.retrieval_scores_json != second.retrieval_scores_json
    assert first.grounding_scores_json != second.grounding_scores_json
    assert first.generation_scores_json != second.generation_scores_json

    assert first.score_dir.parent == second.score_dir.parent == first.tool_root / "scores"


def test_resolve_eval_score_input_uses_direct_labeled_input_path(
    workspace: WorkspacePaths,
    tmp_path: Path,
) -> None:
    """A direct labeled input path is used verbatim (resolved)."""
    csv = tmp_path / "labeled.csv"
    csv.write_text("example_id\nexample-1\n", encoding="utf-8")

    resolved = resolve_eval_score_input(workspace=workspace, task=Task.RETRIEVAL, path=csv)

    assert resolved.input_csv == csv.resolve()
    assert resolved.source.kind == "direct_path"
    assert resolved.source.ref == str(csv)


def test_resolve_eval_score_input_rejects_missing_direct_path(
    workspace: WorkspacePaths,
    tmp_path: Path,
) -> None:
    """A direct labeled input path that does not exist fails fast."""
    with pytest.raises(FileNotFoundError, match="Scoring input CSV does not exist"):
        resolve_eval_score_input(workspace=workspace, task=Task.RETRIEVAL, path=tmp_path / "nope.csv")


def test_resolve_eval_score_input_uses_explicit_export_id(
    workspace: WorkspacePaths,
) -> None:
    """An explicit export id resolves to the requested task CSV."""
    _write_annotation_export(
        workspace=workspace,
        export_id="export-a",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.GROUNDING],
    )

    resolved = resolve_eval_score_input(workspace=workspace, task=Task.GROUNDING, export_id="export-a")

    assert resolved.input_csv == workspace.base_dir / "annotation" / "exports" / "export-a" / "grounding.csv"
    assert resolved.source.kind == "annotation_export"
    assert resolved.source.ref == "export-a"
    assert resolved.source.resolved_path == "annotation/exports/export-a/grounding.csv"


def test_resolve_eval_score_input_falls_back_to_latest_export(
    workspace: WorkspacePaths,
) -> None:
    """With no selector, the latest valid annotation export is used (interim fallback)."""
    _write_annotation_export(
        workspace=workspace,
        export_id="older",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )
    _write_annotation_export(
        workspace=workspace,
        export_id="newer",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        tasks=[Task.RETRIEVAL],
    )

    resolved = resolve_eval_score_input(workspace=workspace, task=Task.RETRIEVAL)

    assert resolved.input_csv == workspace.base_dir / "annotation" / "exports" / "newer" / "retrieval.csv"
    assert resolved.source.kind == "annotation_export"
    assert resolved.source.ref == "newer"


def test_resolve_eval_score_input_prediction_run_not_supported(
    workspace: WorkspacePaths,
) -> None:
    """A prediction run as the only selector is deferred until eval predict lands."""
    with pytest.raises(NotImplementedError, match="prediction run is not yet supported"):
        resolve_eval_score_input(workspace=workspace, task=Task.RETRIEVAL, prediction_id="run-1")


def test_resolve_eval_score_input_rejects_path_and_export_together(
    workspace: WorkspacePaths,
    tmp_path: Path,
) -> None:
    """A direct path and an export id together is an error - there is no precedence."""
    csv = tmp_path / "labeled.csv"
    csv.write_text("example_id\nexample-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="path, export_id"):
        resolve_eval_score_input(workspace=workspace, task=Task.RETRIEVAL, path=csv, export_id="export-a")


def test_resolve_eval_score_input_rejects_missing_export_metadata(
    workspace: WorkspacePaths,
) -> None:
    """Export selection requires export metadata."""
    export_dir = workspace.base_dir / "annotation" / "exports" / "broken"
    export_dir.mkdir(parents=True)
    (export_dir / "retrieval.csv").write_text("example_id\nexample-1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Annotation export metadata does not exist"):
        resolve_eval_score_input(workspace=workspace, task=Task.RETRIEVAL, export_id="broken")


def test_provenance_path_relativizes_inside_workspace(tmp_path: Path) -> None:
    """An input under the workspace is recorded relative to it."""
    csv = tmp_path / "eval" / "scores" / "in.csv"
    assert provenance_path(input_csv=csv, base_dir=tmp_path) == "eval/scores/in.csv"


def test_provenance_path_keeps_absolute_outside_workspace(tmp_path: Path) -> None:
    """An input outside the workspace is recorded as its absolute path."""
    outside = Path("/data/external/labeled.csv")
    assert provenance_path(input_csv=outside, base_dir=tmp_path) == str(outside)
