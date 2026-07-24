"""API orchestration for evaluation workflows."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pragmata.api._error_log import error_log
from pragmata.core.eval.export import export_eval_predict_meta, export_eval_train_meta
from pragmata.core.eval.imports import (
    import_eval_predict_frame,
    import_eval_score_frame,
    import_eval_train_frame,
)
from pragmata.core.eval.scoring import ScoreReport, build_score_report
from pragmata.core.eval.tlmtc_adapters import run_tlmtc_predict, run_tlmtc_train
from pragmata.core.eval.transforms import build_tlmtc_frame
from pragmata.core.paths.eval_paths import (
    resolve_eval_predict_meta_path,
    resolve_eval_predict_paths,
    resolve_eval_score_input,
    resolve_eval_score_paths,
    resolve_eval_train_meta_path,
    resolve_eval_train_paths,
    resolve_eval_train_run_id,
)
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import EvalPredictMeta, EvalTrainMeta
from pragmata.core.settings.eval_settings import (
    EvalPredictSettings,
    EvalScoreSettings,
    EvalTrainSettings,
)
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file


def train_evaluator(
    *,
    labeled_data_path: str | Path | Unset = UNSET,
    export_id: str | Unset = UNSET,
    task: str | Task | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    target_name: str | Unset = UNSET,
    checkpoint: str | Unset = UNSET,
    proxy_checkpoint: str | Unset = UNSET,
    scale_learning_rate: bool | Unset = UNSET,
    sequence_length: int | Unset = UNSET,
    train_kwargs: dict[str, Any] | Unset = UNSET,
) -> Any:
    """Train a supervised evaluator model from labeled Pragmata eval data.

    Args:
        labeled_data_path: Optional direct path to labeled training data. If
            omitted, the latest or selected annotation export containing data
            for the selected ``task`` is used.
        export_id: Optional annotation export identifier. Ignored when
            ``labeled_data_path`` is provided. When used, the selected export
            is resolved to the task-specific training CSV for ``task``.
        task: Annotation task to train an evaluator for. Supported values are
            ``"retrieval"``, ``"grounding"``, and ``"generation"``.
        base_dir: Workspace base directory. Defaults to the current working
            directory.
        config_path: Path to a YAML configuration file.
        target_name: Display name passed to tlmtc for logs and reports.
            Defaults to a task-specific evaluation label, e.g. "Retrieval evaluation".
        checkpoint: Target checkpoint used for final fine-tuning. Defaults to
            ``"jhu-clsp/mmBERT-base"``.
        proxy_checkpoint: Proxy checkpoint used for hyperparameter tuning.
            Defaults to ``"jhu-clsp/mmBERT-small"``.
        scale_learning_rate: Whether tlmtc should scale a proxy-tuned learning
            rate for the target checkpoint. Defaults to ``True`` because the
            default proxy and target checkpoints differ.
        sequence_length: Maximum combined tokenized sequence length passed to
            tlmtc for ``text`` and ``text_pair``. Defaults to ``1024``.
        train_kwargs: Additional tlmtc-owned keyword arguments passed through to
            ``tlmtc.train_tlmtc``.

    Returns:
        Result metadata containing resolved filesystem paths for a single tlmtc
        training run, including the run ID, run directory, model directory,
        prepared split artifacts, metadata sidecar, and evaluation artifacts.
    """
    settings = EvalTrainSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,  # Environment-derived settings are not wired for train_evaluator yet.
        overrides={
            "base_dir": base_dir,
            "labeled_data_path": labeled_data_path,
            "export_id": export_id,
            "task": task,
            "target_name": target_name,
            "checkpoint": checkpoint,
            "proxy_checkpoint": proxy_checkpoint,
            "scale_learning_rate": scale_learning_rate,
            "sequence_length": sequence_length,
            "train_kwargs": train_kwargs,
        },
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    train_paths = resolve_eval_train_paths(
        workspace=workspace,
        task=settings.task,
        labeled_data_path=settings.labeled_data_path,
        export_id=settings.export_id,
    ).ensure_dirs()

    eval_frame = import_eval_train_frame(
        path=train_paths.training_input_csv,
        task=settings.task,
    )
    tlmtc_frame = build_tlmtc_frame(
        eval_frame,
        task=settings.task,
        mode="train",
    )

    assert settings.target_name is not None
    result = run_tlmtc_train(
        labeled_data=tlmtc_frame,
        work_dir=train_paths.tool_root,
        target_name=settings.target_name,
        checkpoint=settings.checkpoint,
        proxy_checkpoint=settings.proxy_checkpoint,
        scale_learning_rate=settings.scale_learning_rate,
        sequence_length=settings.sequence_length,
        train_kwargs=settings.train_kwargs,
    )
    export_eval_train_meta(
        meta=EvalTrainMeta(
            run_id=result.paths.run_id,
            task=settings.task,
            annotation_export_id=train_paths.annotation_export_id,
        ),
        path=resolve_eval_train_meta_path(
            workspace=workspace,
            run_id=result.paths.run_id,
        ),
    )

    return result


def predict_labels(
    *,
    unlabeled_data_path: str | Path | Unset = UNSET,
    evaluator_run_id: str | Unset = UNSET,
    task: str | Task | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    predict_kwargs: dict[str, Any] | Unset = UNSET,
) -> Any:
    """Predict evaluation labels for an unlabeled Pragmata eval dataset.

    Args:
        unlabeled_data_path: Direct path to the task-specific unlabeled CSV.
        evaluator_run_id: Optional evaluator training run identifier. If absent
            from both the call and configuration, Pragmata selects the latest
            evaluator compatible with ``task``.
        task: Evaluation task to predict labels for. Supported values are
            ``"retrieval"``, ``"grounding"``, and ``"generation"``.
        base_dir: Workspace base directory. Defaults to the current working
            directory.
        config_path: Path to a YAML configuration file.
        predict_kwargs: Additional tlmtc-owned keyword arguments passed through
            to ``tlmtc.predict_tlmtc``.

    Returns:
        Result metadata for the completed tlmtc prediction run. Its ``paths``
        attribute contains the resolved prediction filesystem layout, including
        the generated probabilities and predictions CSV artifacts. A
        ``pragmata_predict.meta.json`` sidecar is written beside those artifacts
        so the run can be discovered and scored (``pragmata eval score
        --prediction-id``).
    """
    settings = EvalPredictSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,  # Environment-derived settings are not wired for predict_labels yet.
        overrides={
            "base_dir": base_dir,
            "unlabeled_data_path": unlabeled_data_path,
            "evaluator_run_id": evaluator_run_id,
            "task": task,
            "predict_kwargs": predict_kwargs,
        },
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    predict_paths = resolve_eval_predict_paths(
        workspace=workspace,
        unlabeled_data_path=settings.unlabeled_data_path,
    ).ensure_dirs()
    resolved_evaluator_run_id = resolve_eval_train_run_id(
        workspace=workspace,
        task=settings.task,
        evaluator_run_id=settings.evaluator_run_id,
    )

    eval_frame = import_eval_predict_frame(
        path=predict_paths.prediction_input_csv,
        task=settings.task,
    )
    tlmtc_frame = build_tlmtc_frame(
        eval_frame,
        task=settings.task,
        mode="predict",
    )

    result = run_tlmtc_predict(
        unlabeled_data=tlmtc_frame,
        work_dir=predict_paths.tool_root,
        evaluator_run_id=resolved_evaluator_run_id,
        predict_kwargs=settings.predict_kwargs,
    )
    export_eval_predict_meta(
        meta=EvalPredictMeta(
            run_id=result.paths.run_id,
            task=settings.task,
            unlabeled_data_path=str(predict_paths.prediction_input_csv),
        ),
        path=resolve_eval_predict_meta_path(
            workspace=workspace,
            run_id=result.paths.run_id,
        ),
    )

    return result


def score(
    *,
    base_dir: str | Path | Unset = UNSET,
    score_id: str | Unset = UNSET,
    path: str | Path | Unset = UNSET,
    export_id: str | Unset = UNSET,
    prediction_id: str | Unset = UNSET,
    task: str | Task | Unset = UNSET,
    n_resamples: int | Unset = UNSET,
    ci: float | Unset = UNSET,
    seed: int | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
) -> ScoreReport:
    """Score labeled eval data into corpus metrics with confidence intervals.

    Reads one labeled CSV, computes the task's taxonomy metrics as corpus point
    estimates with a confidence interval on each (Wilson for proportion metrics,
    percentile bootstrap for the continuous retrieval metrics), and writes the
    task's ``*_scores.json`` report. The input is selected by a single mutually
    exclusive selector (``path``, ``export_id``, or ``prediction_id``); with no
    selector the latest annotation export is used. The resolved input and how it
    was selected are recorded on the report's ``source``.

    Args:
        base_dir: Workspace base directory. Defaults to the current directory.
        score_id: Output identifier; names ``<base_dir>/eval/scores/<score_id>/``.
            Defaults to a generated value.
        path: Direct path to the labeled CSV to score.
        export_id: Annotation export identifier; resolves to the task-specific
            exported CSV.
        prediction_id: Pragmata prediction run identifier. Not yet supported -
            prediction-output scoring lands with ``pragmata eval predict``.
        task: Annotation task to score (``"retrieval"``, ``"grounding"``, or
            ``"generation"``).
        n_resamples: Bootstrap iterations for the continuous metrics. Defaults to 1000.
        ci: Confidence level for every interval. Defaults to 0.95.
        seed: Optional RNG seed for reproducible bootstrap intervals.
        config_path: Path to a YAML configuration file.

    Returns:
        The task-specific score report, also written to ``*_scores.json``.

    Raises:
        ValueError: If more than one input selector (``path`` / ``export_id`` /
            ``prediction_id``) is given.
        EvalInputSchemaError: If the input violates the score contract.
        FileNotFoundError: If the selected input CSV, annotation export, or
            prediction run is missing.
    """
    settings = EvalScoreSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,
        overrides={
            "base_dir": base_dir,
            "score_id": score_id,
            "path": path,
            "export_id": export_id,
            "prediction_id": prediction_id,
            "task": task,
            "n_resamples": n_resamples,
            "ci": ci,
            "seed": seed,
        },
    )
    workspace = WorkspacePaths.from_base_dir(settings.base_dir)
    score_paths = resolve_eval_score_paths(workspace=workspace, score_id=settings.score_id).ensure_dirs()

    with error_log(score_paths.score_dir):
        resolved = resolve_eval_score_input(
            workspace=workspace,
            task=settings.task,
            path=settings.path,
            export_id=settings.export_id,
            prediction_id=settings.prediction_id,
        )
        frame = import_eval_score_frame(path=resolved.input_csv, task=settings.task, source=resolved.source)
        report = build_score_report(
            frame,
            task=settings.task,
            ci=settings.ci,
            n_resamples=settings.n_resamples,
            seed=settings.seed,
            source=resolved.source,
            created_at=datetime.now(UTC),
        )
        match settings.task:
            case Task.RETRIEVAL:
                output_json = score_paths.retrieval_scores_json
            case Task.GROUNDING:
                output_json = score_paths.grounding_scores_json
            case Task.GENERATION:
                output_json = score_paths.generation_scores_json
            case _:
                raise ValueError(f"Unsupported eval task: {settings.task!r}")
        output_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return report
