"""API orchestration for evaluation workflows."""

from pathlib import Path
from typing import Any

from pragmata.core.eval.export import export_eval_train_meta
from pragmata.core.eval.imports import import_eval_train_frame
from pragmata.core.eval.tlmtc_adapters import run_tlmtc_train
from pragmata.core.eval.transforms import build_tlmtc_frame
from pragmata.core.paths.eval_paths import resolve_eval_train_meta_path, resolve_eval_train_paths
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_output import EvalTrainMeta
from pragmata.core.settings.eval_settings import EvalTrainSettings
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
    trust_remote_code: bool | Unset = UNSET,
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
            ``"EuroBERT/EuroBERT-610m"``.
        proxy_checkpoint: Proxy checkpoint used for hyperparameter tuning.
            Defaults to ``"EuroBERT/EuroBERT-210m"``.
        scale_learning_rate: Whether tlmtc should scale a proxy-tuned learning
            rate for the target checkpoint. Defaults to ``True`` because the
            default proxy and target checkpoints differ.
        sequence_length: Maximum combined tokenized sequence length passed to
            tlmtc for ``text`` and ``text_pair``. Defaults to ``1024``.
        trust_remote_code: Whether Hugging Face loading may execute custom
            checkpoint code. Defaults to ``True`` because it is required by the
            default checkpoint and proxy checkpoint.
        train_kwargs: Additional tlmtc-owned keyword arguments passed through to
            ``tlmtc.train_tlmtc``.

    Returns:
        Result metadata containing resolved filesystem paths for a single tlmtc
        training run, including the run ID, run directory, model directory,
        prepared split artifacts, metadata sidecar, and evaluation artifacts
        under ``<base_dir>/eval/train_outputs/<run_id>/``.
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
            "trust_remote_code": trust_remote_code,
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
        trust_remote_code=settings.trust_remote_code,
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
