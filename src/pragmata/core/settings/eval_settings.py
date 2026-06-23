"""Evaluation workflow settings."""

from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from pydantic import Field, PositiveInt, model_validator

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.settings_base import ResolveSettings

DEFAULT_EVAL_TRAIN_TARGET_NAMES: dict[Task, str] = {
    Task.RETRIEVAL: "Retrieval evaluation",
    Task.GROUNDING: "Grounding evaluation",
    Task.GENERATION: "Generation evaluation",
}


class EvalTrainSettings(ResolveSettings):
    """Evaluator training settings.

    Attributes:
        base_dir: Workspace base directory. Pragmata resolves the eval tool root
            under this directory and passes `<base_dir>/eval/` to tlmtc as its
            training `work_dir`.
        labeled_data_path: Optional direct path to labeled training data. Use this
            for standalone eval training without relying on annotation-tool export
            discovery. If omitted, the training workflow selects an annotation
            export by `export_id` or falls back to the latest available export.
        export_id: Optional annotation export identifier used to select a specific
            exported annotation dataset. Ignored when `labeled_data_path` is
            provided.
        task: Annotation task to train an evaluator for. The task determines the
            Pragmata-owned task mapping, serialization, labels, and internally
            derived split-group column.
        target_name: Display name passed to tlmtc for training logs and reports.
            Defaults to a task-specific Pragmata label.
        checkpoint: Encoder-only Hugging Face checkpoint identifier or local path
            used for final fine-tuning.
        proxy_checkpoint: Encoder-only Hugging Face checkpoint identifier used for
            hyperparameter tuning.
        scale_learning_rate: Whether tlmtc should scale a proxy-tuned learning
            rate for the target checkpoint.
        sequence_length: Maximum tokenized sequence length passed to tlmtc.
            Defaults to 1024 for paired RAG text with long-context EuroBERT
            checkpoints.
        trust_remote_code: Whether Hugging Face loading may execute custom remote
            code for the selected checkpoints.
        train_kwargs: Additional `train_tlmtc`-specific keyword arguments passed
            through to the tlmtc API. Use this for tlmtc-owned options.
    """

    base_dir: Path = Field(default_factory=Path.cwd)
    labeled_data_path: Path | None = None
    export_id: str | None = None
    task: Task
    target_name: str | None = Field(default=None, min_length=1)
    checkpoint: str = Field(default="EuroBERT/EuroBERT-610m", min_length=1)
    proxy_checkpoint: str = Field(default="EuroBERT/EuroBERT-210m", min_length=1)
    scale_learning_rate: bool = True
    sequence_length: PositiveInt = 1024
    trust_remote_code: bool = True
    train_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def default_target_name(
        self,
    ) -> Self:
        """Fill the task-specific target name when the caller omits one."""
        if self.target_name is None:
            self.target_name = DEFAULT_EVAL_TRAIN_TARGET_NAMES[self.task]
        return self


class EvalPredictSettings(ResolveSettings):
    """Evaluator prediction settings.

    Attributes:
        base_dir: Workspace base directory. Pragmata resolves the eval tool root
            under this directory and passes `<base_dir>/eval/` to tlmtc as its
            prediction `work_dir`.
        unlabeled_data_path: Direct path to unlabeled data that should receive
            predicted evaluation labels. Prediction input is intentionally explicit;
            it is not inferred from prior tool outputs.
        evaluator_run_id: Optional trained evaluator run identifier. This selects
            the tlmtc training run to load. If omitted, tlmtc's default latest-run
            lookup is used.
        task: Annotation task to predict labels for. The task determines the
            Pragmata-owned task mapping, serialization, labels, and output contract.
        predict_kwargs: Additional `predict_tlmtc`-specific keyword arguments passed
            through to the tlmtc API. Use this for tlmtc-owned options.
    """

    base_dir: Path = Field(default_factory=Path.cwd)
    unlabeled_data_path: Path
    evaluator_run_id: str | None = None
    task: Task
    predict_kwargs: dict[str, Any] = Field(default_factory=dict)


class EvalScoreSettings(ResolveSettings):
    """Evaluation scoring settings.

    Attributes:
        base_dir: Workspace base directory. Pragmata resolves score artifacts under
            `<base_dir>/eval/scores/<score_id>/`.
        score_id: Unique identifier for the score run. Used to name the
            Pragmata-owned score artifact directory.
        labeled_input_path: Optional direct path to labeled data to score. Use this
            for standalone scoring, including human-labeled datasets or externally
            prepared labeled records. If provided, it takes precedence over
            `prediction_run_id`.
        prediction_run_id: Optional Pragmata prediction run identifier. Use this to
            score labels produced by `pragmata eval predict`. If omitted together
            with `labeled_input_path`, the scoring workflow selects the latest
            available prediction run.
        task: Annotation task to score. The task determines which task-specific
            label contract and score metrics are applied.
    """

    base_dir: Path = Field(default_factory=Path.cwd)
    score_id: str = Field(default_factory=lambda: uuid4().hex)
    labeled_input_path: Path | None = None
    prediction_run_id: str | None = None
    task: Task
