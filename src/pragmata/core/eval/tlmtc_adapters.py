"""tlmtc call boundary for eval workflows."""

from pathlib import Path
from typing import Any

import pandas as pd

# Keep this in sync with the TLMTC arguments managed by run_tlmtc_train.
_PRAGMATA_MANAGED_TLMTC_TRAIN_ARGS = frozenset(
    {
        "labeled_data",
        "work_dir",
        "target_name",
        "checkpoint",
        "proxy_checkpoint",
        "scale_learning_rate",
        "sequence_length",
    }
)

# Keep this in sync with the TLMTC arguments managed by run_tlmtc_predict.
_PRAGMATA_MANAGED_TLMTC_PREDICT_ARGS = frozenset(
    {
        "unlabeled_data",
        "work_dir",
        "run_id",
    }
)


def run_tlmtc_train(
    *,
    labeled_data: str | Path | pd.DataFrame,
    work_dir: Path,
    target_name: str,
    checkpoint: str,
    proxy_checkpoint: str,
    scale_learning_rate: bool,
    sequence_length: int,
    train_kwargs: dict[str, Any],
) -> Any:
    """Train a pragmata evaluator model through tlmtc.

    Args:
        labeled_data: Path to labeled multi-label training data, or an in-memory
            DataFrame. The data must contain a ``text`` column, at least two
            binary ``label_*`` columns, and optionally a ``text_pair`` column.
        work_dir: Base eval work directory passed through to tlmtc.
        target_name: Display name used in tlmtc logs and reports.
        checkpoint: Target checkpoint used for final fine-tuning.
        proxy_checkpoint: Proxy checkpoint used for hyperparameter tuning.
        scale_learning_rate: Whether to scale the proxy-tuned learning rate for
            the target checkpoint.
        sequence_length: Maximum tokenized sequence length.
        train_kwargs: Additional tlmtc-owned keyword arguments. Keys managed by
            pragmata are rejected.

    Returns:
        The result returned by ``tlmtc.train_tlmtc``.

    Raises:
        ImportError: If tlmtc is not installed.
        ValueError: If ``train_kwargs`` attempts to override a dedicated
            Pragmata-owned train argument.
    """
    overlapping_keys = _PRAGMATA_MANAGED_TLMTC_TRAIN_ARGS.intersection(train_kwargs)
    if overlapping_keys:
        overlapping = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"train_kwargs must not override pragmata-managed train settings: {overlapping}. "
            "Pass these via dedicated arguments instead."
        )

    try:
        from tlmtc import train_tlmtc
    except ImportError as exc:
        raise ImportError("tlmtc is required for evaluator training. Install pragmata with the 'eval' extra.") from exc

    train_args: dict[str, Any] = {
        "labeled_data": labeled_data,
        "work_dir": work_dir,
        "target_name": target_name,
        "checkpoint": checkpoint,
        "proxy_checkpoint": proxy_checkpoint,
        "scale_learning_rate": scale_learning_rate,
        "sequence_length": sequence_length,
    }
    train_args.update(train_kwargs)

    return train_tlmtc(**train_args)


def run_tlmtc_predict(
    *,
    unlabeled_data: str | Path | pd.DataFrame,
    work_dir: Path,
    evaluator_run_id: str,
    predict_kwargs: dict[str, Any],
) -> Any:
    """Predict evaluation labels through tlmtc.

    Args:
        unlabeled_data: Path to unlabeled prediction data, or an in-memory
            DataFrame containing tlmtc's text input columns.
        work_dir: Base eval work directory passed through to tlmtc.
        evaluator_run_id: Concrete, task-compatible evaluator training run selected by pragmata.
            This is forwarded to tlmtc as ``run_id``.
        predict_kwargs: Additional tlmtc-owned prediction arguments, such as batch
            size, device selection, verbosity, and inference backend. Arguments
            managed by Pragmata are rejected.

    Returns:
        The result returned by ``tlmtc.predict_tlmtc``.

    Raises:
        ImportError: If tlmtc is not installed.
        ValueError: If ``predict_kwargs`` attempts to override a dedicated
            Pragmata-owned prediction argument.
    """
    overlapping_keys = _PRAGMATA_MANAGED_TLMTC_PREDICT_ARGS.intersection(predict_kwargs)
    if overlapping_keys:
        overlapping = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"predict_kwargs must not override pragmata-managed predict settings: {overlapping}. "
            "Pass these via dedicated arguments instead."
        )

    try:
        from tlmtc import predict_tlmtc
    except ImportError as exc:
        raise ImportError(
            "tlmtc is required for evaluator prediction. Install pragmata with the 'eval' extra."
        ) from exc

    predict_args: dict[str, Any] = {
        "unlabeled_data": unlabeled_data,
        "work_dir": work_dir,
        "run_id": evaluator_run_id,
    }
    predict_args.update(predict_kwargs)

    return predict_tlmtc(**predict_args)
