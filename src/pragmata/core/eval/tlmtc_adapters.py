"""tlmtc call boundary for eval workflows."""

from pathlib import Path
from typing import Any

_RESERVED_TRAIN_KWARGS = frozenset(
    {
        "raw_csv",
        "work_dir",
        "target_name",
        "checkpoint",
        "proxy_checkpoint",
        "scale_learning_rate",
        "sequence_length",
        "trust_remote_code",
    }
)


def train_evaluator(
    *,
    raw_csv: Path,
    work_dir: Path,
    target_name: str,
    checkpoint: str,
    proxy_checkpoint: str,
    scale_learning_rate: bool,
    sequence_length: int,
    trust_remote_code: bool,
    train_kwargs: dict[str, Any],
) -> Any:
    """Train a pragmata evaluator model through tlmtc.

    Args:
        raw_csv: tlmtc-compatible labeled training CSV.
        work_dir: Base eval work directory passed through to tlmtc.
        target_name: Display name used in tlmtc logs and reports.
        checkpoint: Target checkpoint used for final fine-tuning.
        proxy_checkpoint: Proxy checkpoint used for hyperparameter tuning.
        scale_learning_rate: Whether to scale the proxy-tuned learning rate for
            the target checkpoint.
        sequence_length: Maximum tokenized sequence length.
        trust_remote_code: Whether Hugging Face loading may execute custom
            checkpoint code.
        train_kwargs: Additional tlmtc-owned keyword arguments. Keys managed by
            pragmata are rejected.

    Returns:
        The result returned by ``tlmtc.train_tlmtc``.

    Raises:
        ImportError: If tlmtc is not installed.
        ValueError: If ``train_kwargs`` attempts to override a dedicated
            Pragmata-owned train argument.
    """
    overlapping_keys = _RESERVED_TRAIN_KWARGS.intersection(train_kwargs)
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
        "raw_csv": raw_csv,
        "work_dir": work_dir,
        "target_name": target_name,
        "checkpoint": checkpoint,
        "proxy_checkpoint": proxy_checkpoint,
        "scale_learning_rate": scale_learning_rate,
        "sequence_length": sequence_length,
        "trust_remote_code": trust_remote_code,
    }
    train_args.update(train_kwargs)

    return train_tlmtc(**train_args)
