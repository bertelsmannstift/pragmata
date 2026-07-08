"""Export evaluator artifacts to disk."""

from pathlib import Path

from pragmata.core.atomic_io import atomic_write_json
from pragmata.core.schemas.eval_output import EvalTrainMeta


def export_eval_train_meta(
    meta: EvalTrainMeta,
    path: Path,
) -> None:
    """Write Pragmata-owned evaluator training metadata to disk as JSON.

    Args:
        meta: Validated evaluator training metadata to persist.
        path: Destination path for the JSON sidecar.
    """
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Eval train metadata parent directory does not exist: {path.parent}")

    atomic_write_json(meta.model_dump(mode="json"), path)
