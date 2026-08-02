"""Export evaluator artifacts to disk."""

from pathlib import Path

from pragmata.core.atomic_io import atomic_write_json
from pragmata.core.schemas.eval_output import EvalPredictMeta, EvalTrainMeta


def export_eval_meta(meta: EvalTrainMeta | EvalPredictMeta, path: Path) -> None:
    """Write Pragmata-owned evaluator run metadata to disk as JSON.

    Shared by train and predict runs (``EvalTrainMeta`` / ``EvalPredictMeta``);
    the sidecar is written beside the run's tlmtc artifacts.

    Args:
        meta: Validated evaluator run metadata to persist.
        path: Destination path for the JSON sidecar.
    """
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Eval metadata parent directory does not exist: {path.parent}")

    atomic_write_json(meta.model_dump(mode="json"), path)
