"""Dataframe imports for eval workflows."""

from pathlib import Path

import pandas as pd

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import validate_eval_train_frame


def import_eval_train_frame(
    *,
    path: Path,
    task: Task,
) -> pd.DataFrame:
    """Read and validate a labeled eval training dataframe.

    Args:
        path: Resolved CSV path to read.
        task: Annotation task that determines the dataframe contract.

    Returns:
        Validated dataframe with original columns preserved.
    """
    frame = pd.read_csv(path, encoding="utf-8")
    return validate_eval_train_frame(frame, task=task)
