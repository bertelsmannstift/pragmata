"""Transforms from pragmata eval frames to tlmtc-compatible frames."""

from typing import Literal

import pandas as pd

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import LABEL_COLUMNS_BY_TASK, TEXT_COLUMNS_BY_TASK


def build_tlmtc_frame(
    frame: pd.DataFrame,
    *,
    task: Task,
    mode: Literal["train", "predict"],
) -> pd.DataFrame:
    """Rename validated Pragmata eval columns to tlmtc's expected names.

    Args:
        frame: Validated eval dataframe returned by ``import_eval_train_frame``
            or ``import_eval_predict_frame``.
        task: Eval task that determines the source text, text-pair, and label
            columns.
        mode: Target tlmtc workflow. Train mode also renames task label columns
            to ``label_*`` columns. Predict mode only renames text columns.

    Returns:
        Dataframe with all input columns preserved, a reset integer index, and
        task-specific columns renamed to tlmtc's ``text``, ``text_pair``, and
        train-only ``label_*`` names.

    Raises:
        ValueError: If ``mode`` is unsupported or the input already contains a
            tlmtc-reserved output column that Pragmata derives internally.
    """
    if mode not in {"train", "predict"}:
        raise ValueError(f"Unsupported eval transform mode: {mode!r}.")

    text_column, text_pair_column = TEXT_COLUMNS_BY_TASK[task]
    rename_map = {
        text_column: "text",
        text_pair_column: "text_pair",
    }

    if mode == "train":
        source_label_columns = LABEL_COLUMNS_BY_TASK[task]
        label_columns = tuple(f"label_{column}" for column in source_label_columns)
        rename_map.update(dict(zip(source_label_columns, label_columns, strict=True)))

    reserved_columns = sorted(set(rename_map.values()).intersection(frame.columns))
    if reserved_columns:
        raise ValueError(f"Input contains reserved tlmtc columns that Pragmata derives internally: {reserved_columns}.")

    return frame.reset_index(drop=True).rename(columns=rename_map)
