"""Transforms from pragmata eval frames to tlmtc-compatible frames."""

import logging
from typing import Literal

import pandas as pd

from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import LABEL_COLUMNS_BY_TASK, TEXT_COLUMNS_BY_TASK

logger = logging.getLogger(__name__)

_DUPLICATE_KEY_COLUMNS_BY_TASK: dict[Task, tuple[str, ...]] = {
    Task.RETRIEVAL: ("record_uuid", "chunk_id"),
    Task.GROUNDING: ("record_uuid",),
    Task.GENERATION: ("record_uuid",),
}


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
            to ``label_*`` columns. Retrieval train mode additionally maps
            ``record_uuid`` to ``split_group`` so rows from the same retrieval
            example stay in the same train/validation/test split. Predict mode
            only renames text columns.

    Returns:
        Dataframe with all input columns preserved, a reset integer index, and
        task-specific columns renamed to tlmtc's ``text``, ``text_pair``, and
        train-only ``label_*``, and retrieval-train-only ``split_group`` names.

    Raises:
        ValueError: If ``mode`` is unsupported or the input already contains a
            tlmtc-reserved output column that Pragmata derives internally.
    """
    if mode not in {"train", "predict"}:
        raise ValueError(f"Unsupported eval transform mode: {mode!r}.")

    consolidated_frame = consolidate_labels_by_majority(frame, task=task) if mode == "train" else frame

    text_column, text_pair_column = TEXT_COLUMNS_BY_TASK[task]
    rename_map = {
        text_column: "text",
        text_pair_column: "text_pair",
    }

    if mode == "train":
        source_label_columns = LABEL_COLUMNS_BY_TASK[task]
        rename_map.update({column: f"label_{column}" for column in source_label_columns})

        if task == Task.RETRIEVAL:
            rename_map["record_uuid"] = "split_group"

    reserved_columns = sorted(set(rename_map.values()).intersection(consolidated_frame.columns))
    if reserved_columns:
        raise ValueError(f"Input contains reserved tlmtc columns that Pragmata derives internally: {reserved_columns}.")

    return consolidated_frame.reset_index(drop=True).rename(columns=rename_map)


def consolidate_labels_by_majority(
    frame: pd.DataFrame,
    *,
    task: Task,
) -> pd.DataFrame:
    """Collapse repeated annotation units to one row via per-label majority consensus.

    Shared by eval train and score ingestion: multiple annotator rows for the same
    scoring unit (retrieval ``(record_uuid, chunk_id)``; grounding/generation
    ``record_uuid``) are reduced to a single row. Each label with a strict majority
    (> half positive) is set independently; a tied label (exact 50/50 on an even
    number of annotators) falls back to the selected source row's value. When an
    observed row matches all strict-majority labels, that row is selected to preserve
    non-label metadata deterministically. A no-op when no unit is duplicated.
    """
    key_columns = _DUPLICATE_KEY_COLUMNS_BY_TASK[task]
    label_columns = LABEL_COLUMNS_BY_TASK[task]

    if any(column not in frame.columns for column in key_columns):
        return frame

    duplicate_mask = frame.duplicated(subset=key_columns, keep=False)
    if not duplicate_mask.any():
        return frame

    working = frame.reset_index(drop=True)
    duplicate_mask = working.duplicated(subset=key_columns, keep=False)
    subsequent_duplicate_mask = working.duplicated(subset=key_columns, keep="first")
    working_labels = working.loc[:, label_columns].astype("int64")

    consolidated = working.loc[~subsequent_duplicate_mask].copy()

    replacement_sources_by_target: dict[int, int] = {}
    label_overrides_by_target: dict[int, pd.Series] = {}

    duplicate_groups = working.loc[duplicate_mask].groupby(
        list(key_columns),
        sort=False,
        dropna=False,
    )

    for _, group in duplicate_groups:
        target_position = group.index[0]
        selected_position = target_position

        labels = working_labels.loc[group.index]
        positive_counts = labels.sum(axis=0)
        majority_threshold = len(group) / 2

        strict_majority_labels = positive_counts.ne(majority_threshold)
        majority_vector = positive_counts.gt(majority_threshold).astype("int64")

        if strict_majority_labels.any():
            matching_rows = (
                labels.loc[:, strict_majority_labels]
                .eq(
                    majority_vector.loc[strict_majority_labels],
                    axis=1,
                )
                .all(axis=1)
            )

            if matching_rows.any():
                selected_position = matching_rows.idxmax()

            label_overrides_by_target[target_position] = majority_vector.loc[strict_majority_labels]

        if selected_position != target_position:
            replacement_sources_by_target[target_position] = selected_position

    if replacement_sources_by_target:
        target_positions = list(replacement_sources_by_target)
        source_positions = list(replacement_sources_by_target.values())

        consolidated.loc[target_positions, :] = working.loc[
            source_positions,
            consolidated.columns,
        ].to_numpy()

    for target_position, label_override in label_overrides_by_target.items():
        consolidated.loc[target_position, label_override.index] = label_override

    logger.info(
        "Consolidated duplicate eval rows by majority for %s: input_rows=%d output_rows=%d "
        "collapsed_rows=%d duplicate_units=%d key_columns=%s",
        task.value,
        len(frame),
        len(consolidated),
        len(frame) - len(consolidated),
        duplicate_groups.ngroups,
        key_columns,
    )

    return consolidated.reset_index(drop=True)
