"""Dataframe imports for eval workflows."""

from pathlib import Path

import pandas as pd

from pragmata.core.eval.transforms import consolidate_labels_by_majority
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.schemas.eval_input import (
    TEXT_COLUMNS_BY_TASK,
    EvalInputSchemaError,
    validate_eval_predict_frame,
    validate_eval_score_frame,
    validate_eval_train_frame,
)
from pragmata.core.schemas.eval_output import ScoreInputSource


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


def import_eval_predict_frame(
    *,
    path: Path,
    task: Task,
) -> pd.DataFrame:
    """Read and validate an unlabeled eval prediction dataframe.

    Args:
        path: CSV path to read.
        task: Annotation task that determines the dataframe contract.

    Returns:
        Validated dataframe with original columns preserved.
    """
    frame = pd.read_csv(path, encoding="utf-8")
    return validate_eval_predict_frame(frame, task=task)


def import_eval_score_frame(
    *,
    path: Path,
    task: Task,
    source: ScoreInputSource,
) -> pd.DataFrame:
    """Read, prepare, and validate a labeled eval scoring dataframe.

    Direct paths and annotation exports are already Pragmata-shaped and are read
    and validated as-is. Prediction-run inputs are tlmtc-shaped - they carry the
    generic ``text``/``text_pair`` columns - so ``source.kind`` drives an inverse
    mapping back to the task-specific column names (via ``TEXT_COLUMNS_BY_TASK``)
    before validation; identity and label columns pass through unchanged.

    Each scoring unit must be unique so it is not double-counted in the metric
    denominators: retrieval rows are keyed by ``(record_uuid, chunk_id)`` (with
    ``chunk_rank`` unique within a query); grounding/generation are one row per
    ``record_uuid``. Multiple annotator rows for the same unit are consolidated to
    a single row by per-label majority (``consolidate_labels_by_majority``, shared
    with train ingestion) - a no-op when units are already unique.
    ``_guard_unique_scoring_units`` then runs as a post-collapse invariant: it still
    hard-errors on a residual duplicate the majority collapse cannot resolve, e.g.
    two distinct ``chunk_id``s sharing a ``chunk_rank``.

    Args:
        path: Resolved CSV path to read.
        task: Annotation task that determines the dataframe contract.
        source: Provenance of the input; ``source.kind`` decides whether the
            frame needs tlmtc text-column restoration.

    Returns:
        Validated dataframe with Pragmata task columns, one row per scoring unit.

    Raises:
        EvalInputSchemaError: If the frame violates the score contract or retains a
            duplicate scoring unit after majority consolidation.
    """
    frame = pd.read_csv(path, encoding="utf-8")
    if source.kind == "model_prediction":
        frame = _restore_pragmata_text_columns(frame, task=task)
    validated = validate_eval_score_frame(frame, task=task)
    consolidated = consolidate_labels_by_majority(validated, task=task)
    _guard_unique_scoring_units(consolidated, task=task)
    return consolidated


def _restore_pragmata_text_columns(frame: pd.DataFrame, *, task: Task) -> pd.DataFrame:
    """Invert the tlmtc predict mapping: restore task text columns from ``text``/``text_pair``."""
    text_column, text_pair_column = TEXT_COLUMNS_BY_TASK[task]
    return frame.rename(columns={"text": text_column, "text_pair": text_pair_column})


def _guard_unique_scoring_units(frame: pd.DataFrame, *, task: Task) -> None:
    """Reject duplicate scoring units that would double-count in metric means."""
    if task == Task.RETRIEVAL:
        _reject_duplicates(frame, ["record_uuid", "chunk_id"], task, "chunk")
        _reject_duplicates(frame, ["record_uuid", "chunk_rank"], task, "chunk rank")
    else:
        _reject_duplicates(frame, ["record_uuid"], task, "query")


def _reject_duplicates(frame: pd.DataFrame, keys: list[str], task: Task, unit: str) -> None:
    duplicated = frame.duplicated(subset=keys, keep=False)
    if duplicated.any():
        raise EvalInputSchemaError(
            f"Scoring input for {task.value} has {int(duplicated.sum())} row(s) with a duplicate "
            f"{unit} key {tuple(keys)}; each unit must be unique so metric denominators are not double-counted."
        )
