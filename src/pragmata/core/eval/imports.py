"""Dataframe imports for eval workflows."""

import logging
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

logger = logging.getLogger(__name__)


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
    allow_incomplete_panels: bool = False,
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
        allow_incomplete_panels: Permit retrieval panels whose labeled chunks do not
            cover ``n_retrieved_chunks``. Off by default; see
            ``_guard_complete_panels``.

    Returns:
        Validated dataframe with Pragmata task columns, one row per scoring unit.

    Raises:
        EvalInputSchemaError: If the frame violates the score contract, retains a
            duplicate scoring unit after majority consolidation, or contains an
            incomplete retrieval panel without ``allow_incomplete_panels``.
    """
    frame = pd.read_csv(path, encoding="utf-8")
    if source.kind == "model_prediction":
        frame = _restore_pragmata_text_columns(frame, task=task)
    validated = validate_eval_score_frame(frame, task=task)
    consolidated = consolidate_labels_by_majority(validated, task=task)
    _guard_unique_scoring_units(consolidated, task=task)
    _guard_complete_panels(consolidated, task=task, allow_incomplete=allow_incomplete_panels)
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


def _guard_complete_panels(frame: pd.DataFrame, *, task: Task, allow_incomplete: bool) -> None:
    """Reject retrieval panels whose labeled chunks do not cover the retrieval.

    Every retrieval metric averages over a query's chunk set, so a partial panel is
    not a smaller sample of the same quantity - it changes the metric. Precision@K
    over 2 labeled chunks of a K=5 retrieval has the wrong denominator, and the
    rank-sensitive metrics (MRR, NDCG) are biased upward because annotators work
    top-down, so the unjudged low-rank chunks can never lower the score.

    The check needs ``n_retrieved_chunks`` (the query's true K, carried by annotation
    exports). Where no panel carries a known K the completeness of the input is
    unknowable here, so the frame passes with a warning rather than a hard failure -
    direct-path and prediction inputs are not required to carry export metadata.

    ``allow_incomplete=True`` (CLI: ``--allow-incomplete-panels``) skips the check for
    callers who accept the bias, e.g. to score everything for a coverage comparison.
    """
    if task != Task.RETRIEVAL or allow_incomplete:
        return
    panels = _panels_with_known_k(frame)
    if panels.empty:
        logger.warning(
            "score input: retrieval panel completeness unverifiable (no panel carries "
            "n_retrieved_chunks metadata) - metrics over partial panels are biased upward, "
            "see import_eval_score_frame."
        )
        return
    short = panels[panels["n_chunks"] < panels["k"]]
    if not short.empty:
        raise EvalInputSchemaError(
            f"Scoring input for {task.value} has {len(short)} panel(s) with fewer labeled chunks "
            f"than n_retrieved_chunks (e.g. record_uuid {short.index[0]!r}: "
            f"{int(short.iloc[0]['n_chunks'])} of {int(short.iloc[0]['k'])}); partial panels bias "
            f"every retrieval metric, so filter them out or pass allow_incomplete_panels=True "
            f"(--allow-incomplete-panels) to score anyway."
        )


def _reject_duplicates(frame: pd.DataFrame, keys: list[str], task: Task, unit: str) -> None:
    duplicated = frame.duplicated(subset=keys, keep=False)
    if duplicated.any():
        raise EvalInputSchemaError(
            f"Scoring input for {task.value} has {int(duplicated.sum())} row(s) with a duplicate "
            f"{unit} key {tuple(keys)}; each unit must be unique so metric denominators are not double-counted."
        )


def _panels_with_known_k(frame: pd.DataFrame) -> pd.DataFrame:
    """Count distinct labeled chunks and K per retrieval panel, keeping only panels with a known K.

    ``n_retrieved_chunks`` is absent from direct-path and prediction inputs, and is the
    ``-1`` sentinel on export records predating the metadata backfill; both mean "K
    unknown here", so such panels cannot be checked for completeness.
    """
    if "n_retrieved_chunks" not in frame.columns:
        return pd.DataFrame(columns=["n_chunks", "k"])
    per_query = frame.groupby("record_uuid").agg(
        n_chunks=("chunk_id", "nunique"),
        k=("n_retrieved_chunks", "max"),
    )
    return per_query[per_query["k"] > 0]
