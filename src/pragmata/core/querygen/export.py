"""Export assembled synthetic query artifacts to disk."""

import json
import os
from pathlib import Path
from uuid import uuid4

from pragmata.core.csv_io import write_csv
from pragmata.core.schemas.querygen_output import (
    PlanningBatchArtifact,
    PlanningSummaryArtifact,
    RealizationBatchArtifact,
    SelectedBlueprintsArtifact,
    SyntheticQueriesMeta,
    SyntheticQueryRow,
)


def _atomic_write_json(*, data: dict, path: Path) -> None:
    """Atomically write ``data`` as JSON to ``path`` (tempfile + rename).

    The tempfile name is uniquified with PID + a random hex suffix so that two
    writers targeting the same path (e.g. a half-dead run and a relaunch under
    the same run_id) cannot clobber each other's tempfile mid-write; the final
    ``Path.replace`` is atomic on a POSIX filesystem. No fsync (matching the
    repo's existing atomic-write idiom): a torn or zero-length file left by a
    crash fails validation on read and is treated as drift -> recompute, so the
    durability gap cannot silently corrupt a run.
    """
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def export_queries(
    rows: list[SyntheticQueryRow],
    meta: SyntheticQueriesMeta,
    queries_path: Path,
    meta_path: Path,
) -> None:
    """Export synthetic query rows to CSV and run metadata to JSON.

    Args:
        rows: Final assembled synthetic query rows to export as CSV.
        meta: Dataset-level metadata sidecar to export as JSON.
        queries_path: Output path for the synthetic query CSV file.
        meta_path: Output path for the synthetic query metadata JSON file.
    """
    write_csv(rows, queries_path)
    meta_path.write_text(
        json.dumps(meta.model_dump(mode="json")),
        encoding="utf-8",
    )


def export_planning_summary(
    artifact: PlanningSummaryArtifact,
    artifact_path: Path,
) -> None:
    """Write a planning-summary artifact to disk as JSON.

    Args:
        artifact: Validated planning-summary artifact to persist.
        artifact_path: Destination path for the JSON artifact.
    """
    _atomic_write_json(data=artifact.model_dump(mode="json"), path=artifact_path)


def export_planning_batch_artifact(
    *,
    artifact: PlanningBatchArtifact,
    path: Path,
) -> None:
    """Atomically persist a Stage 1 planning-batch artifact as JSON.

    Args:
        artifact: Validated planning-batch artifact to persist.
        path: Destination path (``planning_batches/batch_NNNN.json``).
    """
    _atomic_write_json(data=artifact.model_dump(mode="json"), path=path)


def export_selected_blueprints(
    *,
    artifact: SelectedBlueprintsArtifact,
    path: Path,
) -> None:
    """Atomically persist the frozen Stage 1 result artifact as JSON.

    Args:
        artifact: Validated selected-blueprints artifact to persist.
        path: Destination path (``selected_blueprints.json``).
    """
    _atomic_write_json(data=artifact.model_dump(mode="json"), path=path)


def export_realization_batch_artifact(
    *,
    artifact: RealizationBatchArtifact,
    path: Path,
) -> None:
    """Atomically persist a Stage 2 realization-batch artifact as JSON.

    Args:
        artifact: Validated realization-batch artifact to persist.
        path: Destination path (``realization_batches/batch_NNNN.json``).
    """
    _atomic_write_json(data=artifact.model_dump(mode="json"), path=path)
