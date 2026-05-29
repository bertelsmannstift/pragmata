"""Export assembled synthetic query artifacts to disk."""

import json
from pathlib import Path

from pragmata.core.atomic_io import atomic_write_text
from pragmata.core.csv_io import write_csv
from pragmata.core.schemas.querygen_output import (
    PlanningBatchArtifact,
    PlanningSummaryArtifact,
    SyntheticQueriesMeta,
    SyntheticQueryRow,
)


def _atomic_write_json(*, data: dict, path: Path) -> None:
    """Atomically write ``data`` as indented JSON to ``path`` via :func:`atomic_write_text`.

    Indented (``indent=2``) so the persisted checkpoint/summary artifacts are
    human-scannable when inspected; round-trip reads validate semantically, so
    the formatting carries no functional meaning.
    """
    with atomic_write_text(path) as handle:
        handle.write(json.dumps(data, indent=2) + "\n")


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
