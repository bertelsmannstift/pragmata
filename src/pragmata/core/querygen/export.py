"""Export assembled synthetic query artifacts to disk."""

from pathlib import Path

from pragmata.core.atomic_io import atomic_write_json
from pragmata.core.csv_io import write_csv
from pragmata.core.schemas.querygen_output import (
    PlanningSummaryArtifact,
    SyntheticQueriesMeta,
    SyntheticQueryRow,
)


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
    atomic_write_json(meta.model_dump(mode="json"), meta_path)


def export_planning_summary(
    artifact: PlanningSummaryArtifact,
    artifact_path: Path,
) -> None:
    """Write a planning-summary artifact to disk as JSON.

    Args:
        artifact: Validated planning-summary artifact to persist.
        artifact_path: Destination path for the JSON artifact.
    """
    atomic_write_json(artifact.model_dump(mode="json"), artifact_path)
