"""Export assembled synthetic query artifacts to disk."""

import json
from pathlib import Path

from pragmata.core.csv_io import write_csv
from pragmata.core.schemas.querygen_output import PlanningSummaryArtifact, SyntheticQueriesMeta, SyntheticQueryRow


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
    """Write a planning-memory artifact to disk as JSON.

    Args:
        artifact: Validated planning-memory artifact to persist.
        path: Destination path for the JSON artifact.
    """
    artifact_path.write_text(
        json.dumps(artifact.model_dump(mode="json")),
        encoding="utf-8",
    )
