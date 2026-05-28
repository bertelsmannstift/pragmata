"""Read helper for the frozen Stage 1 result artifact (``selected_blueprints.json``)."""

import importlib.metadata
import json
from pathlib import Path

from pragmata.core.schemas.querygen_output import SelectedBlueprintsArtifact


def read_selected_blueprints_artifact(
    *,
    path: Path,
    expected_spec_fingerprint: str,
    expected_source_run_id: str,
    expected_n_queries: int,
    expected_batch_size: int,
    expected_near_duplicate_tolerance: float,
) -> SelectedBlueprintsArtifact | None:
    """Load and validate the frozen Stage 1 result at ``path``.

    Args:
        path: Frozen-result path (``selected_blueprints.json``).
        expected_spec_fingerprint: Fingerprint the current run expects.
        expected_source_run_id: Run identifier the current run expects.
        expected_n_queries: Total queries the current run expects.
        expected_batch_size: Batch size the current run expects.
        expected_near_duplicate_tolerance: Deduplication tolerance the current
            run expects.

    Returns:
        The validated artifact, or ``None`` when the file is absent or its
        validated header does not match the current run (spec fingerprint,
        running pragmata version, source run id, n_queries, batch_size,
        near_duplicate_tolerance). ``embedding_model`` is recorded for
        provenance but is intentionally NOT validated here.

    Raises:
        json.JSONDecodeError: The file content is not valid JSON.
        pydantic.ValidationError: The JSON does not match the artifact schema.

    The api layer catches these and treats the frozen result as absent.
    """
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = SelectedBlueprintsArtifact.model_validate(payload)

    if (
        artifact.spec_fingerprint != expected_spec_fingerprint
        or artifact.pragmata_version != importlib.metadata.version("pragmata")
        or artifact.source_run_id != expected_source_run_id
        or artifact.n_queries != expected_n_queries
        or artifact.batch_size != expected_batch_size
        or artifact.near_duplicate_tolerance != expected_near_duplicate_tolerance
    ):
        return None

    return artifact
