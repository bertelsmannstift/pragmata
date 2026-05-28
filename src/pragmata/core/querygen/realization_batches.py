"""Read helper for Stage 2 realization-batch checkpoint artifacts."""

import importlib.metadata
import json
from pathlib import Path

from pragmata.core.schemas.querygen_output import RealizationBatchArtifact


def read_realization_batch_artifact(
    *,
    path: Path,
    expected_spec_fingerprint: str,
    expected_source_run_id: str,
    expected_n_queries: int,
    expected_batch_size: int,
    expected_candidate_ids: list[str],
) -> RealizationBatchArtifact | None:
    """Load and validate a Stage 2 realization-batch checkpoint at ``path``.

    Args:
        path: Checkpoint path (``realization_batches/batch_NNNN.json``).
        expected_spec_fingerprint: Fingerprint the current run expects.
        expected_source_run_id: Run identifier the current run expects.
        expected_n_queries: Total queries the current run expects.
        expected_batch_size: Batch size the current run expects.
        expected_candidate_ids: Candidate IDs of this Stage 2 batch's blueprints,
            taken from the frozen ``selected_blueprints`` chunk.

    Returns:
        The validated artifact, or ``None`` when the file is absent or its
        header does not match the current run (spec fingerprint, running
        pragmata version, source run id, n_queries, batch_size, or
        candidate_ids). The candidate_ids check is what prevents a stale
        checkpoint from realizing a different blueprint after the frozen
        Stage 1 result changed.

    Raises:
        json.JSONDecodeError: The file content is not valid JSON.
        pydantic.ValidationError: The JSON does not match the artifact schema.

    The api layer catches these and treats the checkpoint as drifted.
    """
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = RealizationBatchArtifact.model_validate(payload)

    if (
        artifact.spec_fingerprint != expected_spec_fingerprint
        or artifact.pragmata_version != importlib.metadata.version("pragmata")
        or artifact.source_run_id != expected_source_run_id
        or artifact.n_queries != expected_n_queries
        or artifact.batch_size != expected_batch_size
        or artifact.candidate_ids != expected_candidate_ids
    ):
        return None

    return artifact
