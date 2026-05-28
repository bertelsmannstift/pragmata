"""Read helper for Stage 1 planning-batch checkpoint artifacts."""

import importlib.metadata
import json
from pathlib import Path

from pragmata.core.schemas.querygen_output import PlanningBatchArtifact


def read_planning_batch_artifact(
    *,
    path: Path,
    expected_spec_fingerprint: str,
    expected_source_run_id: str,
    expected_n_queries: int,
    expected_batch_size: int,
    expected_candidate_ids: list[str],
    expected_enable_planning_memory: bool,
) -> PlanningBatchArtifact | None:
    """Load and validate a Stage 1 planning-batch checkpoint at ``path``.

    Args:
        path: Checkpoint path (``planning_batches/batch_NNNN.json``).
        expected_spec_fingerprint: Fingerprint the current run expects.
        expected_source_run_id: Run identifier the current run expects.
        expected_n_queries: Total queries the current run expects.
        expected_batch_size: Batch size the current run expects.
        expected_candidate_ids: Candidate IDs assigned to this batch.
        expected_enable_planning_memory: Planning-memory setting the current run
            expects (it shapes the blueprints, so a change invalidates).

    Returns:
        The validated artifact, or ``None`` when the file is absent or its
        header does not match the current run (spec fingerprint, running
        pragmata version, source run id, n_queries, batch_size,
        enable_planning_memory, or candidate_ids) -- i.e. the checkpoint belongs
        to an incompatible run and must be redone.

    Raises:
        json.JSONDecodeError: The file content is not valid JSON.
        pydantic.ValidationError: The JSON does not match the artifact schema.

    The api layer catches these and treats the checkpoint as drifted.
    """
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = PlanningBatchArtifact.model_validate(payload)

    if (
        artifact.spec_fingerprint != expected_spec_fingerprint
        or artifact.pragmata_version != importlib.metadata.version("pragmata")
        or artifact.source_run_id != expected_source_run_id
        or artifact.n_queries != expected_n_queries
        or artifact.batch_size != expected_batch_size
        or artifact.enable_planning_memory != expected_enable_planning_memory
        or artifact.candidate_ids != expected_candidate_ids
    ):
        return None

    return artifact
