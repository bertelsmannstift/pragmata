"""Read helper for Stage 1 planning-batch checkpoint artifacts."""

from pathlib import Path

from pragmata.core.querygen.checkpoint_read import read_checkpoint_artifact
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
    expected_llm_fingerprint: str,
) -> tuple[PlanningBatchArtifact | None, list[str]]:
    """Load a Stage 1 planning-batch checkpoint at ``path`` and report drift.

    Returns ``(artifact, [])`` when the checkpoint is reusable, ``(None, [])``
    when the file is absent, and ``(None, drifted)`` when its header differs
    from the current run (spec fingerprint, running pragmata version, LLM-config
    fingerprint, source run id, n_queries, batch_size, enable_planning_memory,
    or candidate_ids). Raises ``json.JSONDecodeError`` / ``UnicodeDecodeError``
    / ``pydantic.ValidationError`` on a malformed or torn file (self-heal case).
    See :func:`read_checkpoint_artifact`.
    """
    return read_checkpoint_artifact(
        path=path,
        model_cls=PlanningBatchArtifact,
        expected_header={
            "spec_fingerprint": expected_spec_fingerprint,
            "source_run_id": expected_source_run_id,
            "n_queries": expected_n_queries,
            "batch_size": expected_batch_size,
            "enable_planning_memory": expected_enable_planning_memory,
            "llm_fingerprint": expected_llm_fingerprint,
            "candidate_ids": expected_candidate_ids,
        },
    )
