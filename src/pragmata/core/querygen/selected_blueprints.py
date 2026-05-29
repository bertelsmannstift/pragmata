"""Read helper for the frozen Stage 1 result artifact (``selected_blueprints.json``)."""

from pathlib import Path

from pragmata.core.querygen.checkpoint_read import DriftedField, read_checkpoint_artifact
from pragmata.core.schemas.querygen_output import SelectedBlueprintsArtifact


def read_selected_blueprints_artifact(
    *,
    path: Path,
    expected_spec_fingerprint: str,
    expected_source_run_id: str,
    expected_n_queries: int,
    expected_batch_size: int,
    expected_near_duplicate_tolerance: float,
    expected_enable_planning_memory: bool,
    expected_llm_fingerprint: str,
) -> tuple[SelectedBlueprintsArtifact | None, list[DriftedField]]:
    """Load the frozen Stage 1 result at ``path`` and report drift.

    Returns ``(artifact, [])`` when the frozen result is reusable, ``(None, [])``
    when the file is absent, and ``(None, drifted)`` when its header differs
    from the current run (spec fingerprint, running pragmata version, LLM-config
    fingerprint, source run id, n_queries, batch_size, near_duplicate_tolerance,
    enable_planning_memory). ``embedding_model`` is recorded for provenance but
    intentionally NOT compared. Raises ``json.JSONDecodeError`` /
    ``UnicodeDecodeError`` / ``pydantic.ValidationError`` on a malformed or torn
    file (self-heal case). See :func:`read_checkpoint_artifact`.
    """
    return read_checkpoint_artifact(
        path=path,
        model_cls=SelectedBlueprintsArtifact,
        expected_header={
            "spec_fingerprint": expected_spec_fingerprint,
            "source_run_id": expected_source_run_id,
            "n_queries": expected_n_queries,
            "batch_size": expected_batch_size,
            "near_duplicate_tolerance": expected_near_duplicate_tolerance,
            "enable_planning_memory": expected_enable_planning_memory,
            "llm_fingerprint": expected_llm_fingerprint,
        },
    )
