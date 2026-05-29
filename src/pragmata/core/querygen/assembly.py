"""Assembly of synthetic query generation output."""

import importlib.metadata
from datetime import UTC, datetime

from pragmata.core.querygen.planning_summary import fingerprint_querygen_spec
from pragmata.core.schemas.querygen_input import QueryGenSpec
from pragmata.core.schemas.querygen_output import (
    PlanningBatchArtifact,
    PlanningSummaryArtifact,
    SelectedBlueprintsArtifact,
    SyntheticQueriesMeta,
    SyntheticQueryRow,
)
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery
from pragmata.core.schemas.querygen_summary import PlanningSummaryState


def _build_query_ids(
    run_id: str,
    n_queries: int,
) -> list[str]:
    """Build deterministic final query IDs for assembled synthetic queries.

    Args:
        run_id: Stable run identifier for the current synthetic-query run.
        n_queries: Number of final query IDs to generate.

    Returns:
        Deterministically ordered final query IDs.
    """
    return [f"{run_id}_q{index}" for index in range(1, n_queries + 1)]


def assemble_query_rows(
    blueprints: list[QueryBlueprint],
    realized_queries: list[RealizedQuery],
    run_id: str,
) -> list[SyntheticQueryRow]:
    """Assemble final synthetic query rows from stage-1 blueprints and stage-2 realized queries.

    Args:
        blueprints: Selected post-deduplication stage-1 blueprints.
        realized_queries: Filtered stage-2 realized queries.
        run_id: Stable run identifier used for final query ID generation.

    Returns:
        Final assembled synthetic query rows, ordered by ``realized_queries``.
    """
    blueprint_by_candidate_id = {blueprint.candidate_id: blueprint for blueprint in blueprints}
    query_ids = _build_query_ids(run_id=run_id, n_queries=len(realized_queries))

    return [
        SyntheticQueryRow(
            query_id=query_id,
            query=realized_query.query,
            domain=blueprint_by_candidate_id[realized_query.candidate_id].domain,
            role=blueprint_by_candidate_id[realized_query.candidate_id].role,
            language=blueprint_by_candidate_id[realized_query.candidate_id].language,
            topic=blueprint_by_candidate_id[realized_query.candidate_id].topic,
            intent=blueprint_by_candidate_id[realized_query.candidate_id].intent,
            task=blueprint_by_candidate_id[realized_query.candidate_id].task,
            difficulty=blueprint_by_candidate_id[realized_query.candidate_id].difficulty,
            format=blueprint_by_candidate_id[realized_query.candidate_id].format,
        )
        for query_id, realized_query in zip(query_ids, realized_queries, strict=True)
    ]


def assemble_queries_meta(
    run_id: str,
    n_requested_queries: int,
    n_returned_queries: int,
    model_provider: str,
    planning_model: str,
    realization_model: str,
) -> SyntheticQueriesMeta:
    """Assemble dataset-level metadata for a synthetic-query run.

    Args:
        run_id: Stable run identifier.
        n_requested_queries: Number of queries requested for the run.
        n_returned_queries: Number of final returned queries.
        model_provider: Configured chat-model provider.
        planning_model: Model identifier used for stage 1 planning.
        realization_model: Model identifier used for stage 2 realization.

    Returns:
        Dataset-level metadata with internally stamped creation time.
    """
    return SyntheticQueriesMeta(
        run_id=run_id,
        created_at=datetime.now(UTC),
        n_requested_queries=n_requested_queries,
        n_returned_queries=n_returned_queries,
        model_provider=model_provider,
        planning_model=planning_model,
        realization_model=realization_model,
    )


def assemble_planning_batch_artifact(
    *,
    spec_fingerprint: str,
    llm_fingerprint: str,
    source_run_id: str,
    n_queries: int,
    batch_size: int,
    batch_idx: int,
    candidate_ids: list[str],
    blueprints: list[QueryBlueprint],
    planning_summary_state: PlanningSummaryState | None,
    enable_planning_memory: bool,
) -> PlanningBatchArtifact:
    """Assemble a Stage 1 planning-batch artifact, stamping version + time.

    Args:
        spec_fingerprint: Fingerprint of the resolved querygen spec.
        llm_fingerprint: Fingerprint of the output-shaping LLM settings; a
            change invalidates the checkpoint.
        source_run_id: Run identifier that produced this batch.
        n_queries: Total queries requested for the run.
        batch_size: Configured batch size for the run.
        batch_idx: Zero-based index of this planning batch.
        candidate_ids: Candidate IDs assigned to this batch.
        blueprints: Blueprints produced for this batch.
        planning_summary_state: Planning-summary state after this batch, if any.
        enable_planning_memory: Whether planning memory was enabled (a change
            invalidates the checkpoint, since it shapes the blueprints).

    Returns:
        A validated ``PlanningBatchArtifact`` with internally stamped
        ``pragmata_version`` and ``created_at``.
    """
    return PlanningBatchArtifact(
        spec_fingerprint=spec_fingerprint,
        pragmata_version=importlib.metadata.version("pragmata"),
        llm_fingerprint=llm_fingerprint,
        source_run_id=source_run_id,
        n_queries=n_queries,
        batch_size=batch_size,
        batch_idx=batch_idx,
        enable_planning_memory=enable_planning_memory,
        candidate_ids=candidate_ids,
        blueprints=blueprints,
        planning_summary_state=planning_summary_state,
        created_at=datetime.now(UTC),
    )


def assemble_selected_blueprints_artifact(
    *,
    spec_fingerprint: str,
    llm_fingerprint: str,
    source_run_id: str,
    n_queries: int,
    batch_size: int,
    near_duplicate_tolerance: float,
    enable_planning_memory: bool,
    embedding_model: str,
    blueprints: list[QueryBlueprint],
) -> SelectedBlueprintsArtifact:
    """Assemble the frozen Stage 1 result artifact, stamping provenance.

    Args:
        spec_fingerprint: Fingerprint of the resolved querygen spec.
        llm_fingerprint: Fingerprint of the output-shaping LLM settings; a
            change invalidates the checkpoint.
        source_run_id: Run identifier that produced this result.
        n_queries: Total queries requested for the run.
        batch_size: Configured batch size for the run.
        near_duplicate_tolerance: Tolerance used for the deduplication that
            produced ``blueprints``.
        enable_planning_memory: Whether planning memory was enabled (a change
            invalidates the frozen result, since it shapes the blueprints).
        embedding_model: Embedding model checkpoint used for deduplication
            (recorded for provenance).
        blueprints: Final post-deduplication selected blueprints.

    Returns:
        A validated ``SelectedBlueprintsArtifact`` with internally stamped
        ``pragmata_version`` and ``created_at``.
    """
    return SelectedBlueprintsArtifact(
        spec_fingerprint=spec_fingerprint,
        pragmata_version=importlib.metadata.version("pragmata"),
        llm_fingerprint=llm_fingerprint,
        source_run_id=source_run_id,
        n_queries=n_queries,
        batch_size=batch_size,
        near_duplicate_tolerance=near_duplicate_tolerance,
        enable_planning_memory=enable_planning_memory,
        embedding_model=embedding_model,
        blueprints=blueprints,
        created_at=datetime.now(UTC),
    )


def assemble_planning_summary(
    spec: QueryGenSpec,
    run_id: str,
    state: PlanningSummaryState,
) -> PlanningSummaryArtifact:
    """Assemble a planning-summary artifact from the final summary state.

    Args:
        spec: Resolved query-generation specification for the run.
        run_id: Unique run identifier used as the source for the artifact.
        state: Final run-level planning-summary produced by the summary-updater stage.

    Returns:
        A validated ``PlanningSummaryArtifact`` with internally generated
        ``spec_fingerprint`` and ``created_at`` metadata.
    """
    return PlanningSummaryArtifact(
        spec_fingerprint=fingerprint_querygen_spec(spec),
        source_run_id=run_id,
        created_at=datetime.now(UTC),
        state=state,
    )
