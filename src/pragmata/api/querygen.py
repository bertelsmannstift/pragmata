"""API orchestration for the synthetic query generation workflow."""

import json
import logging
from itertools import islice
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from pragmata.api._error_log import error_log
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.paths.querygen_paths import QueryGenRunPaths, resolve_querygen_paths
from pragmata.core.querygen.assembly import (
    assemble_planning_batch_artifact,
    assemble_planning_summary,
    assemble_queries_meta,
    assemble_query_rows,
    assemble_realization_batch_artifact,
    assemble_selected_blueprints_artifact,
)
from pragmata.core.querygen.batching import build_candidate_ids, chunk_blueprints, iter_batch_sizes
from pragmata.core.querygen.deduplication import deduplicate_blueprints
from pragmata.core.querygen.export import (
    export_planning_batch_artifact,
    export_planning_summary,
    export_queries,
    export_realization_batch_artifact,
    export_selected_blueprints,
)
from pragmata.core.querygen.filtering import filter_aligned_candidate_ids
from pragmata.core.querygen.planning import run_planning_stage
from pragmata.core.querygen.planning_batches import read_planning_batch_artifact
from pragmata.core.querygen.planning_summary import (
    fingerprint_querygen_spec,
    read_planning_summary_artifact,
    run_planning_summary,
)
from pragmata.core.querygen.realization import run_realization_stage
from pragmata.core.querygen.realization_batches import read_realization_batch_artifact
from pragmata.core.querygen.selected_blueprints import read_selected_blueprints_artifact
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery
from pragmata.core.schemas.querygen_summary import PlanningSummaryState
from pragmata.core.settings.querygen_settings import QueryGenRunSettings
from pragmata.core.settings.settings_base import UNSET, Unset, load_config_file, resolve_api_key

logger = logging.getLogger(__name__)


class QueryGenRunResult(BaseModel):
    """Returned run descriptor for a synthetic query generation run.

    Attributes:
        settings: Fully resolved run settings used for the run.
        paths: Filesystem paths to the exported run artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    settings: QueryGenRunSettings
    paths: QueryGenRunPaths


def _resume_or_run_planning_batch(
    *,
    paths: QueryGenRunPaths,
    settings: QueryGenRunSettings,
    spec_fingerprint: str,
    api_key: str,
    batch_idx: int,
    total_batches: int,
    batch_candidate_ids: list[str],
    planning_summary_state: PlanningSummaryState | None,
    resume_exhausted: bool,
) -> tuple[list[QueryBlueprint], PlanningSummaryState | None, bool]:
    """Resume one Stage 1 batch from a valid checkpoint, or run it fresh and persist.

    Returns ``(batch_blueprints, planning_summary_state, resume_exhausted)``.
    ``resume_exhausted`` sticks True once any batch is recomputed: the
    planning-summary chain then diverges from what later checkpoints recorded,
    so every subsequent batch must also run fresh. This also keeps resume to a
    contiguous prefix, so a missing or drifted checkpoint never causes a later
    batch's stale state to be threaded forward.

    Args:
        paths: Resolved run paths (provides ``planning_batches_dir``).
        settings: Resolved run settings.
        spec_fingerprint: Fingerprint of the resolved spec for this run.
        api_key: Provider API key.
        batch_idx: Zero-based index of this planning batch.
        total_batches: Total number of planning batches (for logging).
        batch_candidate_ids: Candidate IDs assigned to this batch.
        planning_summary_state: Planning-summary state seeding this batch.
        resume_exhausted: Whether an earlier batch already ran fresh this run.
    """
    checkpoint_path = paths.planning_batches_dir / f"batch_{batch_idx:04d}.json"

    if not resume_exhausted:
        try:
            artifact = read_planning_batch_artifact(
                path=checkpoint_path,
                expected_spec_fingerprint=spec_fingerprint,
                expected_source_run_id=settings.run_id,
                expected_n_queries=settings.n_queries,
                expected_batch_size=settings.batch_size,
                expected_candidate_ids=batch_candidate_ids,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Stage 1 batch %d/%d checkpoint %s unreadable (%s); rerunning it and all later batches",
                batch_idx + 1,
                total_batches,
                checkpoint_path.name,
                exc,
            )
            artifact = None
        if artifact is not None:
            logger.info(
                "Stage 1 batch %d/%d resumed from checkpoint for run %s",
                batch_idx + 1,
                total_batches,
                settings.run_id,
            )
            return artifact.blueprints, artifact.planning_summary_state, False

    batch_blueprints = run_planning_stage(
        spec=settings.spec,
        llm_settings=settings.llm,
        api_key=api_key,
        batch_candidate_ids=batch_candidate_ids,
        planning_summary=planning_summary_state,
    )

    if settings.enable_planning_memory:
        planning_summary_state = run_planning_summary(
            spec=settings.spec,
            candidates=batch_blueprints,
            llm_settings=settings.llm,
            api_key=api_key,
            prior_summary_state=planning_summary_state,
        )

    export_planning_batch_artifact(
        artifact=assemble_planning_batch_artifact(
            spec_fingerprint=spec_fingerprint,
            source_run_id=settings.run_id,
            n_queries=settings.n_queries,
            batch_size=settings.batch_size,
            batch_idx=batch_idx,
            candidate_ids=batch_candidate_ids,
            blueprints=batch_blueprints,
            planning_summary_state=planning_summary_state,
        ),
        path=checkpoint_path,
    )
    logger.info(
        "Stage 1 batch %d/%d complete for run %s (%d blueprints)",
        batch_idx + 1,
        total_batches,
        settings.run_id,
        len(batch_blueprints),
    )
    return batch_blueprints, planning_summary_state, True


def _resume_or_run_realization_batch(
    *,
    paths: QueryGenRunPaths,
    settings: QueryGenRunSettings,
    spec_fingerprint: str,
    api_key: str,
    batch_idx: int,
    total_batches: int,
    blueprint_batch: list[QueryBlueprint],
    fresh: bool,
) -> list[RealizedQuery]:
    """Resume one Stage 2 batch from a valid checkpoint, or run it fresh and persist.

    Stage 2 batches are independent (no chained state), so each batch resumes on
    its own merits -- a missing or drifted batch does not force later batches to
    re-run. The ``candidate_ids`` check on read guarantees a checkpoint is only
    reused for the exact blueprints in this chunk of the frozen result.

    Args:
        paths: Resolved run paths (provides ``realization_batches_dir``).
        settings: Resolved run settings.
        spec_fingerprint: Fingerprint of the resolved spec for this run.
        api_key: Provider API key.
        batch_idx: Zero-based index of this realization batch.
        total_batches: Total number of realization batches (for logging).
        blueprint_batch: Selected blueprints to realize in this batch.
        fresh: When True, ignore any existing checkpoint and recompute.
    """
    checkpoint_path = paths.realization_batches_dir / f"batch_{batch_idx:04d}.json"
    batch_candidate_ids = [blueprint.candidate_id for blueprint in blueprint_batch]

    if not fresh:
        try:
            artifact = read_realization_batch_artifact(
                path=checkpoint_path,
                expected_spec_fingerprint=spec_fingerprint,
                expected_source_run_id=settings.run_id,
                expected_n_queries=settings.n_queries,
                expected_batch_size=settings.batch_size,
                expected_candidate_ids=batch_candidate_ids,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Stage 2 batch %d/%d checkpoint %s unreadable (%s); rerunning it",
                batch_idx + 1,
                total_batches,
                checkpoint_path.name,
                exc,
            )
            artifact = None
        if artifact is not None:
            logger.info(
                "Stage 2 batch %d/%d resumed from checkpoint for run %s",
                batch_idx + 1,
                total_batches,
                settings.run_id,
            )
            return artifact.queries

    realized_queries = run_realization_stage(
        candidates=blueprint_batch,
        llm_settings=settings.llm,
        api_key=api_key,
    )

    export_realization_batch_artifact(
        artifact=assemble_realization_batch_artifact(
            spec_fingerprint=spec_fingerprint,
            source_run_id=settings.run_id,
            n_queries=settings.n_queries,
            batch_size=settings.batch_size,
            batch_idx=batch_idx,
            candidate_ids=batch_candidate_ids,
            queries=realized_queries,
        ),
        path=checkpoint_path,
    )
    logger.info(
        "Stage 2 batch %d/%d complete for run %s (%d realized)",
        batch_idx + 1,
        total_batches,
        settings.run_id,
        len(realized_queries),
    )
    return realized_queries


def gen_queries(
    *,
    domains: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    roles: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    languages: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    topics: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    intents: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    tasks: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    disallowed_topics: list[str] | Unset = UNSET,
    difficulty: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    formats: str | list[str] | list[dict[str, object]] | Unset = UNSET,
    base_dir: str | Path | Unset = UNSET,
    config_path: str | Path | Unset = UNSET,
    n_queries: PositiveInt | Unset = UNSET,
    run_id: str | Unset = UNSET,
    model_provider: str | Unset = UNSET,
    planning_model: str | Unset = UNSET,
    realization_model: str | Unset = UNSET,
    requests_per_second: float | Unset = UNSET,
    check_every_n_seconds: float | Unset = UNSET,
    max_bucket_size: int | Unset = UNSET,
    base_url: str | Unset = UNSET,
    planning_model_kwargs: dict[str, Any] | Unset = UNSET,
    realization_model_kwargs: dict[str, Any] | Unset = UNSET,
    batch_size: PositiveInt | Unset = UNSET,
    near_duplicate_tolerance: float | Unset = UNSET,
    enable_planning_memory: bool | Unset = UNSET,
    fresh: bool = False,
) -> QueryGenRunResult:
    """Generate synthetic chatbot queries from a query-generation specification.

    This function resolves the effective runtime configuration, validates provider
    credentials, prepares filesystem paths for the run, executes the staged query-generation
    workflow, and exports the generated query artifacts to disk.

    Args:
        domains: Domain choices for the generated queries. The domain is the
            setting or subject area in which a query arises.
        roles: Role choices for the generated queries. The role is the user
            persona or perspective from which a query is asked.
        languages: Language choices for the generated queries. The language is
            the language in which the realized query should be written.
        topics: Topic choices for the generated queries. The topic is the
            concrete subject matter the query concerns.
        intents: Intent choices for the generated queries. The intent is the
            underlying user goal or motivation behind the request.
        tasks: Task choices for the generated queries. The task is the type of
            information-processing task the user wants performed.
        disallowed_topics: Optional topics that must not appear in generated queries.
        difficulty: Optional difficulty choices. This reflects the expected
            complexity level of the request.
        formats: Optional format choices. This reflects the expected answer format
            implied by the request.
        base_dir: Workspace base directory for run artifacts. Defaults to the
            current working directory.
        config_path: Path to a YAML configuration file.
        n_queries: Number of queries to prepare. Defaults to 50.
        batch_size: Number of queries to generate per LLM call. Larger
            values use fewer, bigger calls; smaller values split
            generation into more repeated calls. Defaults to 25.
        near_duplicate_tolerance: Similarity tolerance used for semantic
            near-duplicate blueprint removal. Must be in the range (0, 1],
            where lower values deduplicate more aggressively and higher values
            allow more similar blueprints, including close paraphrases, to remain.
            Defaults to 0.95.
        enable_planning_memory: Whether to enable planning memory for the run.
            Defaults to True. When enabled, an additional LLM updates and persists a
            compact summary of prior blueprint generation across batches and compatible
            runs to reduce redundant or near-duplicate queries and improve diversity.
        fresh: When True, ignore any existing on-disk checkpoints and frozen
            result for this run and recompute from scratch (artifacts are still
            written). Defaults to False, i.e. resume from whatever is present.
        run_id: Explicit run identifier. Defaults to an auto-generated UUID
            hex string.
        model_provider: Chat model provider to use. Defaults to "mistralai".
            Requires a corresponding API key via environment variables.
            (e.g., MISTRAL_API_KEY, OPENAI_API_KEY, etc.)
        planning_model: Model identifier for the planning stage. Defaults to
            "magistral-medium-latest".
        realization_model: Model identifier for the realization stage. Defaults
            to "mistral-medium-latest".
        requests_per_second: Maximum number of llm requests per second allowed by the
            in-memory rate limiter. Defaults to 1.0.
        check_every_n_seconds: Interval in seconds at which the llm rate limiter checks
            for available capacity. Defaults to 1.0.
        max_bucket_size: Maximum burst size for the llm rate limiter. Defaults to 1.
        base_url: Optional custom API endpoint for the provider (e.g., Azure
            OpenAI deployments).
        planning_model_kwargs: Additional provider-specific keyword arguments passed
            through to the planning-stage chat model. Also used by the planning-summary
            updater because it invokes the configured planning model.
        realization_model_kwargs: Additional provider-specific keyword arguments passed
            through to the realization-stage chat model.

    Returns:
        QueryGenRunResult: Run descriptor containing the resolved settings used
        for the run and the filesystem paths to the exported artifacts.
    """
    settings = QueryGenRunSettings.resolve(
        config=load_config_file(config_path) if isinstance(config_path, (str, Path)) else None,
        env=None,  # Environment-derived settings are not wired for querygen yet.
        overrides={
            "spec": {
                "domain_context": {
                    "domains": domains,
                    "roles": roles,
                    "languages": languages,
                },
                "knowledge_scope": {
                    "topics": topics,
                },
                "scenario": {
                    "intents": intents,
                    "tasks": tasks,
                    "difficulty": difficulty,
                },
                "format_requests": {
                    "formats": formats,
                },
                "safety": {
                    "disallowed_topics": disallowed_topics,
                },
            },
            "llm": {
                "model_provider": model_provider,
                "planning_model": planning_model,
                "realization_model": realization_model,
                "requests_per_second": requests_per_second,
                "check_every_n_seconds": check_every_n_seconds,
                "max_bucket_size": max_bucket_size,
                "base_url": base_url,
                "planning_model_kwargs": planning_model_kwargs,
                "realization_model_kwargs": realization_model_kwargs,
            },
            "base_dir": base_dir,
            "run_id": run_id,
            "n_queries": n_queries,
            "batch_size": batch_size,
            "near_duplicate_tolerance": near_duplicate_tolerance,
            "enable_planning_memory": enable_planning_memory,
        },
    )

    api_key = resolve_api_key(settings.llm.model_provider)

    spec_fingerprint = fingerprint_querygen_spec(settings.spec)
    paths = resolve_querygen_paths(
        workspace=WorkspacePaths.from_base_dir(settings.base_dir),
        run_id=settings.run_id,
        spec_fingerprint=spec_fingerprint,
    ).ensure_dirs()

    logger.info(
        "Starting query generation run %s (n_queries=%d, batch_size=%d)",
        settings.run_id,
        settings.n_queries,
        settings.batch_size,
    )

    with error_log(paths.run_dir):
        planning_summary_state: PlanningSummaryState | None = None

        if settings.enable_planning_memory:
            planning_summary_artifact = read_planning_summary_artifact(
                artifact_path=paths.planning_summary_artifact_json,
                spec=settings.spec,
            )
            if planning_summary_artifact is not None:
                planning_summary_state = planning_summary_artifact.state

        # Stage 1: planning. Resume from the frozen result if present (skips the
        # planning loop, deduplication, and embedding-model load); otherwise run
        # the per-batch loop, deduplicate, persist cross-run memory, and freeze
        # the result.
        frozen_result = (
            None
            if fresh
            else read_selected_blueprints_artifact(
                path=paths.selected_blueprints_json,
                expected_spec_fingerprint=spec_fingerprint,
                expected_source_run_id=settings.run_id,
                expected_n_queries=settings.n_queries,
                expected_batch_size=settings.batch_size,
                expected_near_duplicate_tolerance=settings.near_duplicate_tolerance,
            )
        )

        if frozen_result is not None:
            selected_blueprints = frozen_result.blueprints
            logger.info(
                "Stage 1 result loaded from frozen artifact for run %s (%d selected); "
                "skipping planning and deduplication",
                settings.run_id,
                len(selected_blueprints),
            )
        else:
            candidate_ids = build_candidate_ids(settings.n_queries)
            candidate_id_iter = iter(candidate_ids)
            batch_candidate_id_lists = [
                list(islice(candidate_id_iter, current_batch_size))
                for current_batch_size in iter_batch_sizes(
                    n_queries=settings.n_queries,
                    batch_size=settings.batch_size,
                )
            ]
            total_batches = len(batch_candidate_id_lists)

            planning_outputs: list[QueryBlueprint] = []
            resume_exhausted = fresh
            for batch_idx, batch_candidate_ids in enumerate(batch_candidate_id_lists):
                batch_blueprints, planning_summary_state, resume_exhausted = _resume_or_run_planning_batch(
                    paths=paths,
                    settings=settings,
                    spec_fingerprint=spec_fingerprint,
                    api_key=api_key,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    batch_candidate_ids=batch_candidate_ids,
                    planning_summary_state=planning_summary_state,
                    resume_exhausted=resume_exhausted,
                )
                planning_outputs.extend(batch_blueprints)

            filtered_planning_outputs = filter_aligned_candidate_ids(
                items=planning_outputs,
                expected_candidate_ids=candidate_ids,
            )

            selected_blueprints = deduplicate_blueprints(
                filtered_planning_outputs,
                near_duplicate_tolerance=settings.near_duplicate_tolerance,
            )

            logger.info(
                "Stage 1 (query planning) complete for run %s (%d planned -> %d selected)",
                settings.run_id,
                len(planning_outputs),
                len(selected_blueprints),
            )

            # Persist cross-run planning memory now (end of Stage 1) so a later
            # Stage 2 crash still leaves it available to seed future runs.
            if settings.enable_planning_memory and planning_summary_state is not None:
                export_planning_summary(
                    artifact=assemble_planning_summary(
                        spec=settings.spec,
                        run_id=settings.run_id,
                        state=planning_summary_state,
                    ),
                    artifact_path=paths.planning_summary_artifact_json,
                )

            # Freeze the Stage 1 result last: its presence is the "Stage 1 done"
            # marker that lets a rerun skip everything above.
            export_selected_blueprints(
                artifact=assemble_selected_blueprints_artifact(
                    spec_fingerprint=spec_fingerprint,
                    source_run_id=settings.run_id,
                    n_queries=settings.n_queries,
                    batch_size=settings.batch_size,
                    near_duplicate_tolerance=settings.near_duplicate_tolerance,
                    blueprints=selected_blueprints,
                ),
                path=paths.selected_blueprints_json,
            )

        # Stage 2: realization (with per-batch checkpointing). Batches are
        # independent, so each resumes on its own; the final CSV is a
        # deterministic projection of the frozen result + these outputs.
        realization_batches = list(
            chunk_blueprints(
                blueprints=selected_blueprints,
                chunk_size=settings.batch_size,
            )
        )
        total_realization_batches = len(realization_batches)

        realization_outputs: list[RealizedQuery] = []
        for batch_idx, blueprint_batch in enumerate(realization_batches):
            realization_outputs.extend(
                _resume_or_run_realization_batch(
                    paths=paths,
                    settings=settings,
                    spec_fingerprint=spec_fingerprint,
                    api_key=api_key,
                    batch_idx=batch_idx,
                    total_batches=total_realization_batches,
                    blueprint_batch=blueprint_batch,
                    fresh=fresh,
                )
            )

        filtered_realization_outputs = filter_aligned_candidate_ids(
            items=realization_outputs,
            expected_candidate_ids=[blueprint.candidate_id for blueprint in selected_blueprints],
        )

        logger.info(
            "Stage 2 (query realization) complete for run %s (%d realized -> %d selected)",
            settings.run_id,
            len(realization_outputs),
            len(filtered_realization_outputs),
        )

        # The final CSV is a positional projection of the realized set. Warn (do
        # not fail -- short results are tolerated) if realization is incomplete,
        # since then row count and query_ids differ from the frozen blueprint set.
        if len(filtered_realization_outputs) < len(selected_blueprints):
            logger.warning(
                "Stage 2 for run %s realized %d of %d selected blueprints; the CSV "
                "reflects only realized queries and positional query_ids may shift",
                settings.run_id,
                len(filtered_realization_outputs),
                len(selected_blueprints),
            )

        # Assembly and export:
        rows = assemble_query_rows(
            blueprints=selected_blueprints,
            realized_queries=filtered_realization_outputs,
            run_id=settings.run_id,
        )

        meta = assemble_queries_meta(
            run_id=settings.run_id,
            n_requested_queries=settings.n_queries,
            n_returned_queries=len(rows),
            model_provider=settings.llm.model_provider,
            planning_model=settings.llm.planning_model,
            realization_model=settings.llm.realization_model,
        )

        export_queries(
            rows=rows,
            meta=meta,
            queries_path=paths.synthetic_queries_csv,
            meta_path=paths.synthetic_queries_meta_json,
        )

    logger.info(
        "Query generation run %s complete (%d returned queries)",
        settings.run_id,
        len(rows),
    )

    return QueryGenRunResult(settings=settings, paths=paths)
