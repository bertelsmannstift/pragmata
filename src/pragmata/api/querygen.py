"""API orchestration for the synthetic query generation workflow."""

import logging
from itertools import islice
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, PositiveInt

from pragmata.api._error_log import error_log
from pragmata.core.paths.paths import WorkspacePaths
from pragmata.core.paths.querygen_paths import QueryGenRunPaths, resolve_querygen_paths
from pragmata.core.querygen.assembly import assemble_planning_summary, assemble_queries_meta, assemble_query_rows
from pragmata.core.querygen.batching import build_candidate_ids, chunk_blueprints, iter_batch_sizes
from pragmata.core.querygen.deduplication import deduplicate_blueprints
from pragmata.core.querygen.export import export_planning_summary, export_queries
from pragmata.core.querygen.filtering import filter_aligned_candidate_ids
from pragmata.core.querygen.planning import run_planning_stage
from pragmata.core.querygen.planning_summary import (
    fingerprint_querygen_spec,
    read_planning_summary_artifact,
    run_planning_summary,
)
from pragmata.core.querygen.realization import run_realization_stage
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
    model_kwargs: dict[str, Any] | Unset = UNSET,
    batch_size: PositiveInt | Unset = UNSET,
    near_duplicate_tolerance: float | Unset = UNSET,
    enable_planning_memory: bool | Unset = UNSET,
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
        model_kwargs: Additional provider-specific keyword arguments passed
            through to the underlying chat model.

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
                "model_kwargs": model_kwargs,
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

    paths = resolve_querygen_paths(
        workspace=WorkspacePaths.from_base_dir(settings.base_dir),
        run_id=settings.run_id,
        spec_fingerprint=fingerprint_querygen_spec(settings.spec),
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

        # Stage 1: planning
        candidate_ids = build_candidate_ids(settings.n_queries)
        candidate_id_iter = iter(candidate_ids)
        planning_outputs: list[QueryBlueprint] = []

        for current_batch_size in iter_batch_sizes(
            n_queries=settings.n_queries,
            batch_size=settings.batch_size,
        ):
            batch_candidate_ids = list(islice(candidate_id_iter, current_batch_size))
            batch_blueprints = run_planning_stage(
                spec=settings.spec,
                llm_settings=settings.llm,
                api_key=api_key,
                batch_candidate_ids=batch_candidate_ids,
                planning_summary=planning_summary_state,
            )
            planning_outputs.extend(batch_blueprints)

            if settings.enable_planning_memory:
                planning_summary_state = run_planning_summary(
                    spec=settings.spec,
                    candidates=batch_blueprints,
                    llm_settings=settings.llm,
                    api_key=api_key,
                    prior_summary_state=planning_summary_state,
                )

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

        # Stage 2: realization
        realization_outputs: list[RealizedQuery] = []

        for blueprint_batch in chunk_blueprints(
            blueprints=selected_blueprints,
            chunk_size=settings.batch_size,
        ):
            realization_outputs.extend(
                run_realization_stage(
                    candidates=blueprint_batch,
                    llm_settings=settings.llm,
                    api_key=api_key,
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

        if settings.enable_planning_memory and planning_summary_state is not None:
            planning_summary_artifact = assemble_planning_summary(
                spec=settings.spec,
                run_id=settings.run_id,
                state=planning_summary_state,
            )
            export_planning_summary(
                artifact=planning_summary_artifact,
                artifact_path=paths.planning_summary_artifact_json,
            )

    logger.info(
        "Query generation run %s complete (%d returned queries)",
        settings.run_id,
        len(rows),
    )

    return QueryGenRunResult(settings=settings, paths=paths)
