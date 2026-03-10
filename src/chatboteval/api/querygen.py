"""API orchestration for the synthetic query generation workflow."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from chatboteval.core.paths.paths import WorkspacePaths
from chatboteval.core.paths.querygen_paths import QueryGenRunPaths, resolve_querygen_paths
from chatboteval.core.settings.querygen_settings import QueryGenRunSettings
from chatboteval.core.settings.settings_base import UNSET, load_config_file, resolve_provider_api_key


class QueryGenRunResult(BaseModel):
    """Prepared invocation state for a synthetic query generation run.

    Attributes:
        settings: Fully resolved run settings.
        paths: Filesystem paths for run artifacts.
    """

    settings: QueryGenRunSettings
    paths: QueryGenRunPaths


def gen_queries(
    *,
    domains: str | list[str] | list[dict[str, object]],
    roles: str | list[str] | list[dict[str, object]],
    languages: str | list[str] | list[dict[str, object]],
    topics: str | list[str] | list[dict[str, object]],
    intents: str | list[str] | list[dict[str, object]],
    tasks: str | list[str] | list[dict[str, object]],
    disallowed_topics: str | list[str] | list[dict[str, object]] | object = UNSET,
    difficulty: str | list[str] | list[dict[str, object]] | object = UNSET,
    formats: str | list[str] | list[dict[str, object]] | object = UNSET,
    base_dir: str | Path | object = UNSET,
    config_path: str | Path | object = UNSET,
    n_queries: int | object = UNSET,
    run_id: str | object = UNSET,
    model_provider: str | object = UNSET,
    planning_model: str | object = UNSET,
    realization_model: str | object = UNSET,
    base_url: str | object = UNSET,
    model_kwargs: dict[str, Any] | object = UNSET,
) -> QueryGenRunResult:
    """Prepare a synthetic query generation run.

    This function resolves the effective runtime configuration,
    validates provider credentials, and prepares filesystem paths
    for the run. It does not yet execute the query generation workflow.

    Args:
        domains: Domain distribution specification.
        roles: Role distribution specification.
        languages: Language distribution specification.
        topics: Topic distribution specification.
        intents: Intent distribution specification.
        tasks: Task distribution specification.
        disallowed_topics: Optional topics excluded from generation.
        difficulty: Optional difficulty distribution specification.
        formats: Optional response format distribution specification.
        base_dir: Optional workspace base directory.
        config_path: Optional configuration file path.
        n_queries: Optional number of queries to generate.
        run_id: Optional run identifier.
        model_provider: Optional LLM provider name.
        planning_model: Optional planning model identifier.
        realization_model: Optional realization model identifier.
        base_url: Optional provider base URL.
        model_kwargs: Optional additional model configuration.

    Returns:
        QueryGenRunResult: Prepared run state containing resolved settings
        and filesystem paths.
    """
    settings = QueryGenRunSettings.resolve(
        config=load_config_file(config_path) if config_path is not UNSET else None,
        env=os.environ,
        overrides={
            "spec": {
                "domains": domains,
                "roles": roles,
                "languages": languages,
                "topics": topics,
                "intents": intents,
                "tasks": tasks,
                "disallowed_topics": disallowed_topics,
                "difficulty": difficulty,
                "formats": formats,
            },
            "llm": {
                "model_provider": model_provider,
                "planning_model": planning_model,
                "realization_model": realization_model,
                "base_url": base_url,
                "model_kwargs": model_kwargs,
            },
            "base_dir": base_dir,
            "run_id": run_id,
            "n_queries": n_queries,
        },
    )

    resolve_provider_api_key(settings.llm.model_provider)

    paths = resolve_querygen_paths(
        workspace=WorkspacePaths.from_base_dir(settings.base_dir),
        run_id=settings.run_id,
    ).ensure_dirs()

    return QueryGenRunResult(settings=settings, paths=paths)
