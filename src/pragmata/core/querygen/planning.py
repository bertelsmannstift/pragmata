"""Stage-1 planning executor for synthetic query generation."""

from pragmata.core.querygen.llm import LlmInitializationError, build_llm_runnable
from pragmata.core.querygen.prompts import SYSTEM_PROMPT_PLANNING, USER_PROMPT_PLANNING
from pragmata.core.schemas.querygen_input import QueryGenSpec, WeightedValue
from pragmata.core.schemas.querygen_plan import QueryBlueprint, QueryBlueprintList
from pragmata.core.settings.querygen_settings import LlmSettings

_UNSPECIFIED = "Not specified"


class PlanningStageError(RuntimeError):
    """Raised when a planning-stage invocation fails."""


def _format_weighted_values(values: list[WeightedValue] | None) -> str:
    """Format weighted categorical values for prompt injection.

    Args:
        values: Canonicalized weighted values from the resolved query-generation spec.

    Returns:
        A human-readable string representation suitable for prompt variables.
    """
    if values is None:
        return _UNSPECIFIED

    return ", ".join(f"{item.value} (weight={item.weight:g})" for item in values)


def _format_string_list(values: list[str] | None) -> str:
    """Format string lists for prompt injection.

    Args:
        values: List of strings from the resolved query-generation spec.

    Returns:
        A human-readable string representation suitable for prompt variables.
    """
    if not values:
        return _UNSPECIFIED

    return ", ".join(values)


def _build_planning_prompt_vars(
    spec: QueryGenSpec,
    batch_candidate_ids: list[str],
) -> dict[str, object]:
    """Build invoke-time prompt variables for one planning-stage batch.

    Args:
        spec: Resolved query-generation specification.
        batch_candidate_ids: Candidate IDs assigned to this single planning invocation.

    Returns:
        Prompt variables aligned with the stage-1 planning prompt placeholders.
    """
    if not batch_candidate_ids:
        raise ValueError("batch_candidate_ids must not be empty")

    return {
        "candidate_ids": "\n    - " + "\n    - ".join(batch_candidate_ids),
        "domains": _format_weighted_values(spec.domain_context.domains),
        "roles": _format_weighted_values(spec.domain_context.roles),
        "languages": _format_weighted_values(spec.domain_context.languages),
        "topics": _format_weighted_values(spec.knowledge_scope.topics),
        "intents": _format_weighted_values(spec.scenario.intents),
        "tasks": _format_weighted_values(spec.scenario.tasks),
        "difficulty": _format_weighted_values(spec.scenario.difficulty),
        "formats": _format_weighted_values(spec.format_requests.formats),
        "disallowed_topics": _format_string_list(spec.safety.disallowed_topics),
        "n_queries": len(batch_candidate_ids),
    }


def run_planning_stage(
    spec: QueryGenSpec,
    llm_settings: LlmSettings,
    api_key: str,
    batch_candidate_ids: list[str],
) -> list[QueryBlueprint]:
    """Run one stage-1 planning invocation.

    Args:
        spec: Resolved query-generation specification.
        llm_settings: LLM settings for the query-generation workflow.
        api_key: Provider API key for the configured planning model.
        batch_candidate_ids: Candidate IDs assigned to this single planning invocation.

    Returns:
        The list of structured candidate blueprints returned by the planning stage.
    """
    prompt_vars = _build_planning_prompt_vars(
        spec=spec,
        batch_candidate_ids=batch_candidate_ids,
    )

    try:
        llm_runnable = build_llm_runnable(
            system_text=SYSTEM_PROMPT_PLANNING,
            user_text=USER_PROMPT_PLANNING,
            model_provider=llm_settings.model_provider,
            model=llm_settings.planning_model,
            api_key=api_key,
            output_schema=QueryBlueprintList,
            requests_per_second=llm_settings.requests_per_second,
            check_every_n_seconds=llm_settings.check_every_n_seconds,
            max_bucket_size=llm_settings.max_bucket_size,
            base_url=llm_settings.base_url,
            model_kwargs=llm_settings.model_kwargs,
        )
        llm_output = llm_runnable.invoke(prompt_vars)
    except LlmInitializationError:
        raise
    except Exception as exc:
        raise PlanningStageError("Planning stage invocation failed.") from exc

    return llm_output.candidates
