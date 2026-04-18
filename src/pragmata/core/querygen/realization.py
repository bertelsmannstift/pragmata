"""Stage-2 realization executor for synthetic query generation."""

from pragmata.core.querygen.llm import LlmInitializationError, build_llm_runnable
from pragmata.core.querygen.prompts import SYSTEM_PROMPT_REALIZATION, USER_PROMPT_REALIZATION
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_realize import RealizedQuery, RealizedQueryList
from pragmata.core.settings.querygen_settings import LlmSettings


class RealizationStageError(RuntimeError):
    """Raised when a realization-stage invocation fails."""


def format_blueprint(
    candidate: QueryBlueprint,
) -> str:
    """Format one query blueprint for prompt injection.

    Args:
        candidate: Structured candidate query blueprint selected for realization.

    Returns:
        A human-readable multiline representation of the blueprint with
        deterministic field order and optional fields included only when present.
    """
    lines = [
        f"- candidate_id: {candidate.candidate_id}",
        f"  domain: {candidate.domain}",
        f"  role: {candidate.role}",
        f"  language: {candidate.language}",
        f"  topic: {candidate.topic}",
        f"  intent: {candidate.intent}",
        f"  task: {candidate.task}",
    ]

    if candidate.difficulty is not None:
        lines.append(f"  difficulty: {candidate.difficulty}")

    if candidate.format is not None:
        lines.append(f"  format: {candidate.format}")

    lines.extend(
        [
            f"  user_scenario: {candidate.user_scenario}",
            f"  information_need: {candidate.information_need}",
        ]
    )

    return "\n".join(lines)


def _build_realization_prompt_vars(
    candidates: list[QueryBlueprint],
) -> dict[str, object]:
    """Build invoke-time prompt variables for one realization-stage batch.

    Args:
        candidates: Selected stage-1 candidate blueprints for this single
            realization invocation.

    Returns:
        A dict containing a single ``query_blueprints`` key whose value is a
        human-readable formatted string representation of the input blueprints.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    formatted_blueprints = "\n\n".join(format_blueprint(candidate) for candidate in candidates)

    return {
        "query_blueprints": formatted_blueprints,
    }


def run_realization_stage(
    candidates: list[QueryBlueprint],
    llm_settings: LlmSettings,
    api_key: str,
) -> list[RealizedQuery]:
    """Run one stage-2 realization invocation.

    Args:
        candidates: Selected stage-1 candidate blueprints for this single
            realization invocation.
        llm_settings: LLM settings for the query-generation workflow.
        api_key: Provider API key for the configured realization model.

    Returns:
        The list of realized queries returned by the realization stage.
    """
    prompt_vars = _build_realization_prompt_vars(candidates)

    try:
        llm_runnable = build_llm_runnable(
            system_text=SYSTEM_PROMPT_REALIZATION,
            user_text=USER_PROMPT_REALIZATION,
            model_provider=llm_settings.model_provider,
            model=llm_settings.realization_model,
            api_key=api_key,
            output_schema=RealizedQueryList,
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
        raise RealizationStageError("Realization stage invocation failed.") from exc

    return llm_output.queries
