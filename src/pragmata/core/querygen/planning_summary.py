"""Stage-1 planning summary executor for synthetic query generation."""

import hashlib
import json
from pathlib import Path

from pragmata.core.querygen.llm import LlmInitializationError, build_llm_runnable
from pragmata.core.querygen.planning import format_string_list, format_weighted_values
from pragmata.core.querygen.prompts import (
    SYSTEM_PROMPT_PLANNING_SUMMARY,
    USER_PROMPT_PLANNING_SUMMARY,
)
from pragmata.core.querygen.realization import format_blueprint
from pragmata.core.schemas.querygen_input import QueryGenSpec
from pragmata.core.schemas.querygen_output import PlanningSummaryArtifact
from pragmata.core.schemas.querygen_plan import QueryBlueprint
from pragmata.core.schemas.querygen_summary import PlanningSummaryState
from pragmata.core.settings.querygen_settings import LlmSettings


class PlanningSummaryStageError(RuntimeError):
    """Raised when a planning-summary-stage invocation fails."""


def _serialize_spec_content(
    spec: QueryGenSpec,
) -> str:
    """Build a deterministic content-only serialization for a querygen spec.

    Args:
        spec: Resolved query-generation specification.

    Returns:
        A canonical JSON string suitable for stable hashing.
    """
    payload = spec.model_dump(mode="json")

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def fingerprint_querygen_spec(
    spec: QueryGenSpec,
) -> str:
    """Return a deterministic SHA-256 fingerprint for a querygen spec.

    Args:
        spec: Resolved query-generation specification.

    Returns:
        Stable SHA-256 hex digest of the canonical serialized spec content.
    """
    serialized = _serialize_spec_content(spec)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_planning_summary_artifact(
    artifact_path: Path,
    spec: QueryGenSpec,
) -> PlanningSummaryArtifact | None:
    """Read a planning-summary artifact if it matches the current spec.

    Args:
        artifact_path: Path to the persisted planning-summary artifact JSON file.
        spec: Resolved query-generation specification for the current run.

    Returns:
        The validated planning-summary artifact when the file exists and its
        ``spec_fingerprint`` exactly matches the current spec fingerprint.
        Returns ``None`` when the path does not exist or when the artifact
        fingerprint is incompatible with the current spec.
    """
    if not artifact_path.exists():
        return None

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact = PlanningSummaryArtifact.model_validate(payload)

    current_spec_fingerprint = fingerprint_querygen_spec(spec)
    if artifact.spec_fingerprint != current_spec_fingerprint:
        return None

    return artifact


def _normalize_multiline(value: str) -> str:
    """Normalize multiline text to a single line for prompt safety."""
    return " ".join(value.splitlines()).strip()


def _format_prior_summary_state(
    prior_summary_state: PlanningSummaryState,
) -> str:
    """Format a prior planning summary for prompt injection.

    Args:
        prior_summary_state: Prior summary state carried forward from earlier batches.

    Returns:
        A deterministic human-readable representation of the prior planning summary.
    """
    return (
        "- redundancy_patterns:\n"
        f"  {_normalize_multiline(prior_summary_state.redundancy_patterns)}\n"
        "- diversification_targets:\n"
        f"  {_normalize_multiline(prior_summary_state.diversification_targets)}\n"
        "- coverage_notes:\n"
        f"  {_normalize_multiline(prior_summary_state.coverage_notes)}"
    )


def _build_planning_summary_prompt_vars(
    spec: QueryGenSpec,
    candidates: list[QueryBlueprint],
    prior_summary_state: PlanningSummaryState | None,
) -> dict[str, object]:
    """Build invoke-time prompt variables for one planning-summary.

    Args:
        spec: Resolved query-generation specification.
        candidates: Stage-1 candidates for this single summary-updater invocation.
        prior_summary_state: Optional prior planning summary state.

    Returns:
        Prompt variables aligned with the summary-updater prompt placeholders.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    formatted_blueprints = "\n\n".join(format_blueprint(candidate) for candidate in candidates)

    return {
        "domains": format_weighted_values(spec.domain_context.domains),
        "roles": format_weighted_values(spec.domain_context.roles),
        "languages": format_weighted_values(spec.domain_context.languages),
        "topics": format_weighted_values(spec.knowledge_scope.topics),
        "intents": format_weighted_values(spec.scenario.intents),
        "tasks": format_weighted_values(spec.scenario.tasks),
        "difficulty": format_weighted_values(spec.scenario.difficulty),
        "formats": format_weighted_values(spec.format_requests.formats),
        "disallowed_topics": format_string_list(spec.safety.disallowed_topics),
        "prior_planning_summary": (
            _format_prior_summary_state(prior_summary_state)
            if prior_summary_state is not None
            else "No prior planning summary available yet."
        ),
        "query_blueprints": formatted_blueprints,
    }


def run_planning_summary(
    spec: QueryGenSpec,
    candidates: list[QueryBlueprint],
    llm_settings: LlmSettings,
    api_key: str,
    prior_summary_state: PlanningSummaryState | None = None,
) -> PlanningSummaryState:
    """Run one summary-updater invocation.

    Args:
        spec: Resolved query-generation specification.
        candidates: Stage-1 candidates for this single summary-updater invocation.
        llm_settings: LLM settings for the query-generation workflow.
        api_key: Provider API key for the configured planning model.
        prior_summary_state: Optional prior planning summary state.

    Returns:
        The updated planning summary state.
    """
    prompt_vars = _build_planning_summary_prompt_vars(
        spec=spec,
        candidates=candidates,
        prior_summary_state=prior_summary_state,
    )

    try:
        llm_runnable = build_llm_runnable(
            system_text=SYSTEM_PROMPT_PLANNING_SUMMARY,
            user_text=USER_PROMPT_PLANNING_SUMMARY,
            model_provider=llm_settings.model_provider,
            model=llm_settings.planning_model,
            api_key=api_key,
            output_schema=PlanningSummaryState,
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
        raise PlanningSummaryStageError(
            "Planning stage invocation failed while updating the planning summary."
        ) from exc

    return llm_output
