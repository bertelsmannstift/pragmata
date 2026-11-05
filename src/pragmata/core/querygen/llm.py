"""LLM composition boundary for synthetic query generation."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_core.runnables.base import RunnableSerializable

from pragmata.core.settings.querygen_settings import LlmSettings
from pragmata.core.types import M

_RESERVED_MODEL_KWARGS = {
    "api_key",
    "base_url",
    "model",
    "model_provider",
    "rate_limiter",
}

_DEFAULT_REQUEST_TIMEOUT_SECONDS = 600


class LlmInitializationError(RuntimeError):
    """Raised when the configured chat model cannot be initialized."""


def build_rate_limiter(llm_settings: LlmSettings) -> InMemoryRateLimiter:
    """Build a rate limiter to share across all stages and batches of one run."""
    return InMemoryRateLimiter(
        requests_per_second=llm_settings.requests_per_second,
        check_every_n_seconds=llm_settings.check_every_n_seconds,
        max_bucket_size=llm_settings.max_bucket_size,
    )


def _build_prompt_template(
    *,
    system_text: str,
    user_text: str,
) -> ChatPromptTemplate:
    """Build a chat prompt template from system and user prompt text.

    Args:
        system_text: System prompt text.
        user_text: User prompt text.

    Returns:
        A two-message chat prompt template with system and user-facing messages.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", user_text),
        ]
    )


def build_llm_runnable(
    *,
    system_text: str,
    user_text: str,
    model_provider: str,
    model: str,
    api_key: str,
    output_schema: type[M],
    rate_limiter: BaseRateLimiter,
    base_url: str | None,
    model_kwargs: dict[str, Any],
) -> RunnableSerializable:
    """Compose a schema-constrained LangChain runnable for query generation.

    Args:
        system_text: System prompt text.
        user_text: User prompt text.
        model_provider: LangChain provider identifier.
        model: Provider-specific model identifier.
        api_key: Provider API key.
        output_schema: Pydantic schema class used for structured output.
        rate_limiter: Shared rate limiter for the query-generation run.
        base_url: Optional provider base URL.
        model_kwargs: Optional extra keyword arguments forwarded to model init.
            A default request ``timeout`` of ``_DEFAULT_REQUEST_TIMEOUT_SECONDS``
            is applied unless ``model_kwargs`` supplies its own ``timeout``.

    Returns:
        A composed LangChain runnable that accepts prompt variables and returns
        structured output validated against ``output_schema``.
    """
    if rate_limiter is None:
        raise ValueError("rate_limiter must not be None.")

    overlapping_keys = _RESERVED_MODEL_KWARGS.intersection(model_kwargs)
    if overlapping_keys:
        overlapping = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"model_kwargs must not override core LLM settings: {overlapping}. "
            "Pass these via dedicated arguments instead."
        )

    init_kwargs: dict[str, Any] = {
        "model": model,
        "model_provider": model_provider,
        "api_key": api_key,
        "timeout": _DEFAULT_REQUEST_TIMEOUT_SECONDS,
        "rate_limiter": rate_limiter,
    }

    if base_url is not None:
        init_kwargs["base_url"] = base_url

    if model_kwargs:
        init_kwargs.update(model_kwargs)

    try:
        llm = init_chat_model(**init_kwargs)
    except Exception as exc:
        raise LlmInitializationError(
            f"Failed to initialize provider '{model_provider}' with model '{model}'. "
            "Ensure the corresponding LangChain provider package is installed and the model is valid."
        ) from exc
    structured_llm = llm.with_structured_output(output_schema)
    retry_llm = structured_llm.with_retry()

    prompt = _build_prompt_template(
        system_text=system_text,
        user_text=user_text,
    )

    return prompt | retry_llm
