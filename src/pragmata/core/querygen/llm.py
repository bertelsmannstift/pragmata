"""LLM composition boundary for synthetic query generation."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.base import RunnableSerializable

from pragmata.core.types import M


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
    requests_per_second: float,
    check_every_n_seconds: float,
    max_bucket_size: int,
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
        requests_per_second: Rate limiter refill rate.
        check_every_n_seconds: Rate limiter polling interval.
        max_bucket_size: Rate limiter burst size.
        base_url: Optional provider base URL.
        model_kwargs: Optional extra keyword arguments forwarded to model init.

    Returns:
        A composed LangChain runnable that accepts prompt variables and returns
        structured output validated against ``output_schema``.
    """
    init_kwargs: dict[str, Any] = {
        "model": model,
        "model_provider": model_provider,
        "api_key": api_key,
        "rate_limiter": InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,
            max_bucket_size=max_bucket_size,
        ),
    }

    if base_url is not None:
        init_kwargs["base_url"] = base_url

    if model_kwargs:
        init_kwargs.update(model_kwargs)

    llm = init_chat_model(**init_kwargs)
    structured_llm = llm.with_structured_output(output_schema)
    retry_llm = structured_llm.with_retry()

    prompt = _build_prompt_template(
        system_text=system_text,
        user_text=user_text,
    )

    return prompt | retry_llm
