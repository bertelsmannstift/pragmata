"""Tests for the synthetic query-generation LLM composition boundary."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel

from pragmata.core.querygen.llm import LlmInitializationError, _build_prompt_template, build_llm_runnable


class _DummyOutputSchema(BaseModel):
    """Minimal structured-output schema for composition tests."""

    value: str


class _FakePrompt:
    """Fake prompt supporting runnable composition via ``|``."""

    def __init__(self) -> None:
        self.or_operands: list[Any] = []
        self.composed_result = object()

    def __or__(self, other: Any) -> object:
        """Record the composed operand and return a sentinel pipeline."""
        self.or_operands.append(other)
        return self.composed_result


@pytest.fixture
def mock_llm_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Mock LangChain model construction and fluent wrapper chaining."""
    mock_llm = MagicMock()
    mock_structured = mock_llm.with_structured_output.return_value
    mock_retry = mock_structured.with_retry.return_value

    mock_init = MagicMock(return_value=mock_llm)
    monkeypatch.setattr("pragmata.core.querygen.llm.init_chat_model", mock_init)

    return {
        "init": mock_init,
        "llm": mock_llm,
        "structured": mock_structured,
        "retry": mock_retry,
    }


def test_build_prompt_template_returns_chat_prompt_template() -> None:
    """_build_prompt_template returns a two-message chat prompt template."""
    prompt = _build_prompt_template(
        system_text="System A",
        user_text="User {val}",
    )

    assert isinstance(prompt, ChatPromptTemplate)

    messages = prompt.format_messages(val=123)
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[0].content == "System A"
    assert messages[1].type == "human"
    assert messages[1].content == "User 123"


def test_build_llm_runnable_composes_wrappers_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
    mock_llm_setup: dict[str, Any],
) -> None:
    """build_llm_runnable composes prompt with retry-wrapped structured LLM."""
    fake_prompt = _FakePrompt()
    monkeypatch.setattr(
        "pragmata.core.querygen.llm._build_prompt_template",
        lambda *, system_text, user_text: fake_prompt,
    )

    result = build_llm_runnable(
        system_text="sys",
        user_text="usr",
        model_provider="openai",
        model="gpt-4o",
        api_key="sk-test",
        output_schema=_DummyOutputSchema,
        requests_per_second=5.0,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
        base_url=None,
        model_kwargs={"temperature": 0},
    )

    mock_llm_setup["init"].assert_called_once()
    init_kwargs = mock_llm_setup["init"].call_args.kwargs
    assert init_kwargs["model"] == "gpt-4o"
    assert init_kwargs["model_provider"] == "openai"
    assert init_kwargs["api_key"] == "sk-test"
    assert init_kwargs["temperature"] == 0
    assert isinstance(init_kwargs["rate_limiter"], InMemoryRateLimiter)

    mock_llm_setup["llm"].with_structured_output.assert_called_once_with(_DummyOutputSchema)
    mock_llm_setup["structured"].with_retry.assert_called_once()

    assert fake_prompt.or_operands == [mock_llm_setup["retry"]]
    assert result is fake_prompt.composed_result


@pytest.mark.parametrize(
    ("base_url", "model_kwargs", "expected_present", "expected_absent"),
    [
        (
            None,
            {},
            {"model", "model_provider", "api_key", "rate_limiter"},
            {"base_url", "temperature"},
        ),
        (
            "https://api.custom.com",
            {"temperature": 0.2},
            {"model", "model_provider", "api_key", "rate_limiter", "base_url", "temperature"},
            set(),
        ),
    ],
)
def test_build_llm_runnable_inserts_optional_init_kwargs_conditionally(
    mock_llm_setup: dict[str, Any],
    base_url: str | None,
    model_kwargs: dict[str, Any],
    expected_present: set[str],
    expected_absent: set[str],
) -> None:
    """Optional init kwargs are forwarded only when provided."""
    build_llm_runnable(
        system_text="s",
        user_text="u",
        model_provider="p",
        model="m",
        api_key="k",
        output_schema=_DummyOutputSchema,
        requests_per_second=1.0,
        check_every_n_seconds=1.0,
        max_bucket_size=1,
        base_url=base_url,
        model_kwargs=model_kwargs,
    )

    init_kwargs = mock_llm_setup["init"].call_args.kwargs
    assert expected_present.issubset(init_kwargs.keys())
    assert expected_absent.isdisjoint(init_kwargs.keys())


def test_build_llm_runnable_wraps_init_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model initialization errors are wrapped in LlmInitializationError."""

    def error_init(**_: Any) -> Any:
        raise ValueError("Invalid Provider")

    monkeypatch.setattr("pragmata.core.querygen.llm.init_chat_model", error_init)

    with pytest.raises(
        LlmInitializationError,
        match=r"Failed to initialize provider 'bad' with model 'm'\.",
    ) as exc_info:
        build_llm_runnable(
            system_text="s",
            user_text="u",
            model_provider="bad",
            model="m",
            api_key="k",
            output_schema=_DummyOutputSchema,
            requests_per_second=1.0,
            check_every_n_seconds=1.0,
            max_bucket_size=1,
            base_url=None,
            model_kwargs={},
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "Invalid Provider"


@pytest.mark.parametrize(
    "reserved_key",
    ["api_key", "base_url", "model", "model_provider", "rate_limiter"],
)
def test_build_llm_runnable_rejects_reserved_model_kwargs(
    mock_llm_setup: dict[str, Any],
    reserved_key: str,
) -> None:
    """Reserved init kwargs cannot be overridden via model_kwargs."""
    with pytest.raises(
        ValueError,
        match=rf"model_kwargs must not override core LLM settings: {reserved_key}",
    ):
        build_llm_runnable(
            system_text="s",
            user_text="u",
            model_provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            output_schema=_DummyOutputSchema,
            requests_per_second=1.0,
            check_every_n_seconds=1.0,
            max_bucket_size=1,
            base_url=None,
            model_kwargs={reserved_key: "override"},
        )

    mock_llm_setup["init"].assert_not_called()
