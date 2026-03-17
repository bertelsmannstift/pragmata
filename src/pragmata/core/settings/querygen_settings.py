"""Run settings models for synthetic query generation."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from chatboteval.core.schemas.querygen_input import QueryGenSpec
from chatboteval.core.settings.settings_base import ResolveSettings


class LlmSettings(BaseModel):
    """Non-secret LLM settings for synthetic query generation."""

    model_config = ConfigDict(extra="forbid")

    model_provider: str = "mistralai"
    planning_model: str = "magistral-medium-latest"
    realization_model: str = "mistral-medium-latest"
    base_url: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class QueryGenRunSettings(ResolveSettings):
    """Synthetic query generation run settings."""

    spec: QueryGenSpec
    llm: LlmSettings = Field(default_factory=LlmSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    n_queries: PositiveInt = 50
