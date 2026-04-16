"""Synthetic query generation run settings."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from pragmata.core.schemas.querygen_input import QueryGenSpec
from pragmata.core.settings.settings_base import ResolveSettings


class LlmSettings(BaseModel):
    """Non-secret LLM settings for synthetic query generation."""

    model_config = ConfigDict(extra="forbid")

    model_provider: str = "mistralai"
    planning_model: str = "magistral-medium-latest"
    realization_model: str = "mistral-medium-latest"
    base_url: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    requests_per_second: float = Field(default=1.0, gt=0)
    check_every_n_seconds: float = Field(default=1.0, gt=0)
    max_bucket_size: int = Field(default=1, ge=1)


class QueryGenRunSettings(ResolveSettings):
    """Synthetic query generation run settings."""

    spec: QueryGenSpec
    llm: LlmSettings = Field(default_factory=LlmSettings)
    base_dir: Path = Field(default_factory=Path.cwd)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    n_queries: PositiveInt = 50
    batch_size: PositiveInt = 25
    enable_planning_memory: bool = True
