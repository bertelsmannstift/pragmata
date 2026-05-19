"""Smoke test: the shipped Docker Compose file is in the wheel and parses.

Marked `packaging` so it's excluded from the default test run (see
pyproject.toml addopts). Run via `python -m pytest -m packaging`.

This test is deliberately minimal — its only job is to catch wheel / sdist
divergence (e.g. the compose file silently dropped from
[tool.hatch.build.targets.wheel] after a refactor). Behavioural correctness
(does Argilla actually come up against the shipped compose) belongs to the
dev-override integration tests; see docs/design/annotation-stack-lifecycle.md
§2.2.
"""

from importlib.resources import files

import pytest
import yaml


@pytest.mark.packaging
def test_shipped_compose_is_packaged() -> None:
    """The shipped compose file resolves via importlib.resources and parses as YAML."""
    path = files("pragmata.annotation").joinpath("docker-compose.yml")
    assert path.is_file(), f"docker-compose.yml missing from pragmata.annotation package data ({path})"
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), "shipped compose did not parse to a mapping"
    assert "services" in parsed, "shipped compose has no `services` block"
    assert "argilla" in parsed["services"], "shipped compose missing `argilla` service"
