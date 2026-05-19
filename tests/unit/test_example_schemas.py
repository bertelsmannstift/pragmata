"""Verify shipped example files parse against the current schemas.

The example files in `pragmata.annotation.examples` ship as package data
so pip-installed users can resolve them via `importlib.resources`. These
tests catch schema drift — if the canonical UserSpec / QueryResponsePair
shape changes and the examples don't, the next `pragmata annotation
setup` / `import` against the examples breaks for end users.
"""

import dataclasses
import json
from importlib.resources import files

from pragmata.core.schemas.annotation_import import QueryResponsePair
from pragmata.core.settings.annotation_settings import AnnotationSettings, UserSpec


def test_users_example_parses_as_user_spec() -> None:
    """users.example.json parses cleanly as a list of UserSpec."""
    text = files("pragmata.annotation.examples").joinpath("users.example.json").read_text(encoding="utf-8")
    data = json.loads(text)
    specs = [UserSpec(**entry) for entry in data]
    assert len(specs) == 3
    assert all(s.password is None for s in specs), "examples must remain passwordless"


def test_users_example_workspaces_match_default_topology() -> None:
    """Example users reference workspace names that exist in the zero-config default topology.

    Without this, `pragmata annotation setup --users users.example.json` (no
    --config) fails because the example asserts workspaces the defaults
    don't provision.
    """
    text = files("pragmata.annotation.examples").joinpath("users.example.json").read_text(encoding="utf-8")
    data = json.loads(text)
    referenced_workspaces = {ws for entry in data for ws in entry["workspaces"]}
    default_workspaces = set(AnnotationSettings().workspace_dataset_map)
    missing = referenced_workspaces - default_workspaces
    assert not missing, (
        f"users.example.json references workspaces not in zero-config defaults: {missing}. "
        f"Either change the example to use default names {default_workspaces}, or pair it with "
        f"a topic.example.yaml that defines those workspace names."
    )


def test_topic_example_parses_as_query_response_pairs() -> None:
    """topic.example.jsonl: every line parses as a QueryResponsePair."""
    text = files("pragmata.annotation.examples").joinpath("topic.example.jsonl").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    pairs = [QueryResponsePair(**json.loads(line)) for line in lines]
    assert len(pairs) >= 1
    languages = {p.language for p in pairs}
    assert {"en", "de"}.issubset(languages), "bundled example should demonstrate the language field across locales"


def test_example_readme_ships_as_package_data() -> None:
    """README.md is part of the package data so pip-installed users can read the schema docs."""
    readme = files("pragmata.annotation.examples").joinpath("README.md")
    assert readme.is_file()
    content = readme.read_text(encoding="utf-8")
    assert "users.example.json" in content
    assert "topic.example.jsonl" in content


def test_examples_dataclass_unknown_fields_rejected() -> None:
    """Catch malformed users.example.json entries that have stray fields."""
    text = files("pragmata.annotation.examples").joinpath("users.example.json").read_text(encoding="utf-8")
    data = json.loads(text)
    allowed = {f.name for f in dataclasses.fields(UserSpec)}
    for entry in data:
        unknown = set(entry) - allowed
        assert not unknown, f"users.example.json entry has unknown fields: {unknown}"
