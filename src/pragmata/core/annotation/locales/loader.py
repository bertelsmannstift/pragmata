"""YAML catalog loader and locale-invariant structural metadata.

A locale catalog is a flat ``dict[(Task, CatalogKind, str), str]`` consumed by
:mod:`pragmata.core.annotation.argilla_task_definitions`. The translatable
strings live in per-locale YAML files; the *structure* (which questions share
the yes/no LabelQuestion, which DiscardReason values exist) is code and lives
here. :func:`load_catalog` fans the YAML inputs over that structure into the
flat catalog shape.
"""

from pathlib import Path
from typing import Any

import yaml

from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import Task

_YES_NO_QUESTIONS_BY_TASK: dict[Task, list[str]] = {
    Task.RETRIEVAL: ["topically_relevant", "evidence_sufficient", "misleading"],
    Task.GROUNDING: [
        "support_present",
        "unsupported_claim_present",
        "contradicted_claim_present",
        "source_cited",
        "fabricated_source",
    ],
    Task.GENERATION: ["proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"],
}


def load_catalog(path: Path) -> Catalog:
    """Build a Catalog from a YAML translation file.

    Raises ``KeyError`` if a required top-level section, task, field, question,
    or label value is missing — fail-loud is intentional so a typo in a locale
    file surfaces at import time, not at first widget render.
    """
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    catalog: Catalog = {}

    for task_name, fields in data["fields"].items():
        task = Task(task_name)
        for field_name, display in fields.items():
            catalog[(task, "field", field_name)] = display

    for task_name, questions in data["questions"].items():
        task = Task(task_name)
        for question_name, display in questions.items():
            catalog[(task, "question", question_name)] = display

    for task_name, text in data["guidelines"].items():
        catalog[(Task(task_name), "guidelines", "")] = text

    labels = data["labels"]
    yes_display, no_display = labels["yes_display"], labels["no_display"]
    discard_reasons: dict[str, str] = labels["discard_reasons"]
    for task, question_names in _YES_NO_QUESTIONS_BY_TASK.items():
        for question in question_names:
            catalog[(task, "label", f"{question}.yes")] = yes_display
            catalog[(task, "label", f"{question}.no")] = no_display
        for reason_value, display in discard_reasons.items():
            catalog[(task, "label", f"discard_reason.{reason_value}")] = display

    return catalog
