"""YAML catalog loader and locale-invariant structural metadata.

A locale catalog is a flat ``dict[(Task, CatalogKind, str), str]`` consumed by
:mod:`pragmata.core.annotation.argilla_task_definitions`. The translatable
strings live in per-locale YAML files; the *structure* (which questions share
the yes/no LabelQuestion, which DiscardReason values exist) is code and lives
here. :func:`load_catalog` fans the YAML inputs over that structure into the
flat catalog shape.
"""

from pathlib import Path

from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import DiscardReason, Task
from pragmata.core.settings.settings_base import load_config_file

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

DISCARD_WIDGET_KEYS: tuple[str, ...] = (
    "panel_summary",
    "panel_help",
    "reason_label",
    "reason_placeholder",
    "notes_label",
    "notes_placeholder",
    "button_label",
)


def load_catalog(path: Path) -> Catalog:
    """Build a Catalog from a YAML translation file.

    Raises ``KeyError`` if a required top-level section, task, field, question,
    or label value is missing — fail-loud is intentional so a typo in a locale
    file surfaces at import time, not at first widget render.
    """
    data = load_config_file(path)
    catalog: Catalog = {}

    for kind in ("field", "question"):
        for task_name, entries in data[f"{kind}s"].items():
            task = Task(task_name)
            for name, display in entries.items():
                catalog[(task, kind, name)] = display

    for task_name, text in data["guidelines"].items():
        catalog[(Task(task_name), "guidelines", "")] = text

    labels = data["labels"]
    yes_display, no_display = labels["yes_display"], labels["no_display"]
    discard_reasons: dict[str, str] = labels["discard_reasons"]
    widget: dict[str, str] = data["widget"]
    for task, question_names in _YES_NO_QUESTIONS_BY_TASK.items():
        for question in question_names:
            catalog[(task, "label", f"{question}.yes")] = yes_display
            catalog[(task, "label", f"{question}.no")] = no_display
        for reason in DiscardReason:
            catalog[(task, "label", f"discard_reason.{reason.value}")] = discard_reasons[reason.value]
        for key in DISCARD_WIDGET_KEYS:
            catalog[(task, "widget", f"discard.{key}")] = widget[key]

    return catalog
