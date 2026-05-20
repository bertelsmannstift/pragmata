"""Locale-invariant structural metadata and catalog builder.

The per-task question registry and the list of widget chrome keys are the
same for every locale; only the displayed strings vary. Per-locale modules
supply translations and call :func:`build_programmatic_entries` to fan them
out into the catalog's keyed tuple shape.
"""

from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import Task

YES_NO_QUESTIONS_BY_TASK: dict[Task, list[str]] = {
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


def build_programmatic_entries(
    *,
    yes_display: str,
    no_display: str,
    discard_reason_displays: dict[str, str],
    discard_widget_displays: dict[str, str],
) -> Catalog:
    """Generate the label and widget catalog rows from per-locale translations.

    Catalog values are duplicated per task even when structurally identical
    (yes/no across all yes/no questions, widget strings across all three
    tasks) to keep the catalog key shape uniform. Raises ``KeyError`` if
    ``discard_widget_displays`` is missing any :data:`DISCARD_WIDGET_KEYS`
    entry.
    """
    entries: Catalog = {}
    for task, question_names in YES_NO_QUESTIONS_BY_TASK.items():
        for question in question_names:
            entries[(task, "label", f"{question}.yes")] = yes_display
            entries[(task, "label", f"{question}.no")] = no_display
        for reason_value, display in discard_reason_displays.items():
            entries[(task, "label", f"discard_reason.{reason_value}")] = display
        for widget_key in DISCARD_WIDGET_KEYS:
            entries[(task, "widget", f"discard.{widget_key}")] = discard_widget_displays[widget_key]
    return entries
