"""English catalog of Argilla dataset display strings.

This is the source-of-truth locale: every other locale must define the same
key set (enforced by tests). Strings extracted verbatim from the original
hardcoded values in :mod:`pragmata.core.annotation.argilla_task_definitions`.

Catalog structure:
- ``"field"`` / ``"question"`` / ``"guidelines"`` entries hold the title or
  guidelines text displayed in the UI.
- ``"label"`` entries hold the UI display text for individual
  ``LabelQuestion`` options. The ``name`` slot is
  ``"<question_name>.<label_value>"`` — the label *value* (e.g. ``"yes"``,
  ``"no"``, ``DiscardReason.UNCLEAR.value``) is the machine identifier
  that lands in exports, while the catalog value is what annotators see.
- ``"widget"`` entries hold the chrome strings inside the injected discard
  HTML widget. All locales ship inside every rendered widget so the JS can
  react to Argilla's live chrome-locale toggle.
"""

from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import DiscardReason, Task

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

_DISCARD_REASON_LABELS: dict[str, str] = {
    DiscardReason.INVALID_OR_UNREALISTIC.value: "Invalid or unrealistic record",
    DiscardReason.UNCLEAR.value: "Unclear query/context/answer relationship",
    DiscardReason.OUTSIDE_REVIEWER_EXPERTISE.value: "Outside reviewer expertise",
}

# Widget strings inside discard_flow.html. Shared across tasks (one set per
# locale); we still key by Task to keep the catalog structure homogeneous,
# but the values are identical for every task within a locale.
_DISCARD_WIDGET_STRINGS: dict[str, str] = {
    "discard.panel_summary": "Discard this record",
    "discard.panel_help": (
        "Use this if the record is unsuitable for annotation (e.g. invalid, unclear, or outside your expertise)."
    ),
    "discard.reason_label": "Reason:",
    "discard.reason_placeholder": "— select —",
    "discard.notes_label": "Optional notes:",
    "discard.notes_placeholder": "Any extra context (optional)",
    "discard.button_label": "Discard",
}


def _build_programmatic_entries() -> Catalog:
    """Generate label-display and widget-string entries.

    Done programmatically because:
    - Every LabelQuestion in EN shares the same yes/no displays.
    - Every task shares the same discard-reason displays.
    - Every task gets the same widget strings (the discard widget is identical
      across retrieval/grounding/generation).

    Spelling these out by hand triples the file length with no information
    gain. Per-locale overrides can still be applied by editing the final
    CATALOG dict.
    """
    entries: Catalog = {}
    for task, question_names in _YES_NO_QUESTIONS_BY_TASK.items():
        for question in question_names:
            entries[(task, "label", f"{question}.yes")] = "Yes"
            entries[(task, "label", f"{question}.no")] = "No"
        for reason_value, display in _DISCARD_REASON_LABELS.items():
            entries[(task, "label", f"discard_reason.{reason_value}")] = display
        for widget_key, widget_value in _DISCARD_WIDGET_STRINGS.items():
            entries[(task, "widget", widget_key)] = widget_value
    return entries


CATALOG: Catalog = {
    # ------------------- RETRIEVAL -------------------
    (Task.RETRIEVAL, "field", "query"): "Query",
    (Task.RETRIEVAL, "field", "chunk"): "Chunk",
    (Task.RETRIEVAL, "field", "generated_answer"): "Generated answer",
    (Task.RETRIEVAL, "field", "discard_flow"): "Discard this record",
    (Task.RETRIEVAL, "question", "topically_relevant"): (
        "Does this passage contain information that is substantively relevant to the query?"
    ),
    (Task.RETRIEVAL, "question", "evidence_sufficient"): (
        "Does this passage provide sufficient evidence to support answering the query?"
    ),
    (Task.RETRIEVAL, "question", "misleading"): (
        "Could this passage plausibly lead to an incorrect or distorted answer?"
    ),
    (Task.RETRIEVAL, "question", "notes"): "Notes (optional)",
    (Task.RETRIEVAL, "question", "discard_reason"): "Discard reason",
    (Task.RETRIEVAL, "question", "discard_notes"): "Discard notes (optional)",
    (Task.RETRIEVAL, "guidelines", ""): "Retrieval. TODO: Revisit after first annotation iteration.",
    # ------------------- GROUNDING -------------------
    (Task.GROUNDING, "field", "answer"): "Answer",
    (Task.GROUNDING, "field", "context_set"): "Context set",
    (Task.GROUNDING, "field", "query"): "Query",
    (Task.GROUNDING, "field", "discard_flow"): "Discard this record",
    (Task.GROUNDING, "question", "support_present"): (
        "Is at least one claim in the answer supported by the provided context?"
    ),
    (Task.GROUNDING, "question", "unsupported_claim_present"): (
        "Does the answer contain claims not supported by the provided context?"
    ),
    (Task.GROUNDING, "question", "contradicted_claim_present"): (
        "Does the provided context contradict any claim in the answer?"
    ),
    (Task.GROUNDING, "question", "source_cited"): "Does the answer contain a citation marker?",
    (Task.GROUNDING, "question", "fabricated_source"): (
        "Does the answer cite a source not present in the retrieved context?"
    ),
    (Task.GROUNDING, "question", "notes"): "Notes (optional)",
    (Task.GROUNDING, "question", "discard_reason"): "Discard reason",
    (Task.GROUNDING, "question", "discard_notes"): "Discard notes (optional)",
    (Task.GROUNDING, "guidelines", ""): "Grounding. TODO: Revisit after first annotation iteration.",
    # ------------------- GENERATION ------------------
    (Task.GENERATION, "field", "query"): "Query",
    (Task.GENERATION, "field", "answer"): "Answer",
    (Task.GENERATION, "field", "context_set"): "Context set",
    (Task.GENERATION, "field", "discard_flow"): "Discard this record",
    (Task.GENERATION, "question", "proper_action"): ("Did the system choose the appropriate action for this query?"),
    (Task.GENERATION, "question", "response_on_topic"): ("Does the response substantively address the user's query?"),
    (Task.GENERATION, "question", "helpful"): (
        "Would this response enable a typical user to make progress on their task?"
    ),
    (Task.GENERATION, "question", "incomplete"): "Does the response fail to cover required parts of the query?",
    (Task.GENERATION, "question", "unsafe_content"): ("Does the response contain unsafe or policy-violating content?"),
    (Task.GENERATION, "question", "notes"): "Notes (optional)",
    (Task.GENERATION, "question", "discard_reason"): "Discard reason",
    (Task.GENERATION, "question", "discard_notes"): "Discard notes (optional)",
    (Task.GENERATION, "guidelines", ""): "Generation. TODO: Revisit after first annotation iteration.",
    # ------------------- LABELS + WIDGET (programmatic) ------------------
    **_build_programmatic_entries(),
}
