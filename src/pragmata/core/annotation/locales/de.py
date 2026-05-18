"""German catalog of Argilla dataset display strings.

All visible strings — including the discard widget's chrome, the
``discard_reason`` and ``discard_notes`` question titles, and the
discard-reason option labels — are translated. The widget finds its
hidden helper questions via Argilla's ``aria-label`` attribute, probing
every supported locale's title string, so translating the titles is
safe.
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
    DiscardReason.INVALID_OR_UNREALISTIC.value: "Ungültiger oder unrealistischer Datensatz",
    DiscardReason.UNCLEAR.value: "Unklarer Zusammenhang zwischen Anfrage/Kontext/Antwort",
    DiscardReason.OUTSIDE_REVIEWER_EXPERTISE.value: "Außerhalb des Fachgebiets der Begutachtung",
}

_DISCARD_WIDGET_STRINGS: dict[str, str] = {
    "discard.panel_summary": "Diesen Datensatz verwerfen",
    "discard.panel_help": (
        "Verwenden Sie dies, wenn der Datensatz nicht zur Annotation geeignet "
        "ist (z. B. ungültig, unklar oder außerhalb Ihres Fachgebiets)."
    ),
    "discard.reason_label": "Grund:",
    "discard.reason_placeholder": "— auswählen —",
    "discard.notes_label": "Optionale Anmerkungen:",
    "discard.notes_placeholder": "Zusätzlicher Kontext (optional)",
    "discard.button_label": "Verwerfen",
}


def _build_programmatic_entries() -> Catalog:
    entries: Catalog = {}
    for task, question_names in _YES_NO_QUESTIONS_BY_TASK.items():
        for question in question_names:
            entries[(task, "label", f"{question}.yes")] = "Ja"
            entries[(task, "label", f"{question}.no")] = "Nein"
        for reason_value, display in _DISCARD_REASON_LABELS.items():
            entries[(task, "label", f"discard_reason.{reason_value}")] = display
        for widget_key, widget_value in _DISCARD_WIDGET_STRINGS.items():
            entries[(task, "widget", widget_key)] = widget_value
    return entries


CATALOG: Catalog = {
    # ------------------- RETRIEVAL -------------------
    (Task.RETRIEVAL, "field", "query"): "Anfrage",
    (Task.RETRIEVAL, "field", "chunk"): "Textabschnitt",
    (Task.RETRIEVAL, "field", "generated_answer"): "Generierte Antwort",
    (Task.RETRIEVAL, "field", "discard_flow"): "Diesen Datensatz verwerfen",
    (Task.RETRIEVAL, "question", "topically_relevant"): (
        "Enthält dieser Textabschnitt Informationen, die für die Anfrage inhaltlich relevant sind?"
    ),
    (Task.RETRIEVAL, "question", "evidence_sufficient"): (
        "Liefert dieser Textabschnitt ausreichende Belege, um die Anfrage zu beantworten?"
    ),
    (Task.RETRIEVAL, "question", "misleading"): (
        "Könnte dieser Textabschnitt plausibel zu einer falschen oder verzerrten Antwort führen?"
    ),
    (Task.RETRIEVAL, "question", "notes"): "Anmerkungen (optional)",
    (Task.RETRIEVAL, "question", "discard_reason"): "Verwerfungsgrund",
    (Task.RETRIEVAL, "question", "discard_notes"): "Verwerfungs-Anmerkungen (optional)",
    (Task.RETRIEVAL, "guidelines", ""): "Retrieval. TODO: Nach erster Annotationsiteration überarbeiten.",
    # ------------------- GROUNDING -------------------
    (Task.GROUNDING, "field", "answer"): "Antwort",
    (Task.GROUNDING, "field", "context_set"): "Kontextmenge",
    (Task.GROUNDING, "field", "query"): "Anfrage",
    (Task.GROUNDING, "field", "discard_flow"): "Diesen Datensatz verwerfen",
    (Task.GROUNDING, "question", "support_present"): (
        "Wird mindestens eine Aussage in der Antwort durch den bereitgestellten Kontext gestützt?"
    ),
    (Task.GROUNDING, "question", "unsupported_claim_present"): (
        "Enthält die Antwort Aussagen, die nicht durch den bereitgestellten Kontext gestützt werden?"
    ),
    (Task.GROUNDING, "question", "contradicted_claim_present"): (
        "Widerspricht der bereitgestellte Kontext einer Aussage in der Antwort?"
    ),
    (Task.GROUNDING, "question", "source_cited"): "Enthält die Antwort eine Quellenangabe?",
    (Task.GROUNDING, "question", "fabricated_source"): (
        "Zitiert die Antwort eine Quelle, die nicht im abgerufenen Kontext enthalten ist?"
    ),
    (Task.GROUNDING, "question", "notes"): "Anmerkungen (optional)",
    (Task.GROUNDING, "question", "discard_reason"): "Verwerfungsgrund",
    (Task.GROUNDING, "question", "discard_notes"): "Verwerfungs-Anmerkungen (optional)",
    (Task.GROUNDING, "guidelines", ""): "Grounding. TODO: Nach erster Annotationsiteration überarbeiten.",
    # ------------------- GENERATION ------------------
    (Task.GENERATION, "field", "query"): "Anfrage",
    (Task.GENERATION, "field", "answer"): "Antwort",
    (Task.GENERATION, "field", "context_set"): "Kontextmenge",
    (Task.GENERATION, "field", "discard_flow"): "Diesen Datensatz verwerfen",
    (Task.GENERATION, "question", "proper_action"): (
        "Hat das System die für diese Anfrage angemessene Aktion gewählt?"
    ),
    (Task.GENERATION, "question", "response_on_topic"): (
        "Geht die Antwort inhaltlich auf die Anfrage des Nutzers ein?"
    ),
    (Task.GENERATION, "question", "helpful"): (
        "Würde diese Antwort einem typischen Nutzer ermöglichen, bei seiner Aufgabe voranzukommen?"
    ),
    (Task.GENERATION, "question", "incomplete"): ("Lässt die Antwort erforderliche Teile der Anfrage unbeantwortet?"),
    (Task.GENERATION, "question", "unsafe_content"): (
        "Enthält die Antwort unsichere oder richtlinienverletzende Inhalte?"
    ),
    (Task.GENERATION, "question", "notes"): "Anmerkungen (optional)",
    (Task.GENERATION, "question", "discard_reason"): "Verwerfungsgrund",
    (Task.GENERATION, "question", "discard_notes"): "Verwerfungs-Anmerkungen (optional)",
    (Task.GENERATION, "guidelines", ""): "Generation. TODO: Nach erster Annotationsiteration überarbeiten.",
    # ------------------- LABELS + WIDGET (programmatic) ------------------
    **_build_programmatic_entries(),
}
