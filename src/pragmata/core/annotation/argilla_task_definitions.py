"""Hardcoded Argilla dataset definitions for the three annotation tasks.

These are Argilla rg.Settings objects - runtime task definitions, NOT boundary
schemas (those live in core/schemas/) or configurable settings (those live in
core/settings/). They encode the annotation protocol (fields, questions, labels)
and are hardcoded per ADR-0009.

Distribution (min_submitted) is intentionally omitted — it is an operational
setting controlled by AnnotationSettings.workspaces and applied at
dataset creation time.
"""

import json
from importlib.resources import files
from string import Template

import argilla as rg

from pragmata.core.annotation.logical_constraints import LOGICAL_CONSTRAINTS
from pragmata.core.schemas.annotation_task import DiscardReason, Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

# Static placeholder values seeded on every record for widget-only CustomFields.
# Argilla's frontend silently skips rendering a CustomField when the record has
# no value for that field, so even pure UI widgets need a placeholder. This is
# the SSOT mirror of the widget CustomField names below.
WIDGET_FIELD_PLACEHOLDERS: dict[str, dict[str, str]] = {
    "discard_flow": {"text": ""},
    "constraints_panel": {"text": ""},
}


DATASET_NAMES: dict[Task, str] = {
    Task.RETRIEVAL: "retrieval",
    Task.GROUNDING: "grounding",
    Task.GENERATION: "generation",
}


def dataset_name(task: Task, *, calibration: bool, dataset_id: str = "") -> str:
    """Always-suffixed Argilla dataset name for a task and purpose.

    Names are unconditional: production datasets are always
    ``task_<task>_production`` and calibration datasets are always
    ``task_<task>_calibration``. The ``dataset_id`` suffix is appended for
    run-scoping when present.
    """
    base = DATASET_NAMES[task]
    purpose = "calibration" if calibration else "production"
    name = f"{base}_{purpose}"
    return f"{name}_{dataset_id}" if dataset_id else name


def _collapsible_field(name: str, title: str, template_text: str) -> rg.CustomField:
    rendered = Template(template_text).substitute(field_name=name, summary_text=title)
    return rg.CustomField(
        name=name,
        title=title,
        template=rendered,
        advanced_mode=True,
        required=True,
    )


def _discard_questions() -> list[rg.LabelQuestion | rg.TextQuestion]:
    return [
        rg.LabelQuestion(
            name="discard_reason",
            title="Discard reason",
            labels=[r.value for r in DiscardReason],
            required=False,
        ),
        rg.TextQuestion(name="discard_notes", title="Discard notes (optional)", required=False),
    ]


def _render_constraints_template(
    task: Task, questions: list, template_text: str, settings: AnnotationSettings
) -> str:
    """Substitute the constraint + question-title payload into ``constraints_field.html``.

    Each constraint's payload severity is the deployment-scope value from
    ``settings.constraint_severity``.
    """
    constraints = LOGICAL_CONSTRAINTS[task]
    referenced = {q for c in constraints for q in (c.when_question, c.then_question)}
    titles = {q.name: q.title for q in questions if isinstance(q, rg.LabelQuestion) and q.name in referenced}
    payloads = [c.to_widget_payload(settings.constraint_severity[c.constraint_id]) for c in constraints]
    return Template(template_text).substitute(
        CONSTRAINTS_JSON=json.dumps(payloads, ensure_ascii=False),
        QUESTION_TITLES_JSON=json.dumps(titles, ensure_ascii=False),
    )


def build_task_settings(settings: AnnotationSettings) -> dict[Task, rg.Settings]:
    """Build Argilla Settings for each annotation task.

    Not cached: result depends on ``settings.constraint_severity``, so callers
    should hold the returned dict for the duration of an import operation
    rather than calling repeatedly. Deferred construction: call after an
    Argilla client is connected (or with a mock client in tests).
    """
    template_text = files("pragmata.core.annotation").joinpath("collapsible_field.html").read_text(encoding="utf-8")
    discard_template = files("pragmata.core.annotation").joinpath("discard_flow.html").read_text(encoding="utf-8")
    constraints_template = (
        files("pragmata.core.annotation").joinpath("constraints_field.html").read_text(encoding="utf-8")
    )

    # Fresh CustomField per task — FieldBase carries a `_dataset` attribute that
    # Argilla's Settings/Dataset plumbing mutates, so sharing one instance across
    # three rg.Settings risks cross-task coupling on future SDK changes.
    def discard_field() -> rg.CustomField:
        return rg.CustomField(
            name="discard_flow",
            title="Discard this record",
            template=discard_template,
            advanced_mode=True,
            required=False,
        )

    def constraints_field(task: Task, questions: list) -> rg.CustomField:
        # Always present on every task. Argilla requires every CustomField to
        # have a value on every record; WIDGET_FIELD_PLACEHOLDERS supplies one
        # for `constraints_panel`. When LOGICAL_CONSTRAINTS[task] is empty the
        # widget evaluates to no hits and stays hidden.
        return rg.CustomField(
            name="constraints_panel",
            title="Constraint checks",
            template=_render_constraints_template(task, questions, constraints_template, settings),
            advanced_mode=True,
            required=False,
        )

    def assemble(task: Task, content_fields: list, questions: list, metadata: list, guidelines: str) -> rg.Settings:
        # Constraints panel sits last in fields so it renders right above the
        # discard panel, adjacent to the submit area where the annotator acts.
        fields = [*content_fields, constraints_field(task, questions), discard_field()]
        return rg.Settings(fields=fields, questions=questions, metadata=metadata, guidelines=guidelines)

    retrieval_questions: list = [
        rg.LabelQuestion(
            name="topically_relevant",
            title="Does this passage contain information that is substantively relevant to the query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="evidence_sufficient",
            title="Does this passage provide sufficient evidence to support answering the query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="misleading",
            title="Could this passage plausibly lead to an incorrect or distorted answer?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.TextQuestion(name="notes", title="Notes (optional)", required=False),
        *_discard_questions(),
    ]

    grounding_questions: list = [
        rg.LabelQuestion(
            name="support_present",
            title="Is at least one claim in the answer supported by the provided context?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="unsupported_claim_present",
            title="Does the answer contain claims not supported by the provided context?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="contradicted_claim_present",
            title="Does the provided context contradict any claim in the answer?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="source_cited",
            title="Does the answer contain a citation marker?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="fabricated_source",
            title="Does the answer cite a source not present in the retrieved context?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.TextQuestion(name="notes", title="Notes (optional)", required=False),
        *_discard_questions(),
    ]

    generation_questions: list = [
        rg.LabelQuestion(
            name="proper_action",
            title="Did the system choose the appropriate action for this query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="response_on_topic",
            title="Does the response substantively address the user's query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="helpful",
            title="Would this response enable a typical user to make progress on their task?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="incomplete",
            title="Does the response fail to cover required parts of the query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="unsafe_content",
            title="Does the response contain unsafe or policy-violating content?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.TextQuestion(name="notes", title="Notes (optional)", required=False),
        *_discard_questions(),
    ]

    return {
        Task.RETRIEVAL: assemble(
            Task.RETRIEVAL,
            content_fields=[
                rg.TextField(name="query", title="Query", required=True),
                rg.TextField(name="chunk", title="Chunk", required=True),
                _collapsible_field("generated_answer", "Generated answer", template_text),
            ],
            questions=retrieval_questions,
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
                rg.TermsMetadataProperty("chunk_id", visible_for_annotators=False),
                rg.TermsMetadataProperty("doc_id", visible_for_annotators=False),
                rg.IntegerMetadataProperty("chunk_rank", min=1, visible_for_annotators=False),
            ],
            guidelines="Retrieval. TODO: Revisit after first annotation iteration.",
        ),
        Task.GROUNDING: assemble(
            Task.GROUNDING,
            content_fields=[
                rg.TextField(name="answer", title="Answer", required=True),
                rg.TextField(name="context_set", title="Context set", required=True),
                _collapsible_field("query", "Query", template_text),
            ],
            questions=grounding_questions,
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
            ],
            guidelines="Grounding. TODO: Revisit after first annotation iteration.",
        ),
        Task.GENERATION: assemble(
            Task.GENERATION,
            content_fields=[
                rg.TextField(name="query", title="Query", required=True),
                rg.TextField(name="answer", title="Answer", required=True),
                _collapsible_field("context_set", "Context set", template_text),
            ],
            questions=generation_questions,
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
            ],
            guidelines="Generation. TODO: Revisit after first annotation iteration.",
        ),
    }
