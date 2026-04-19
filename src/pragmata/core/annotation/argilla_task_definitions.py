"""Hardcoded Argilla dataset definitions for the three annotation tasks.

These are Argilla rg.Settings objects - runtime task definitions, NOT boundary
schemas (those live in core/schemas/) or configurable settings (those live in
core/settings/). They encode the annotation protocol (fields, questions, labels)
and are hardcoded per ADR-0009.

Distribution (min_submitted) is intentionally omitted — it is an operational
setting controlled by AnnotationSettings.min_submitted and applied at
dataset creation time.
"""

import functools
from importlib.resources import files
from string import Template

import argilla as rg

from pragmata.core.schemas.annotation_task import DiscardReason, Task

DATASET_NAMES: dict[Task, str] = {
    Task.RETRIEVAL: "retrieval",
    Task.GROUNDING: "grounding",
    Task.GENERATION: "generation",
}


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


@functools.cache
def build_task_settings() -> dict[Task, rg.Settings]:
    """Build Argilla Settings for each annotation task.

    Deferred construction — call after an Argilla client is connected
    (or with a mock client in tests). Cached after first call.
    """
    template_text = files("pragmata.core.annotation").joinpath("collapsible_field.html").read_text(encoding="utf-8")
    discard_template = files("pragmata.core.annotation").joinpath("discard_flow.html").read_text(encoding="utf-8")
    discard_field = rg.CustomField(
        name="discard_flow",
        title="Discard this record",
        template=discard_template,
        advanced_mode=True,
        required=False,
    )

    return {
        Task.RETRIEVAL: rg.Settings(
            fields=[
                rg.TextField(name="query", title="Query", required=True),
                rg.TextField(name="chunk", title="Chunk", required=True),
                _collapsible_field("generated_answer", "Generated answer", template_text),
                discard_field,
            ],
            questions=[
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
            ],
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
                rg.TermsMetadataProperty("chunk_id", visible_for_annotators=False),
                rg.TermsMetadataProperty("doc_id", visible_for_annotators=False),
                rg.IntegerMetadataProperty("chunk_rank", min=1, visible_for_annotators=False),
            ],
            guidelines="Retrieval. TODO: Revisit after first annotation iteration.",
        ),
        Task.GROUNDING: rg.Settings(
            fields=[
                rg.TextField(name="answer", title="Answer", required=True),
                rg.TextField(name="context_set", title="Context set", required=True),
                _collapsible_field("query", "Query", template_text),
                discard_field,
            ],
            questions=[
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
            ],
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
            ],
            guidelines="Grounding. TODO: Revisit after first annotation iteration.",
        ),
        Task.GENERATION: rg.Settings(
            fields=[
                rg.TextField(name="query", title="Query", required=True),
                rg.TextField(name="answer", title="Answer", required=True),
                _collapsible_field("context_set", "Context set", template_text),
                discard_field,
            ],
            questions=[
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
            ],
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
            ],
            guidelines="Generation. TODO: Revisit after first annotation iteration.",
        ),
    }
