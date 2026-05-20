"""Hardcoded Argilla dataset definitions for the three annotation tasks.

These are Argilla rg.Settings objects - runtime task definitions, NOT boundary
schemas (those live in core/schemas/) or configurable settings (those live in
core/settings/). They encode the annotation protocol (fields, questions, labels)
and are hardcoded per ADR-0009.

Display strings (titles, guidelines, and label option text) are looked up
from per-locale catalogs in :mod:`pragmata.core.annotation.locales`.
Identities (field/question ``name=``) and label *values* (``"yes"``,
``"no"``, ``DiscardReason.*.value``) are stable across locales — exports
carry the value, not the display text, so they merge cleanly across
multi-language deployments.

Distribution (min_submitted) is intentionally omitted — it is an operational
setting controlled by AnnotationSettings.workspaces and applied at
dataset creation time.
"""

import functools
import json
from importlib.resources import files
from string import Template
from typing import Any

import argilla as rg

from pragmata.core.annotation.locales.loader import DISCARD_WIDGET_KEYS
from pragmata.core.annotation.locales.registry import CATALOGS, get_catalog
from pragmata.core.annotation.locales.types import Catalog, CatalogKind
from pragmata.core.schemas.annotation_task import DiscardReason, Locale, Task


class _DiscardTemplate(Template):
    """``string.Template`` with ``@@`` delimiter — JS body uses ``$nuxt``/``$i18n``."""

    delimiter = "@@"


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


def _localised_labels(catalog: Catalog, task: Task, question: str, values: list[str]) -> dict[str, str]:
    """Build the value→display-text map Argilla's LabelQuestion accepts as ``labels=``.

    The stored value (dict key) is the machine identifier — what lands in
    exports. The display text (dict value) is what the annotator sees in
    the UI. Catalog key: ``(task, "label", "<question>.<value>")``.
    """
    return {value: catalog[(task, "label", f"{question}.{value}")] for value in values}


def _discard_questions(task: Task, catalog: Catalog) -> list[rg.LabelQuestion | rg.TextQuestion]:
    discard_reason_values = [r.value for r in DiscardReason]
    return [
        rg.LabelQuestion(
            name="discard_reason",
            title=catalog[(task, "question", "discard_reason")],
            labels=_localised_labels(catalog, task, "discard_reason", discard_reason_values),
            required=False,
        ),
        rg.TextQuestion(
            name="discard_notes",
            title=catalog[(task, "question", "discard_notes")],
            required=False,
        ),
    ]


def _discard_i18n_payload_for_locale(loc: Locale, task: Task) -> dict[str, Any]:
    """Per-locale block of strings the discard widget JS reads at runtime.

    Includes the widget's own chrome strings, the discard-reason option
    list (value + display text), and the two helper-question titles the
    widget uses for ``aria-label`` matching when scoping the native
    Argilla cards it hides.
    """
    catalog = CATALOGS[loc]
    payload: dict[str, Any] = {key: catalog[(task, "widget", f"discard.{key}")] for key in DISCARD_WIDGET_KEYS}
    payload["discard_reason_title"] = catalog[(task, "question", "discard_reason")]
    payload["discard_notes_title"] = catalog[(task, "question", "discard_notes")]
    payload["reason_options"] = [
        {"value": r.value, "text": catalog[(task, "label", f"discard_reason.{r.value}")]} for r in DiscardReason
    ]
    return payload


def _render_discard_template(template_text: str, task: Task, dataset_locale: Locale) -> str:
    """Substitute the all-locales i18n payload into ``discard_flow.html``.

    The widget JS picks the active locale at runtime (Argilla's chrome
    locale, with a fallback chain), so we ship every supported locale's
    strings, ordering ``SUPPORTED_LOCALES`` with the dataset's creation
    locale first. That ordering is consulted when probing Argilla's
    ``aria-label`` attributes for the hidden helper-question cards.
    """
    locales_in_order = [dataset_locale] + [loc for loc in sorted(CATALOGS) if loc != dataset_locale]
    i18n_payload = {loc: _discard_i18n_payload_for_locale(loc, task) for loc in locales_in_order}
    return _DiscardTemplate(template_text).substitute(
        I18N_JSON=json.dumps(i18n_payload, ensure_ascii=False),
        SUPPORTED_LOCALES_JSON=json.dumps(locales_in_order),
        DEFAULT_LOCALE_JSON=json.dumps(dataset_locale),
    )


@functools.cache
def build_task_settings(locale: Locale = "en") -> dict[Task, rg.Settings]:
    """Build Argilla Settings for each annotation task, in the given locale.

    Deferred construction — call after an Argilla client is connected
    (or with a mock client in tests). Cached per locale after first call.
    """
    catalog = get_catalog(locale)
    template_text = files("pragmata.core.annotation").joinpath("collapsible_field.html").read_text(encoding="utf-8")
    discard_template = files("pragmata.core.annotation").joinpath("discard_flow.html").read_text(encoding="utf-8")

    # Fresh CustomField per task — FieldBase carries a `_dataset` attribute that
    # Argilla's Settings/Dataset plumbing mutates, so sharing one instance across
    # three rg.Settings risks cross-task coupling on future SDK changes.
    def discard_field(task: Task) -> rg.CustomField:
        return rg.CustomField(
            name="discard_flow",
            title=catalog[(task, "field", "discard_flow")],
            template=_render_discard_template(discard_template, task, locale),
            advanced_mode=True,
            required=False,
        )

    def t(task: Task, kind: CatalogKind, name: str) -> str:
        return catalog[(task, kind, name)]

    def _yes_no(task: Task, question: str) -> dict[str, str]:
        return _localised_labels(catalog, task, question, ["yes", "no"])

    return {
        Task.RETRIEVAL: rg.Settings(
            fields=[
                rg.TextField(name="query", title=t(Task.RETRIEVAL, "field", "query"), required=True),
                rg.TextField(name="chunk", title=t(Task.RETRIEVAL, "field", "chunk"), required=True),
                _collapsible_field("generated_answer", t(Task.RETRIEVAL, "field", "generated_answer"), template_text),
                discard_field(Task.RETRIEVAL),
            ],
            questions=[
                rg.LabelQuestion(
                    name="topically_relevant",
                    title=t(Task.RETRIEVAL, "question", "topically_relevant"),
                    labels=_yes_no(Task.RETRIEVAL, "topically_relevant"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="evidence_sufficient",
                    title=t(Task.RETRIEVAL, "question", "evidence_sufficient"),
                    labels=_yes_no(Task.RETRIEVAL, "evidence_sufficient"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="misleading",
                    title=t(Task.RETRIEVAL, "question", "misleading"),
                    labels=_yes_no(Task.RETRIEVAL, "misleading"),
                    required=True,
                ),
                rg.TextQuestion(name="notes", title=t(Task.RETRIEVAL, "question", "notes"), required=False),
                *_discard_questions(Task.RETRIEVAL, catalog),
            ],
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
                rg.TermsMetadataProperty("chunk_id", visible_for_annotators=False),
                rg.TermsMetadataProperty("doc_id", visible_for_annotators=False),
                rg.IntegerMetadataProperty("chunk_rank", min=1, visible_for_annotators=False),
            ],
            guidelines=t(Task.RETRIEVAL, "guidelines", ""),
        ),
        Task.GROUNDING: rg.Settings(
            fields=[
                rg.TextField(name="answer", title=t(Task.GROUNDING, "field", "answer"), required=True),
                rg.TextField(name="context_set", title=t(Task.GROUNDING, "field", "context_set"), required=True),
                _collapsible_field("query", t(Task.GROUNDING, "field", "query"), template_text),
                discard_field(Task.GROUNDING),
            ],
            questions=[
                rg.LabelQuestion(
                    name="support_present",
                    title=t(Task.GROUNDING, "question", "support_present"),
                    labels=_yes_no(Task.GROUNDING, "support_present"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="unsupported_claim_present",
                    title=t(Task.GROUNDING, "question", "unsupported_claim_present"),
                    labels=_yes_no(Task.GROUNDING, "unsupported_claim_present"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="contradicted_claim_present",
                    title=t(Task.GROUNDING, "question", "contradicted_claim_present"),
                    labels=_yes_no(Task.GROUNDING, "contradicted_claim_present"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="source_cited",
                    title=t(Task.GROUNDING, "question", "source_cited"),
                    labels=_yes_no(Task.GROUNDING, "source_cited"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="fabricated_source",
                    title=t(Task.GROUNDING, "question", "fabricated_source"),
                    labels=_yes_no(Task.GROUNDING, "fabricated_source"),
                    required=True,
                ),
                rg.TextQuestion(name="notes", title=t(Task.GROUNDING, "question", "notes"), required=False),
                *_discard_questions(Task.GROUNDING, catalog),
            ],
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
            ],
            guidelines=t(Task.GROUNDING, "guidelines", ""),
        ),
        Task.GENERATION: rg.Settings(
            fields=[
                rg.TextField(name="query", title=t(Task.GENERATION, "field", "query"), required=True),
                rg.TextField(name="answer", title=t(Task.GENERATION, "field", "answer"), required=True),
                _collapsible_field("context_set", t(Task.GENERATION, "field", "context_set"), template_text),
                discard_field(Task.GENERATION),
            ],
            questions=[
                rg.LabelQuestion(
                    name="proper_action",
                    title=t(Task.GENERATION, "question", "proper_action"),
                    labels=_yes_no(Task.GENERATION, "proper_action"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="response_on_topic",
                    title=t(Task.GENERATION, "question", "response_on_topic"),
                    labels=_yes_no(Task.GENERATION, "response_on_topic"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="helpful",
                    title=t(Task.GENERATION, "question", "helpful"),
                    labels=_yes_no(Task.GENERATION, "helpful"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="incomplete",
                    title=t(Task.GENERATION, "question", "incomplete"),
                    labels=_yes_no(Task.GENERATION, "incomplete"),
                    required=True,
                ),
                rg.LabelQuestion(
                    name="unsafe_content",
                    title=t(Task.GENERATION, "question", "unsafe_content"),
                    labels=_yes_no(Task.GENERATION, "unsafe_content"),
                    required=True,
                ),
                rg.TextQuestion(name="notes", title=t(Task.GENERATION, "question", "notes"), required=False),
                *_discard_questions(Task.GENERATION, catalog),
            ],
            metadata=[
                rg.TermsMetadataProperty("record_uuid", visible_for_annotators=False),
                rg.TermsMetadataProperty("language", visible_for_annotators=False),
            ],
            guidelines=t(Task.GENERATION, "guidelines", ""),
        ),
    }
