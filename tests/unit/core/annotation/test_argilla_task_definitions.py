"""Unit tests for Argilla dataset task config constants."""

import argilla as rg
import pytest

from pragmata.core.annotation.argilla_task_definitions import (
    DATASET_NAMES,
    build_task_settings,
)
from pragmata.core.schemas.annotation_task import DiscardReason, Task
from pragmata.core.settings.annotation_settings import AnnotationSettings

_TASK_SETTINGS = build_task_settings(AnnotationSettings())
_RETRIEVAL = _TASK_SETTINGS[Task.RETRIEVAL]
_GROUNDING = _TASK_SETTINGS[Task.GROUNDING]
_GENERATION = _TASK_SETTINGS[Task.GENERATION]


def _field_names(settings: rg.Settings) -> list[str]:
    return [f.name for f in settings.fields]


def _question_names(settings: rg.Settings) -> list[str]:
    return [q.name for q in settings.questions]


def _get_field(settings: rg.Settings, name: str):
    return next((f for f in settings.fields if f.name == name), None)


def _get_question(settings: rg.Settings, name: str):
    return next((q for q in settings.questions if q.name == name), None)


class TestTask1RetrievalSettings:
    def test_fields_present(self):
        names = _field_names(_RETRIEVAL)
        assert "query" in names
        assert "chunk" in names
        assert "generated_answer" in names

    def test_query_and_chunk_are_textfields(self):
        assert isinstance(_get_field(_RETRIEVAL, "query"), rg.TextField)
        assert isinstance(_get_field(_RETRIEVAL, "chunk"), rg.TextField)

    def test_generated_answer_is_customfield(self):
        field = _get_field(_RETRIEVAL, "generated_answer")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True

    def test_customfield_has_details_summary(self):
        field = _get_field(_RETRIEVAL, "generated_answer")
        assert "<details" in field.template
        assert "<summary>" in field.template

    def test_questions(self):
        names = _question_names(_RETRIEVAL)
        assert "topically_relevant" in names
        assert "evidence_sufficient" in names
        assert "misleading" in names
        assert "notes" in names

    def test_label_questions_are_binary(self):
        for qname in ("topically_relevant", "evidence_sufficient", "misleading"):
            q = _get_question(_RETRIEVAL, qname)
            assert isinstance(q, rg.LabelQuestion)
            assert set(q.labels) == {"yes", "no"}
            assert q.required is True

    def test_notes_question(self):
        q = _get_question(_RETRIEVAL, "notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False

    def test_guidelines_non_empty(self):
        assert _RETRIEVAL.guidelines
        assert len(_RETRIEVAL.guidelines) > 0


class TestTask2GroundingSettings:
    def test_fields_present(self):
        names = _field_names(_GROUNDING)
        assert "answer" in names
        assert "context_set" in names
        assert "query" in names

    def test_answer_and_context_set_are_textfields(self):
        assert isinstance(_get_field(_GROUNDING, "answer"), rg.TextField)
        assert isinstance(_get_field(_GROUNDING, "context_set"), rg.TextField)

    def test_query_is_customfield(self):
        field = _get_field(_GROUNDING, "query")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True

    def test_customfield_has_details_summary(self):
        field = _get_field(_GROUNDING, "query")
        assert "<details" in field.template
        assert "<summary>" in field.template

    def test_questions(self):
        names = _question_names(_GROUNDING)
        for label in (
            "support_present",
            "unsupported_claim_present",
            "contradicted_claim_present",
            "source_cited",
            "fabricated_source",
        ):
            assert label in names
        assert "notes" in names

    def test_label_questions_are_binary(self):
        labels = (
            "support_present",
            "unsupported_claim_present",
            "contradicted_claim_present",
            "source_cited",
            "fabricated_source",
        )
        for qname in labels:
            q = _get_question(_GROUNDING, qname)
            assert isinstance(q, rg.LabelQuestion)
            assert set(q.labels) == {"yes", "no"}
            assert q.required is True

    def test_notes_question(self):
        q = _get_question(_GROUNDING, "notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False

    def test_guidelines_non_empty(self):
        assert _GROUNDING.guidelines


class TestTask3GenerationSettings:
    def test_fields_present(self):
        names = _field_names(_GENERATION)
        assert "query" in names
        assert "answer" in names
        assert "context_set" in names

    def test_query_and_answer_are_textfields(self):
        assert isinstance(_get_field(_GENERATION, "query"), rg.TextField)
        assert isinstance(_get_field(_GENERATION, "answer"), rg.TextField)

    def test_context_set_is_customfield(self):
        field = _get_field(_GENERATION, "context_set")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True

    def test_customfield_has_details_summary(self):
        field = _get_field(_GENERATION, "context_set")
        assert "<details" in field.template
        assert "<summary>" in field.template

    def test_questions(self):
        names = _question_names(_GENERATION)
        for label in ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"):
            assert label in names
        assert "notes" in names

    def test_label_questions_are_binary(self):
        for qname in ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"):
            q = _get_question(_GENERATION, qname)
            assert isinstance(q, rg.LabelQuestion)
            assert set(q.labels) == {"yes", "no"}
            assert q.required is True

    def test_notes_question(self):
        q = _get_question(_GENERATION, "notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False

    def test_guidelines_non_empty(self):
        assert _GENERATION.guidelines


@pytest.mark.parametrize(
    "settings",
    [_RETRIEVAL, _GROUNDING, _GENERATION],
    ids=["retrieval", "grounding", "generation"],
)
class TestDiscardContract:
    """Discard reason/notes questions are a shared contract across all task types."""

    def test_discard_reason_question(self, settings):
        q = _get_question(settings, "discard_reason")
        assert isinstance(q, rg.LabelQuestion)
        assert q.required is False
        assert set(q.labels) == {r.value for r in DiscardReason}

    def test_discard_notes_question(self, settings):
        q = _get_question(settings, "discard_notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False


class TestMetadataProperties:
    def test_retrieval_metadata(self):
        meta = [m.name for m in _RETRIEVAL.metadata]
        assert meta == ["record_uuid", "language", "chunk_id", "doc_id", "chunk_rank"]

    def test_grounding_metadata(self):
        meta = [m.name for m in _GROUNDING.metadata]
        assert meta == ["record_uuid", "language"]

    def test_generation_metadata(self):
        meta = [m.name for m in _GENERATION.metadata]
        assert meta == ["record_uuid", "language"]


class TestTaskSettingsLookup:
    def test_task_settings_covers_all_tasks(self):
        assert set(_TASK_SETTINGS.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}

    def test_task_settings_values(self):
        assert _TASK_SETTINGS[Task.RETRIEVAL] is _RETRIEVAL
        assert _TASK_SETTINGS[Task.GROUNDING] is _GROUNDING
        assert _TASK_SETTINGS[Task.GENERATION] is _GENERATION


class TestDatasetNames:
    def test_dataset_names_covers_all_tasks(self):
        assert set(DATASET_NAMES.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}

    def test_dataset_name_values(self):
        assert DATASET_NAMES[Task.RETRIEVAL] == "retrieval"
        assert DATASET_NAMES[Task.GROUNDING] == "grounding"
        assert DATASET_NAMES[Task.GENERATION] == "generation"


class TestDiscardFlowI18nPayload:
    """Guard against drift between DiscardReason enum and the discard widget payload.

    Option values are now injected via the i18n JSON baked into the rendered
    ``discard_flow.html`` template at build time, not hardcoded in the HTML.
    """

    def test_rendered_template_contains_all_enum_values(self):
        # Option values are injected via the i18n JSON payload at build time,
        # not hardcoded in the HTML. Assert each DiscardReason.value appears
        # in the rendered output of the discard CustomField.
        discard_field = next(f for f in _RETRIEVAL.fields if f.name == "discard_flow")
        rendered = discard_field.template
        for reason in DiscardReason:
            assert f'"value": "{reason.value}"' in rendered, (
                f"DiscardReason.{reason.name} value {reason.value!r} missing from rendered widget"
            )


class TestYesNoCatalogSync:
    """Guard the yes/no question set against drift between the two encodings.

    The set lives in two places: ``rg.LabelQuestion`` definitions in
    ``argilla_task_definitions`` and ``_YES_NO_QUESTIONS_BY_TASK`` in
    ``loader.py`` (which drives ``.yes`` / ``.no`` catalog keys). Both sides
    must enumerate the same yes/no question set per task; otherwise label
    lookup silently breaks at dataset creation.
    """

    @pytest.mark.parametrize(
        ("task", "settings"),
        [
            (Task.RETRIEVAL, _RETRIEVAL),
            (Task.GROUNDING, _GROUNDING),
            (Task.GENERATION, _GENERATION),
        ],
        ids=["retrieval", "grounding", "generation"],
    )
    def test_yes_no_questions_match_loader_map(self, task, settings):
        from pragmata.core.annotation.locales.loader import _YES_NO_QUESTIONS_BY_TASK

        yes_no_question_names = {
            q.name for q in settings.questions if isinstance(q, rg.LabelQuestion) and set(q.labels) == {"yes", "no"}
        }
        assert yes_no_question_names == set(_YES_NO_QUESTIONS_BY_TASK[task])


class TestCatalogDrivesRenderedOutput:
    """The catalog is the source of truth for user-visible strings.

    Swapping a catalog entry must change the rendered title; identities
    (``name=``) and label values stay frozen so exports merge cleanly.
    """

    def test_catalog_drives_field_title(self, monkeypatch):
        from pragmata.core.annotation.locales.registry import CATALOGS

        sentinel = "SENTINEL_TITLE"
        stub = dict(CATALOGS["en"])
        stub[(Task.RETRIEVAL, "field", "query")] = sentinel
        monkeypatch.setitem(CATALOGS, "en", stub)
        rendered = build_task_settings(AnnotationSettings(), "en")[Task.RETRIEVAL]
        query_field = _get_field(rendered, "query")
        assert query_field is not None
        assert query_field.title == sentinel


class TestConstraintsField:
    """The constraints_panel CustomField wires LOGICAL_CONSTRAINTS into the annotator UI."""

    def test_present_on_every_task(self):
        # Even Generation (no constraints) gets the field: every record carries a
        # constraints_panel placeholder value, and Argilla rejects records
        # whose field keys don't all exist on the dataset.
        for settings in (_RETRIEVAL, _GROUNDING, _GENERATION):
            assert _get_field(settings, "constraints_panel") is not None

    def test_generation_widget_payload_has_no_constraints(self):
        # Generation has no constraints — the widget should serialise an empty
        # constraints array, so it evaluates to no hits and stays hidden.
        template = _get_field(_GENERATION, "constraints_panel").template
        assert "var CONSTRAINTS = [];" in template

    def test_is_advanced_custom_field(self):
        field = _get_field(_RETRIEVAL, "constraints_panel")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True
        assert field.required is False

    def test_template_substitutions_are_resolved(self):
        # The widget receives both placeholders — verify nothing was left
        # unsubstituted (would be a hard-to-debug runtime JS error).
        for settings in (_RETRIEVAL, _GROUNDING):
            template = _get_field(settings, "constraints_panel").template
            assert "$CONSTRAINTS_JSON" not in template
            assert "$QUESTION_TITLES_JSON" not in template

    def test_template_contains_constraint_messages(self):
        # Constraint messages must round-trip into the widget so the annotator sees them.
        from pragmata.core.annotation.logical_constraints import LOGICAL_CONSTRAINTS

        for task, settings in ((Task.RETRIEVAL, _RETRIEVAL), (Task.GROUNDING, _GROUNDING)):
            template = _get_field(settings, "constraints_panel").template
            for constraint in LOGICAL_CONSTRAINTS[task]:
                # First few words is enough — full message has punctuation/escaping noise.
                assert constraint.message.split(".")[0][:30] in template

    def test_template_carries_question_titles_used_by_constraints(self):
        # Aria-label probing relies on these titles matching the dataset's questions.
        from pragmata.core.annotation.logical_constraints import LOGICAL_CONSTRAINTS

        for task, settings in ((Task.RETRIEVAL, _RETRIEVAL), (Task.GROUNDING, _GROUNDING)):
            template = _get_field(settings, "constraints_panel").template
            for constraint in LOGICAL_CONSTRAINTS[task]:
                for qname in (constraint.when_question, constraint.then_question):
                    title = _get_question(settings, qname).title
                    assert title in template

    def test_field_is_below_content_above_discard(self):
        # The panel should sit right before the discard field so it's adjacent
        # to the submit action area.
        for settings in (_RETRIEVAL, _GROUNDING):
            names = _field_names(settings)
            assert names.index("constraints_panel") == names.index("discard_flow") - 1
