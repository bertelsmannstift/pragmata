"""Unit tests for Argilla dataset task config constants."""

import argilla as rg

from pragmata.core.annotation.argilla_settings import (
    DATASET_NAMES,
    build_task_settings,
)
from pragmata.core.schemas.annotation_task import Task

_TASK_SETTINGS = build_task_settings()
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
        assert DATASET_NAMES[Task.RETRIEVAL] == "task_retrieval"
        assert DATASET_NAMES[Task.GROUNDING] == "task_grounding"
        assert DATASET_NAMES[Task.GENERATION] == "task_generation"
