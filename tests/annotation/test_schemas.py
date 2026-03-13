"""Unit tests for Argilla dataset task config constants."""

import argilla as rg

from chatboteval.api.annotation_task_config import (
    DATASET_NAMES,
    TASK1_RETRIEVAL_SETTINGS,
    TASK2_GROUNDING_SETTINGS,
    TASK3_GENERATION_SETTINGS,
    TASK_SETTINGS,
)
from chatboteval.core.schemas.annotation_task import Task


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
        names = _field_names(TASK1_RETRIEVAL_SETTINGS)
        assert "query" in names
        assert "chunk" in names
        assert "generated_answer" in names

    def test_query_and_chunk_are_textfields(self):
        assert isinstance(_get_field(TASK1_RETRIEVAL_SETTINGS, "query"), rg.TextField)
        assert isinstance(_get_field(TASK1_RETRIEVAL_SETTINGS, "chunk"), rg.TextField)

    def test_generated_answer_is_customfield(self):
        field = _get_field(TASK1_RETRIEVAL_SETTINGS, "generated_answer")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True

    def test_customfield_has_details_summary(self):
        field = _get_field(TASK1_RETRIEVAL_SETTINGS, "generated_answer")
        assert "<details" in field.template
        assert "<summary>" in field.template

    def test_questions(self):
        names = _question_names(TASK1_RETRIEVAL_SETTINGS)
        assert "topically_relevant" in names
        assert "evidence_sufficient" in names
        assert "misleading" in names
        assert "notes" in names

    def test_label_questions_are_binary(self):
        for qname in ("topically_relevant", "evidence_sufficient", "misleading"):
            q = _get_question(TASK1_RETRIEVAL_SETTINGS, qname)
            assert isinstance(q, rg.LabelQuestion)
            assert set(q.labels) == {"yes", "no"}
            assert q.required is True

    def test_notes_question(self):
        q = _get_question(TASK1_RETRIEVAL_SETTINGS, "notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False

    def test_guidelines_non_empty(self):
        assert TASK1_RETRIEVAL_SETTINGS.guidelines
        assert len(TASK1_RETRIEVAL_SETTINGS.guidelines) > 0


class TestTask2GroundingSettings:
    def test_fields_present(self):
        names = _field_names(TASK2_GROUNDING_SETTINGS)
        assert "answer" in names
        assert "context_set" in names
        assert "query" in names

    def test_answer_and_context_set_are_textfields(self):
        assert isinstance(_get_field(TASK2_GROUNDING_SETTINGS, "answer"), rg.TextField)
        assert isinstance(_get_field(TASK2_GROUNDING_SETTINGS, "context_set"), rg.TextField)

    def test_query_is_customfield(self):
        field = _get_field(TASK2_GROUNDING_SETTINGS, "query")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True

    def test_customfield_has_details_summary(self):
        field = _get_field(TASK2_GROUNDING_SETTINGS, "query")
        assert "<details" in field.template
        assert "<summary>" in field.template

    def test_questions(self):
        names = _question_names(TASK2_GROUNDING_SETTINGS)
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
            q = _get_question(TASK2_GROUNDING_SETTINGS, qname)
            assert isinstance(q, rg.LabelQuestion)
            assert set(q.labels) == {"yes", "no"}
            assert q.required is True

    def test_notes_question(self):
        q = _get_question(TASK2_GROUNDING_SETTINGS, "notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False

    def test_guidelines_non_empty(self):
        assert TASK2_GROUNDING_SETTINGS.guidelines


class TestTask3GenerationSettings:
    def test_fields_present(self):
        names = _field_names(TASK3_GENERATION_SETTINGS)
        assert "query" in names
        assert "answer" in names
        assert "context_set" in names

    def test_query_and_answer_are_textfields(self):
        assert isinstance(_get_field(TASK3_GENERATION_SETTINGS, "query"), rg.TextField)
        assert isinstance(_get_field(TASK3_GENERATION_SETTINGS, "answer"), rg.TextField)

    def test_context_set_is_customfield(self):
        field = _get_field(TASK3_GENERATION_SETTINGS, "context_set")
        assert isinstance(field, rg.CustomField)
        assert field.advanced_mode is True

    def test_customfield_has_details_summary(self):
        field = _get_field(TASK3_GENERATION_SETTINGS, "context_set")
        assert "<details" in field.template
        assert "<summary>" in field.template

    def test_questions(self):
        names = _question_names(TASK3_GENERATION_SETTINGS)
        for label in ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"):
            assert label in names
        assert "notes" in names

    def test_label_questions_are_binary(self):
        for qname in ("proper_action", "response_on_topic", "helpful", "incomplete", "unsafe_content"):
            q = _get_question(TASK3_GENERATION_SETTINGS, qname)
            assert isinstance(q, rg.LabelQuestion)
            assert set(q.labels) == {"yes", "no"}
            assert q.required is True

    def test_notes_question(self):
        q = _get_question(TASK3_GENERATION_SETTINGS, "notes")
        assert isinstance(q, rg.TextQuestion)
        assert q.required is False

    def test_guidelines_non_empty(self):
        assert TASK3_GENERATION_SETTINGS.guidelines


class TestTaskSettingsLookup:
    def test_task_settings_covers_all_tasks(self):
        assert set(TASK_SETTINGS.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}

    def test_task_settings_values(self):
        assert TASK_SETTINGS[Task.RETRIEVAL] is TASK1_RETRIEVAL_SETTINGS
        assert TASK_SETTINGS[Task.GROUNDING] is TASK2_GROUNDING_SETTINGS
        assert TASK_SETTINGS[Task.GENERATION] is TASK3_GENERATION_SETTINGS


class TestDatasetNames:
    def test_dataset_names_covers_all_tasks(self):
        assert set(DATASET_NAMES.keys()) == {Task.RETRIEVAL, Task.GROUNDING, Task.GENERATION}

    def test_dataset_name_values(self):
        assert DATASET_NAMES[Task.RETRIEVAL] == "task1_retrieval"
        assert DATASET_NAMES[Task.GROUNDING] == "task2_grounding"
        assert DATASET_NAMES[Task.GENERATION] == "task3_generation"
