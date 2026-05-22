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


class TestDiscardFlowHtmlEnumSync:
    """Guard against drift between DiscardReason enum and discard_flow.html options."""

    def test_html_option_values_match_enum(self):
        import re
        from importlib.resources import files

        html = files("pragmata.core.annotation").joinpath("discard_flow.html").read_text(encoding="utf-8")
        option_values = set(re.findall(r'<option value="([^"]+)"', html))
        option_values.discard("")  # ignore placeholder "-- select --" option
        assert option_values == {r.value for r in DiscardReason}


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
            assert "@@CONSTRAINTS_JSON" not in template
            assert "@@QUESTION_TITLES_JSON" not in template

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


class TestConstraintsFieldSeverityWireUp:
    """Widget receives the workspace-resolved severity for each constraint."""

    def _payload_severities(self, template: str) -> dict[str, str]:
        import json
        import re

        match = re.search(r"var CONSTRAINTS = (\[.*?\]);", template, re.DOTALL)
        assert match, "CONSTRAINTS JSON not found in template"
        return {c["constraint_id"]: c["severity"] for c in json.loads(match.group(1))}

    def test_no_overrides_uses_deployment_defaults(self):
        settings = AnnotationSettings()
        task_settings = build_task_settings(settings)
        for task in (Task.RETRIEVAL, Task.GROUNDING):
            severities = self._payload_severities(_get_field(task_settings[task], "constraints_panel").template)
            for constraint_id, sev in severities.items():
                assert sev == settings.constraint_severity[constraint_id]

    def test_workspace_override_reaches_widget(self):
        from pragmata.core.settings.annotation_settings import TaskSettings, WorkspaceSettings

        settings = AnnotationSettings(
            workspaces={
                "retrieval": WorkspaceSettings(
                    constraint_severity={"evidence_requires_relevance": "warn"},
                    tasks={Task.RETRIEVAL: TaskSettings()},
                ),
                "grounding": WorkspaceSettings(tasks={Task.GROUNDING: TaskSettings()}),
                "generation": WorkspaceSettings(tasks={Task.GENERATION: TaskSettings()}),
            },
        )
        task_settings = build_task_settings(settings)
        severities = self._payload_severities(_get_field(task_settings[Task.RETRIEVAL], "constraints_panel").template)
        assert severities["evidence_requires_relevance"] == "warn"
