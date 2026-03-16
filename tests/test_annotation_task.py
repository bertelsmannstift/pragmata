"""Tests for the Task enum."""

import pytest

from chatboteval.core.schemas.annotation_task import Task


def test_task_values():
    assert Task.RETRIEVAL == "retrieval"
    assert Task.GROUNDING == "grounding"
    assert Task.GENERATION == "generation"


def test_task_from_string():
    assert Task("retrieval") is Task.RETRIEVAL


def test_task_invalid_raises():
    with pytest.raises(ValueError):
        Task("invalid")


def test_task_has_three_members():
    assert len(list(Task)) == 3
