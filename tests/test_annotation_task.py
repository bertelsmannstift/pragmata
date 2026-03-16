"""Tests for the Task enum."""

import pytest

from chatboteval.core.schemas.annotation_task import Task


def test_task_values():
    """Task enum members have expected string values."""
    assert Task.RETRIEVAL == "retrieval"
    assert Task.GROUNDING == "grounding"
    assert Task.GENERATION == "generation"


def test_task_from_string():
    """Task can be constructed from its string value."""
    assert Task("retrieval") is Task.RETRIEVAL


def test_task_invalid_raises():
    """Invalid string raises ValueError."""
    with pytest.raises(ValueError):
        Task("invalid")


def test_task_has_three_members():
    """Task enum has exactly three members."""
    assert len(list(Task)) == 3
