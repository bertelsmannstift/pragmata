import pytest
from pydantic import ValidationError

from chatboteval.core.schemas.base import ContractModel, Task


class _Simple(ContractModel):
    x: int


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


def test_contract_model_frozen():
    m = _Simple(x=1)
    with pytest.raises(ValidationError):
        m.x = 2


def test_contract_model_extra_forbidden():
    with pytest.raises(ValidationError):
        _Simple(x=1, y=99)
