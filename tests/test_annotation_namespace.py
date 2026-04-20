"""Tests for annotation subpackage importability."""

import pytest

pytestmark = pytest.mark.packaging


def test_annotation_subpackage_importable() -> None:
    """Annotation subpackage is importable after pip install pragmata[annotation]."""
    from pragmata import annotation

    assert annotation is not None
