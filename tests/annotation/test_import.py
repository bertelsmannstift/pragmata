"""Tests for annotation subpackage importability."""


def test_annotation_subpackage_importable() -> None:
    """Annotation subpackage is importable after pip install chatboteval[annotation]."""
    from chatboteval import annotation

    assert annotation is not None
