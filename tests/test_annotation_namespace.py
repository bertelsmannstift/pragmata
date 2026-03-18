"""Tests for annotation subpackage importability."""


def test_annotation_subpackage_importable() -> None:
    """Annotation subpackage is importable after pip install pragmata[annotation]."""
    from pragmata import annotation

    assert annotation is not None
