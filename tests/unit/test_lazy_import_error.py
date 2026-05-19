"""First-use error UX for the annotation extra.

Block D of the lifecycle migration (docs/design/config-and-settings.md §0):
when a user installs bare `pragmata` (no `[annotation]` extra) and accesses
something from `pragmata.annotation`, the failure mode should be a friendly
ImportError pointing at the fix, not a raw traceback from deep inside the
lazy loader.
"""

import importlib

import pragmata.annotation
import pytest


def test_getattr_wraps_missing_argilla(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing argilla becomes the design-doc 'install with' ImportError."""

    def fake_import_module(name: str) -> object:
        raise ImportError(f"No module named {name!r}", name="argilla")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        pragmata.annotation.__getattr__("setup")

    message = str(exc_info.value)
    assert "'argilla' is required for pragmata.annotation" in message
    assert "pip install 'pragmata[annotation]'" in message
    # Chain preserves the original cause so users / sentry can still see it.
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_getattr_wraps_missing_argilla_submodule(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure to import argilla.<submodule> is treated the same as argilla itself."""

    def fake_import_module(name: str) -> object:
        raise ImportError("No module named 'argilla.client'", name="argilla.client")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"pip install 'pragmata\[annotation\]'"):
        pragmata.annotation.__getattr__("setup")


def test_getattr_does_not_mask_unrelated_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-argilla ImportErrors surface verbatim — broken transitive deps aren't masked."""

    def fake_import_module(name: str) -> object:
        raise ImportError("No module named 'some_other_dep'", name="some_other_dep")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        pragmata.annotation.__getattr__("setup")

    message = str(exc_info.value)
    assert "some_other_dep" in message
    assert "pragmata[annotation]" not in message, (
        "non-argilla ImportError should not be wrapped with the install-extra hint"
    )


def test_getattr_unknown_attribute_still_raises_attribute_error() -> None:
    """Unknown attributes still raise AttributeError (not wrapped as ImportError)."""
    with pytest.raises(AttributeError, match="no attribute 'nonexistent_thing'"):
        pragmata.annotation.__getattr__("nonexistent_thing")
