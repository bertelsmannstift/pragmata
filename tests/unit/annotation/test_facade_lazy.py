"""Unit tests for the lazy pragmata.annotation facade.

Some tests run a subprocess so sys.modules isolation is guaranteed — by the
time pytest reaches these tests, other suites have already loaded
pragmata.api.*, argilla, and most of core/annotation, so in-process sys.modules
assertions cannot verify laziness.
"""

import importlib
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(autouse=True)
def _fresh_facade() -> None:
    """Force a fresh facade import per test so __getattr__ runs cleanly."""
    for name in list(sys.modules):
        if name == "pragmata.annotation" or name.startswith("pragmata.annotation."):
            del sys.modules[name]


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_facade_does_not_load_argilla_or_api_modules() -> None:
    """Importing the facade must not eagerly pull argilla-dependent modules."""
    result = _run_isolated(
        """
        import sys
        import pragmata.annotation  # noqa: F401
        leaks = [
            m for m in (
                "pragmata.api.annotation_setup",
                "pragmata.api.annotation_import",
                "pragmata.api.annotation_export",
                "pragmata.api.annotation_iaa",
                "pragmata.core.annotation.setup",
                "pragmata.core.annotation.export_runner",
                "argilla",
            )
            if m in sys.modules
        ]
        if leaks:
            raise SystemExit(f"eager imports leaked: {leaks}")
        """
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_accessing_task_does_not_trigger_argilla() -> None:
    """Plain schema/enum access should not pull argilla-dependent modules."""
    result = _run_isolated(
        """
        import sys
        import pragmata.annotation as facade
        _ = facade.Task
        if "argilla" in sys.modules:
            raise SystemExit("accessing Task pulled argilla")
        """
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_accessing_setup_loads_api_module() -> None:
    """Accessing an API-bound attribute resolves through importlib once."""
    facade = importlib.import_module("pragmata.annotation")

    setup = facade.setup

    assert "pragmata.api.annotation_setup" in sys.modules
    assert callable(setup)


def test_attribute_is_cached_on_first_access() -> None:
    """__getattr__ writes the resolved value back into globals() for fast repeat access."""
    facade = importlib.import_module("pragmata.annotation")

    first = facade.setup
    second = facade.setup

    assert first is second
    # Confirm it was cached on the module itself, not just by identity.
    assert "setup" in vars(facade)


def test_unknown_attribute_raises_attribute_error() -> None:
    """Lookups outside __all__ surface a clean AttributeError."""
    facade = importlib.import_module("pragmata.annotation")

    with pytest.raises(AttributeError, match="pragmata.annotation"):
        _ = facade.does_not_exist


def test_dir_returns_public_surface() -> None:
    """dir() returns the declared public surface for tooling/autocomplete."""
    facade = importlib.import_module("pragmata.annotation")

    expected = {
        "ExportResult",
        "IaaReport",
        "ImportResult",
        "SetupResult",
        "Task",
        "UserSpec",
        "compute_iaa",
        "export_annotations",
        "import_records",
        "setup",
        "teardown",
    }

    assert set(dir(facade)) == expected
