"""Unit tests for the lazy pragmata.annotation facade.

Laziness assertions run in a subprocess so sys.modules isolation is guaranteed
— by the time pytest reaches these tests, other suites have already loaded
pragmata.api.*, argilla, and most of core/annotation, so in-process sys.modules
checks cannot verify laziness.
"""

import importlib
import subprocess
import sys
import textwrap

import pytest

from pragmata.annotation import _LAZY
from pragmata.annotation import __all__ as facade_all

pytestmark = pytest.mark.packaging


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


_LEAK_CHECK = """
import sys
{import_stmt}
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
    raise SystemExit(f"eager imports leaked: {{leaks}}")
"""


@pytest.mark.parametrize(
    "import_stmt",
    [
        "import pragmata  # noqa: F401",
        "from pragmata import annotation  # noqa: F401",
        "import pragmata.annotation  # noqa: F401",
    ],
    ids=["import_pragmata", "from_pragmata_import_annotation", "import_pragmata_annotation"],
)
def test_facade_import_paths_do_not_load_argilla_or_api_modules(import_stmt: str) -> None:
    """All three reachability paths must stay lazy: no argilla, no api.annotation_*."""
    result = _run_isolated(_LEAK_CHECK.format(import_stmt=import_stmt))
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
    """Accessing an API-bound attribute resolves the backing module on first touch."""
    result = _run_isolated(
        """
        import sys
        import pragmata.annotation as facade
        assert "pragmata.api.annotation_setup" not in sys.modules, "api module loaded too early"
        setup = facade.setup
        assert "pragmata.api.annotation_setup" in sys.modules, "api module not loaded after access"
        assert callable(setup)
        """
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_attribute_is_cached_on_first_access() -> None:
    """__getattr__ writes the resolved value back into globals() for fast repeat access."""
    facade = importlib.import_module("pragmata.annotation")

    first = facade.setup
    second = facade.setup

    assert first is second
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


def test_all_and_lazy_keys_match() -> None:
    """__all__ and _LAZY must stay in sync — adding to one without the other is a bug."""
    assert set(facade_all) == set(_LAZY)
