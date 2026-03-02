"""Smoke tests for the packaging and invocation surface."""

import ast
import pathlib

from typer.testing import CliRunner

import chatboteval.cli
from chatboteval import get_version
from chatboteval.cli.app import app


def test_package_importable() -> None:
    """Smoke test: the installed package can be imported."""
    import chatboteval

    assert chatboteval is not None


def test_curated_symbols_exist() -> None:
    """Smoke: curated public symbols are accessible at the top level."""
    import chatboteval

    assert hasattr(chatboteval, "get_version")


def test_cli_help() -> None:
    """CLI smoke: chatboteval --help exits 0."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_version() -> None:
    """CLI wiring: chatboteval --version exits 0 and prints a version string."""
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert get_version() in result.output


def test_cli_does_not_import_core() -> None:
    """Boundary guard: cli layer must not directly import core."""
    cli_dir = pathlib.Path(chatboteval.cli.__file__).resolve().parent 
    assert cli_dir.exists(), f"CLI directory not found at {cli_dir}"
    for py_file in cli_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("chatboteval.core"), (
                    f"{py_file} imports from chatboteval.core directly — boundary violation"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("chatboteval.core"), (
                        f"{py_file} imports from chatboteval.core directly — boundary violation"
                    )
