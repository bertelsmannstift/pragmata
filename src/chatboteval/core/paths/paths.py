"""Shared workspace path resolver."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self


def coerce_path(
    input_path: str | Path,
) -> Path:
    """Convert a string or Path input into a Path instance.

    Args:
        input_path: Path value provided as a string or `Path`.

    Returns:
        The input converted to a `Path` instance.

    Raises:
        TypeError: If `input_path` is neither `str` nor `Path`.
    """
    if isinstance(input_path, Path):
        return input_path
    if isinstance(input_path, str):
        return Path(input_path)

    raise TypeError(f"Expected path as str or Path, got {type(input_path).__name__}.")


@dataclass(frozen=True)
class WorkspacePaths:
    """Workspace path bundle.

    Attributes:
        base_dir: Workspace root stored as an expanded, absolute path.
    """

    base_dir: Path

    @classmethod
    def from_base_dir(
        cls,
        base_dir: str | Path,
    ) -> Self:
        """Create a workspace path bundle from a base directory.

        Args:
            base_dir: Workspace root as a `str` or `Path`.

        Returns:
            A `WorkspacePaths` instance with a canonicalized `base_dir`.
        """
        normalized_base_dir = coerce_path(base_dir).expanduser().resolve()
        return cls(base_dir=normalized_base_dir)

    def tool_root(
        self,
        tool: str,
    ) -> Path:
        """Return the root directory for a tool within the workspace.

        Args:
            tool: Tool name used as the direct child directory under `base_dir`.

        Returns:
            The tool root path.
        """
        return self.base_dir / tool

    def run_root(
        self,
        *,
        tool: str,
        run_id: str,
        runs_dirname: str = "runs",
    ) -> Path:
        """Return the root directory for a specific tool run.

        Args:
            tool: Tool name under the workspace.
            run_id: Unique run identifier.
            runs_dirname: Name of the intermediate runs directory.

        Returns:
            The run root path.
        """
        return self.tool_root(tool) / runs_dirname / run_id

    def under_base(
        self,
        input_path: str | Path,
    ) -> Path:
        """Resolve a path under the workspace root.

        Absolute paths are normalized as-is. Relative paths are resolved
        against `base_dir`.

        Args:
            input_path: Path to resolve.

        Returns:
            The resolved absolute path.
        """
        path_obj = coerce_path(input_path).expanduser()
        return (self.base_dir / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()
