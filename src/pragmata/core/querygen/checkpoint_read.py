"""Shared loader for querygen checkpoint artifacts (read + header validation)."""

import importlib.metadata
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


def read_checkpoint_artifact(
    *,
    path: Path,
    model_cls: type[ArtifactT],
    expected_header: dict[str, object],
) -> ArtifactT | None:
    """Load and validate a checkpoint artifact, rejecting incompatible ones.

    Args:
        path: Checkpoint file path.
        model_cls: Pydantic artifact schema to validate against.
        expected_header: Mapping of artifact attribute name -> value the current
            run requires. Any mismatch (including a differing running pragmata
            version, which is checked automatically) yields ``None``.

    Returns:
        The validated artifact, or ``None`` when the file is absent, the running
        ``pragmata`` version differs from the artifact's, or any
        ``expected_header`` value does not match -- i.e. the checkpoint belongs
        to an incompatible run and must be redone.

    Raises:
        json.JSONDecodeError: The file content is not valid JSON.
        UnicodeDecodeError: The file is not valid UTF-8 (e.g. torn mid-write).
        pydantic.ValidationError: The JSON does not match ``model_cls``.

    Callers (the api layer) catch these and treat the checkpoint as drift.
    """
    if not path.exists():
        return None

    artifact = model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    if artifact.pragmata_version != importlib.metadata.version("pragmata"):
        return None

    for attribute, expected_value in expected_header.items():
        if getattr(artifact, attribute) != expected_value:
            return None

    return artifact
