"""Shared loader for querygen checkpoint artifacts (read + drift detection)."""

import importlib.metadata
import json
from pathlib import Path
from typing import NamedTuple, TypeVar

from pydantic import BaseModel

ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


class DriftedField(NamedTuple):
    """One header field whose stored value no longer matches the current run.

    Attributes:
        field: Header attribute name that drifted.
        stored: Value recorded in the checkpoint (the "old" value).
        expected: Value the current run requires (the "new" value).
    """

    field: str
    stored: object
    expected: object


def read_checkpoint_artifact(
    *,
    path: Path,
    model_cls: type[ArtifactT],
    expected_header: dict[str, object],
) -> tuple[ArtifactT | None, list[DriftedField]]:
    """Load a checkpoint artifact and report any configuration drift.

    Args:
        path: Checkpoint file path.
        model_cls: Pydantic artifact schema to validate against.
        expected_header: Mapping of artifact attribute name -> value the current
            run requires. The running ``pragmata`` version is compared
            automatically in addition to these.

    Returns:
        A ``(artifact, drifted)`` pair:

        - ``(artifact, [])`` -- the file is present and every header value (and
          the running pragmata version) matches; the checkpoint is reusable.
        - ``(None, [])`` -- the file is absent (nothing to resume here).
        - ``(None, drifted)`` -- the file is present but one or more fields
          differ; ``drifted`` lists each as a :class:`DriftedField` (stored vs
          expected), including ``pragmata_version`` when the running version
          differs. The api layer fails fast on a non-empty ``drifted`` unless
          the run opts into ``force``.

    Raises:
        json.JSONDecodeError: The file content is not valid JSON.
        UnicodeDecodeError: The file is not valid UTF-8 (e.g. torn mid-write).
        pydantic.ValidationError: The JSON does not match ``model_cls``.

    These exceptions signal a damaged (not merely drifted) checkpoint; the api
    layer treats them as a self-heal case and recomputes silently, distinct from
    the deliberate-config-change drift reported via the return value.
    """
    if not path.exists():
        return None, []

    artifact = model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    # The running pragmata version is always part of the expected header: a
    # version change can alter prompt templates or schemas without touching the
    # spec, so a checkpoint written by another version must not be reused.
    expected = {**expected_header, "pragmata_version": importlib.metadata.version("pragmata")}
    drifted = [
        DriftedField(field=attribute, stored=getattr(artifact, attribute), expected=value)
        for attribute, value in expected.items()
        if getattr(artifact, attribute) != value
    ]

    if drifted:
        return None, drifted
    return artifact, []
