"""Atomic file-write helper shared across pragmata subsystems."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO
from uuid import uuid4


@contextmanager
def atomic_write_text(path: Path) -> Iterator[TextIO]:
    """Atomically replace a text file after a successful write.

    Writes UTF-8 text to a unique temporary file in the target directory and
    renames it into place with ``Path.replace`` when the context exits cleanly.
    If the write fails, the temporary file is removed and the existing target is
    left untouched. The parent directory must already exist.

    The handle is opened with ``newline=""`` so callers writing CSV via the
    :mod:`csv` module get correct line endings (harmless for JSON/text); the
    temp name carries the PID and a random suffix so concurrent writers to the
    same target cannot clobber each other.

    Atomic replacement is provided for normal readers, but the file and parent
    directory are intentionally not fsynced: callers that validate-on-read treat
    a corrupt final file as drift and recompute.

    Args:
        path: Final destination path; replaced atomically on clean exit.

    Yields:
        A writable text file handle for the temp file.
    """
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            yield handle
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(data: dict, path: Path) -> None:
    """Atomically write ``data`` as indented JSON with a trailing newline to ``path``.

    Indented (``indent=2``) so persisted artifacts are human-scannable when
    inspected; round-trip reads validate semantically, so the formatting carries
    no functional meaning. Writes via :func:`atomic_write_text`.

    Args:
        data: JSON-serialisable mapping to persist.
        path: Final destination path; replaced atomically on clean exit.
    """
    with atomic_write_text(path) as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
