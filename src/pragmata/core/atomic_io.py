"""Atomic file-write helper shared across pragmata subsystems."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO
from uuid import uuid4


@contextmanager
def atomic_write_text(path: Path) -> Iterator[TextIO]:
    """Yield a text handle whose contents replace ``path`` atomically on success.

    Writes to a uniquified temp file in the same directory and ``Path.replace``s
    it into place when the ``with`` block exits cleanly; on any exception the
    temp file is removed and ``path`` is left untouched. The temp name carries
    the PID and a random suffix so concurrent writers to the same target cannot
    clobber each other's temp file.

    No fsync (matches the repo's existing atomic-write idiom): a torn or
    unflushed file is never the result of a successful rename, and callers that
    validate-on-read treat a corrupt final file as drift and recompute.

    The handle is opened in text mode with ``encoding="utf-8"`` and
    ``newline=""`` (the latter so callers writing CSV via :mod:`csv` get correct
    line endings; harmless for JSON/text). The parent directory must already
    exist.

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
