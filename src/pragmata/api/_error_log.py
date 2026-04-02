"""Scoped file handler that writes ERROR+ to a caller-supplied directory."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


@contextmanager
def error_log(log_dir: Path) -> Generator[None]:
    """Attach a file handler for the duration of a block, then clean up."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "errors.log", delay=True, encoding="utf-8")
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter(_FMT))
    root = logging.getLogger("pragmata")
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
