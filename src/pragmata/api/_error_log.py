"""Scoped file handler that writes ERROR+ to base_dir/annotation/errors.log."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


@contextmanager
def error_log(base_dir: Path) -> Generator[None]:
    """Attach a file handler for the duration of a block, then clean up."""
    log_dir = Path(base_dir).expanduser().resolve() / "annotation"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "errors.log", delay=True)
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter(_FMT))
    root = logging.getLogger("pragmata")
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
