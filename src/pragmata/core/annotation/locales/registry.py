"""Catalog registry — auto-discovers locale YAML files at import.

A deployment adds a locale by dropping ``<code>.yaml`` into this directory;
the registry picks it up at import with no Python edit required. The locale
code is the file stem (``en.yaml`` → ``"en"``).
"""

from pathlib import Path

from pragmata.core.annotation.locales.loader import load_catalog
from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import Locale

_LOCALES_DIR = Path(__file__).parent

CATALOGS: dict[Locale, Catalog] = {
    yaml_file.stem: load_catalog(yaml_file) for yaml_file in sorted(_LOCALES_DIR.glob("*.yaml"))
}


def get_catalog(locale: Locale) -> Catalog:
    """Return the display-string catalog for a locale.

    Raises ``KeyError`` if the locale has no registered catalog.
    """
    return CATALOGS[locale]
