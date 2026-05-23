"""Catalog registry: auto-discovers locale YAML files at import.

Scope: bundled catalogs only. The registry scans ``*.yaml`` files in this
package directory at import time and keys them by file stem
(``en.yaml`` -> ``"en"``). Adding a locale upstream is therefore a
zero-Python-edit change: drop the YAML file alongside ``en.yaml`` and the
locale is registered automatically.

Adding a locale from outside the installed package (user-provided catalog
directory) is not yet supported; a deployment that needs a custom locale
or wants to override a bundled one currently has to vendor or contribute
the YAML upstream. Extending discovery to a configurable directory is
planned as a follow-up.
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
