"""Catalog registry: auto-discovers locale YAML files at import.

Bundled catalogs are loaded eagerly from this package directory at import
time, keyed by file stem (``en.yaml`` -> ``"en"``). A deployment can
additionally register catalogs from any user-provided directory by
calling :func:`register_catalog_dir` (typically wired in from the api/
layer after settings resolution). User-provided files override bundled
entries on stem collision, matching the layering convention of standard
i18n stacks (gettext, i18next).
"""

from pathlib import Path

from pragmata.core.annotation.locales.loader import load_catalog
from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import Locale


def _load_dir(directory: Path) -> dict[Locale, Catalog]:
    return {yaml_file.stem: load_catalog(yaml_file) for yaml_file in sorted(directory.glob("*.yaml"))}


_LOCALES_DIR = Path(__file__).parent

CATALOGS: dict[Locale, Catalog] = _load_dir(_LOCALES_DIR)


def register_catalog_dir(directory: Path) -> None:
    """Layer ``*.yaml`` catalogs from ``directory`` over the bundled set.

    Idempotent: same directory registered twice yields the same end state.
    Clears the downstream ``build_task_settings`` cache so a subsequent
    locale build sees the updated catalog (otherwise a second
    ``import_records`` call in the same process with a different
    ``locale_catalog_dir`` would silently reuse stale ``rg.Settings``).
    """
    from pragmata.core.annotation.argilla_task_definitions import build_task_settings

    CATALOGS.update(_load_dir(directory))
    build_task_settings.cache_clear()


def get_catalog(locale: Locale) -> Catalog:
    """Return the display-string catalog for a locale.

    Raises ``ValueError`` if the locale has no registered catalog (bundled
    or user-provided via :func:`register_catalog_dir`).
    """
    if locale not in CATALOGS:
        raise ValueError(f"Unknown locale: {locale!r}. Supported: {sorted(CATALOGS)}")
    return CATALOGS[locale]
