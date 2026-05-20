"""Locale-specific display strings for Argilla dataset titles, questions, guidelines.

Catalogs are keyed by ``(task, kind, name)`` where ``kind`` is one of
``"field"``, ``"question"``, ``"guidelines"``. Identities (``name=``) and label
values are not translated — only display text.

Adding a locale:
1. Create ``<locale>.py`` in this package exporting ``CATALOG: Catalog``.
2. Mirror the exact key set used in :mod:`pragmata.core.annotation.locales.en`.
3. Add the locale to :class:`pragmata.core.schemas.annotation_task.Locale`.
4. Register the catalog in ``CATALOGS`` below.
"""

from pragmata.core.annotation.locales.de import CATALOG as _DE
from pragmata.core.annotation.locales.en import CATALOG as _EN
from pragmata.core.annotation.locales.types import Catalog, CatalogKey
from pragmata.core.schemas.annotation_task import Locale

CATALOGS: dict[Locale, Catalog] = {
    Locale.EN: _EN,
    Locale.DE: _DE,
}


def get_catalog(locale: Locale) -> Catalog:
    """Return the display-string catalog for a locale."""
    return CATALOGS[locale]


__all__ = ["CATALOGS", "Catalog", "CatalogKey", "get_catalog"]
