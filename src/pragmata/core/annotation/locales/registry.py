"""Catalog registry — locale → display-string catalog."""

from pragmata.core.annotation.locales.en import CATALOG as _EN
from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import Locale

CATALOGS: dict[Locale, Catalog] = {
    Locale.EN: _EN,
}


def get_catalog(locale: Locale) -> Catalog:
    """Return the display-string catalog for a locale."""
    return CATALOGS[locale]
