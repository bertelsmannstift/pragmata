"""Locale-specific display strings for Argilla dataset titles, questions, guidelines.

Catalogs are keyed by ``(task, kind, name)`` where ``kind`` is one of
``"field"``, ``"question"``, ``"guidelines"``, ``"label"``, ``"widget"``.
Identities (``name=``) and label values are not translated — only display text.

Adding a locale:
1. Create ``<locale>.py`` in this package exporting ``CATALOG: Catalog``,
   reusing :func:`pragmata.core.annotation.locales.structure.build_programmatic_entries`
   for the locale-invariant label/widget rows.
2. Add the locale to :class:`pragmata.core.schemas.annotation_task.Locale`.
3. Register the catalog in :data:`pragmata.core.annotation.locales.registry.CATALOGS`.
"""
