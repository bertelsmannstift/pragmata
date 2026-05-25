"""Locale-specific display strings for Argilla dataset titles, questions, guidelines.

Translations live as data in ``<locale>.yaml`` files alongside this module;
:mod:`registry` auto-discovers them at import time. Catalogs are keyed by
``(task, kind, name)`` where ``kind`` is one of ``"field"``, ``"question"``,
``"guidelines"``, ``"label"``. Identities (``name=``) and label values are
not translated — only display text.

Adding a locale:
1. Drop ``<code>.yaml`` in this directory (copy ``en.yaml`` and translate).
2. Run the catalog completeness test; it surfaces any missing or stray keys
   versus the EN reference.

No Python edit required — the file stem becomes the locale code.
"""
