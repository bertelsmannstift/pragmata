# Annotation locales

Locale-specific display strings for Argilla dataset titles, fields, questions,
labels, guidelines, and discard-widget chrome. One catalog per locale; one
registry maps `Locale` enum members to catalogs.

## Package layout

| File           | Purpose                                                              |
|----------------|----------------------------------------------------------------------|
| `types.py`     | `CatalogKind` / `CatalogKey` / `Catalog` type aliases                |
| `structure.py` | Locale-invariant structure: `_YES_NO_QUESTIONS_BY_TASK`, `DISCARD_WIDGET_KEYS`, `build_programmatic_entries()` |
| `registry.py`  | `CATALOGS` dict + `get_catalog(locale)` lookup                       |
| `en.py`        | English catalog (source-of-truth)                                    |
| `de.py`        | German catalog                                                       |
| `__init__.py`  | Package docstring only — import from the specific module             |

## Catalog shape

Catalogs are `dict[(Task, CatalogKind, str), str]` where `CatalogKind` is one
of `"field"`, `"question"`, `"guidelines"`, `"label"`, `"widget"`.

The `name` slot encodes the kind:

- `"field"` / `"question"` — the Argilla `name=` identifier
  (e.g. `"query"`, `"topically_relevant"`)
- `"guidelines"` — always `""` (one guidelines string per task)
- `"label"` — `"<question_name>.<label_value>"`
  (e.g. `"topically_relevant.yes"`, `"discard_reason.unclear"`)
- `"widget"` — `"discard.<widget_key>"` for strings inside the injected
  discard HTML widget (e.g. `"discard.button_label"`)

## What stays stable across locales

These are machine identifiers and never translated — exports and downstream
parsing depend on them:

- Field `name=` and question `name=`
- Label *values* (e.g. `"yes"`, `"no"`, `DiscardReason.UNCLEAR.value`)
- `Locale` enum keys (`EN`, `DE`, ...)
- The set of catalog keys (every locale must define the same keys as `en.py`)

Only the catalog *values* (display text) vary by locale.

## Adding a new locale

### 1. Add the enum member

In `src/pragmata/core/schemas/annotation_task.py`:

```python
class Locale(StrEnum):
    EN = "en"
    DE = "de"
    FR = "fr"  # new
```

### 2. Create the catalog file

Copy `en.py` to `<locale>.py` and translate the values. Keep the keys identical.
Use `build_programmatic_entries()` from `structure.py` for the label/widget rows
— it fans the yes/no display strings across every question in
`_YES_NO_QUESTIONS_BY_TASK` and every key in `DISCARD_WIDGET_KEYS` so you supply
each translation once:

```python
from pragmata.core.annotation.locales.structure import build_programmatic_entries
from pragmata.core.annotation.locales.types import Catalog
from pragmata.core.schemas.annotation_task import DiscardReason, Task

CATALOG: Catalog = {
    (Task.RETRIEVAL, "field", "query"): "Requête",
    # ... rest of field / question / guidelines entries ...
    **build_programmatic_entries(
        yes_display="Oui",
        no_display="Non",
        discard_reason_displays={
            DiscardReason.INVALID_OR_UNREALISTIC.value: "Enregistrement invalide ou irréaliste",
            DiscardReason.UNCLEAR.value: "Relation ambiguë",
            DiscardReason.OUTSIDE_REVIEWER_EXPERTISE.value: "Hors expertise",
        },
        discard_widget_displays={
            "panel_summary": "Rejeter cet enregistrement",
            "panel_help": "...",
            "reason_label": "Motif :",
            "reason_placeholder": "— choisir —",
            "notes_label": "Notes facultatives :",
            "notes_placeholder": "...",
            "button_label": "Rejeter",
        },
    ),
}
```

### 3. Register the catalog

In `registry.py`:

```python
from pragmata.core.annotation.locales.fr import CATALOG as _FR

CATALOGS: dict[Locale, Catalog] = {
    Locale.EN: _EN,
    Locale.DE: _DE,
    Locale.FR: _FR,
}
```

### 4. Run the tests

```sh
uv run python -m pytest tests/unit/core/annotation/test_locales.py -v
```

`TestCatalogCompleteness` is parameterised over `Locale` and will tell you
about any missing or stray keys; `TestLocaleAwareSettings` confirms the
display strings actually flow into the rendered `rg.Settings`.

## Correctness invariant

`_YES_NO_QUESTIONS_BY_TASK` in `structure.py` lists every question that uses
the shared `yes`/`no` LabelQuestion. If you add or rename such a question
elsewhere in the codebase, update this mapping in the same change — otherwise
`build_programmatic_entries()` will emit stale label rows and the catalog
completeness test will fail.

`DISCARD_WIDGET_KEYS` is the source-of-truth list of chrome strings inside the
discard widget. Adding a key requires updating every locale's
`discard_widget_displays` argument; missing keys raise `KeyError` at import.

## What's automatic vs manual

Automatic (handled by `build_programmatic_entries()`):

- Yes/No label rows for every question in `_YES_NO_QUESTIONS_BY_TASK`
- Discard-reason label rows for every `DiscardReason` value
- Discard-widget rows for every `DISCARD_WIDGET_KEYS` entry, duplicated per task

Manual (per task, per locale):

- Field titles (e.g. `"query"`, `"chunk"`, `"answer"`, `"context_set"`)
- Question titles (the full sentence shown above each LabelQuestion)
- Guidelines text (one per task)
- The `"notes"`, `"discard_reason"`, `"discard_notes"` question titles
- The `"discard_flow"` field title
