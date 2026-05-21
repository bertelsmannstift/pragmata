# Annotation locales

Locale-specific display strings for Argilla dataset titles, fields, questions,labels, guidelines, and discard-widget chrome. Translations live as data in per-locale YAML files alongside this README; a registry auto-discovers them at import time. `Locale` is an open `str` type - a locale code is any string the registry knows about.

## Package layout

| File          | Purpose                                                                |
|---------------|------------------------------------------------------------------------|
| `types.py`    | `CatalogKind` / `CatalogKey` / `Catalog` type aliases                  |
| `loader.py`   | YAML loader + locale-invariant structure (`_YES_NO_QUESTIONS_BY_TASK`, `DISCARD_WIDGET_KEYS`) |
| `registry.py` | `CATALOGS` dict (auto-discovered) + `get_catalog(locale)` lookup       |
| `en.yaml`     | English catalog (source-of-truth)                                      |
| `de.yaml`     | German catalog                                                         |
| `__init__.py` | Package docstring only - import from the specific module               |

## Catalog shape

The on-disk shape is a nested YAML mapping (`fields`, `questions`, `guidelines`, `labels`, `widget`); `loader.load_catalog` fans it out into the flat in-memory shape consumers use:

```
dict[(Task, CatalogKind, str), str]
```

where `CatalogKind` is one of `"field"`, `"question"`, `"guidelines"`, `"label"`, `"widget"`. The third tuple slot encodes the kind:

- `"field"` / `"question"` - the Argilla `name=` identifier
  (e.g. `"query"`, `"topically_relevant"`)
- `"guidelines"` - always `""` (one guidelines string per task)
- `"label"` - `"<question_name>.<label_value>"`
  (e.g. `"topically_relevant.yes"`, `"discard_reason.unclear"`)
- `"widget"` - `"discard.<widget_key>"` for strings inside the injected
  discard HTML widget (e.g. `"discard.button_label"`)

## What stays stable across locales

These are machine identifiers and never translated - exports and downstream parsing depend on them:

- Field `name=` and question `name=`
- Label *values* (e.g. `"yes"`, `"no"`, `DiscardReason.UNCLEAR.value`)
- The set of catalog keys (every locale must define the same keys as
  `en.yaml`)

Only the catalog *values* (display text) vary by locale.

## Adding a new locale

### 1. Copy `en.yaml` to `<code>.yaml`

Name the file after the locale code (e.g. `fr.yaml`). The registry picks
it up at import; the file stem is what operators pass to `--locale`.

```sh
cp src/pragmata/core/annotation/locales/en.yaml \
   src/pragmata/core/annotation/locales/fr.yaml
```

### 2. Translate values, keep keys

Edit the new file. Translate every string under `fields:`, `questions:`,
`guidelines:`, `labels:`, and `widget:`. Do not rename keys, do not add new
ones - the keys are wired into Argilla and the discard widget; only the
values change per locale.

```yaml
fields:
  retrieval:
    query: Requête
    chunk: Passage
    # ...

labels:
  yes_display: Oui
  no_display: Non
  discard_reasons:
    invalid_or_unrealistic: Enregistrement invalide ou irréaliste
    unclear: Relation ambiguë
    outside_reviewer_expertise: Hors expertise

widget:
  panel_summary: Rejeter cet enregistrement
  reason_label: "Motif :"
  # ...
```

### 3. Run the tests

```sh
uv run python -m pytest tests/unit/core/annotation/test_locales.py -v
```

`TestCatalogCompleteness` parametrises over every registered locale and
tells you about any missing or stray keys versus `en.yaml`.
`TestLocaleAwareSettings` confirms the display strings actually flow into
the rendered `rg.Settings`.

That's it. No enum to edit, no registry to register, no import statement
to add.

## Correctness invariant

`_YES_NO_QUESTIONS_BY_TASK` in `loader.py` lists every question that uses
the shared `yes`/`no` `LabelQuestion`. If you add or rename such a
question elsewhere in the codebase, update this mapping in the same
change - otherwise the loader will emit stale label rows and the catalog
completeness test will fail.

`DISCARD_WIDGET_KEYS` in `loader.py` is the source-of-truth list of chrome
strings inside the discard widget. Adding a key requires updating the
`widget:` section of every locale YAML; missing keys raise `KeyError` at
load time.

`DiscardReason` (in `core/schemas/annotation_task.py`) drives the
discard-reason label fan-out. Adding a reason requires updating every
locale's `labels.discard_reasons:` map; missing reasons raise `KeyError`
at load time.

## What the YAML must contain

Spelled out per task (under `fields:`, `questions:`, `guidelines:`):

- Field titles (e.g. `"query"`, `"chunk"`, `"answer"`, `"context_set"`)
- Question titles (the full sentence shown above each `LabelQuestion`)
- Guidelines text (one per task)
- The `notes`, `discard_reason`, `discard_notes` question titles
- The `discard_flow` field title

Spelled out once each, then fanned out by the loader:

- `labels.yes_display` / `labels.no_display` - one string each; the loader
  emits a label row for every yes/no `LabelQuestion` in every task
- `labels.discard_reasons.*` - one string per `DiscardReason` value;
  fanned out per task
- `widget.*` - seven chrome strings for the discard widget; fanned out
  per task

So a contributor writes every display string exactly once. The loader
handles the duplication across the per-task catalog rows consumers
expect.
