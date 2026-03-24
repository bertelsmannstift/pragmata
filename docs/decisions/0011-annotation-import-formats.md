# 0011: Annotation Import Accepted Formats

Status: Draft

## Decision

`import_records()` accepts multiple input formats directly — JSON, JSONL, CSV files, HF Datasets, and pandas DataFrames. Format loading is baked into the API layer; CLI remains a thin wrapper per [ADR-0007](0007-packaging-invocation-surface.md).

Previously the import pipeline accepted JSON only ([import pipeline design doc](../design/annotation-import-pipeline.md)), with the rationale that nested chunk structures don't map cleanly to flat CSV rows. This decision expands accepted formats while preserving the canonical schema as the internal validation target.

## Accepted formats

| Format | Input type | Notes |
|--------|-----------|-------|
| JSON | `str \| Path` | Array of canonical records |
| JSONL | `str \| Path` | One record per line |
| CSV | `str \| Path` | Two chunk layouts (see below) |
| HF Dataset | `Dataset` | Converted to `list[dict]` |
| DataFrame | `DataFrame` | Converted to `list[dict]` |
| Raw dicts | `list[dict]` | Pass-through (existing behaviour) |

File format detected by extension; `format=` kwarg overrides for ambiguous cases.

## CSV chunk representation

CSV supports two layouts:

1. **JSON string column** — `chunks` column contains a JSON array string. One row per record.
2. **Denormalised rows** — one row per chunk with `chunk_text`, `chunk_id`, `doc_id`, `chunk_rank` columns. Rows grouped into records by `record_id`/`group` column if present, otherwise by `query` + `answer` content.

## Architecture

- Format loaders live in `core/annotation/loaders.py` — internal, not part of public API
- `import_records()` dispatches by input type, delegates to loader, passes `list[dict]` to existing `validate_records()` pipeline
- Format/file errors raise immediately; record validation errors reported via `ImportResult.errors`

## Consequences

- Users can pass common data formats directly without pre-parsing
- CLI is a thin wrapper: `import_records(client, args.file)` — no format logic in CLI
- CSV chunk representation is defined (resolves the open question from the original design doc)
- Source-system-specific adapters (e.g. transforming a chatbot's raw output to canonical schema) remain out of scope
- Adding new formats requires only a new loader function in `core/` and an extension mapping
