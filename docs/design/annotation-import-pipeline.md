# Import Pipeline

Data pipeline to load captured chatbot query-response data into Argilla annotation datasets. Operates against a source-agnostic canonical schema. Format loading (JSON, JSONL, CSV, HF Dataset, DataFrame) is baked into the API per [ADR-0011](../decisions/0011-annotation-import-formats.md); source-system-specific adapters remain out of scope.

## Responsibilities

**In scope:**
- Accept canonical import records (JSON, JSONL, CSV, HF Dataset, DataFrame — see [ADR-0011](../decisions/0011-annotation-import-formats.md))
- Transform to Argilla record schema
- Load into stratified datasets by annotation task (see [Annotation Protocol](../methodology/annotation-protocol.md))
- Validate records before import
- Assign `record_uuid` metadata for cross-dataset linking

**Out of scope:**
- Data capture from chatbot
- Source-system-specific transformation (handled by source adapter)
- Annotation task and label definition ([Annotation Protocol](../methodology/annotation-protocol.md))
- Export functionality (separate export pipeline)

## Architecture

```

┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ - - - ─ ┐
  External (out of scope)
│ ┌──────────────────────────────────┐                          |
  │  Source RAG System Output        │
│ └──────────────┬───────────────────┘                          │
                 │ query, answer, retrieved chunks, prompt context
│                ▼                                              │
  ┌──────────────────────────────────┐
│ │  Source Adapter (system specific)│                          |
  └──────────────┬───────────────────┘
└ ─ ─ ─ ─ ─ ─ ─ - ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ - - - ┘
                 │ JSON (canonical import schema)
                 ▼
-──────────────────────────────────-──────────────────────────────
   Import Pipeline begins:
-──────────────────────────────────-──────────────────────────────

┌──────────────────────────────────────┐
│  pragmata annotation import <file>│
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│  Transform to            │
│  Argilla Records (SDK)   │
└────────────┬─────────────┘
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
┌────────┐ ┌────────┐ ┌────────┐
│ task1  │ │ task2  │ │ task3  │
│retriev.│ │ground. │ │generat.│
└────────┘ └────────┘ └────────┘
            │ 
            │ (workspace & task distribution → annotators label via Argilla Web UI)
            ▼
```
**Entry point:** `pragmata annotation import <file>` — accepts JSON, JSONL, or CSV files conforming to the canonical import schema. Format detected by extension, overridable via `format=` kwarg. HF Dataset and pandas DataFrame objects also accepted programmatically. See [ADR-0011](../decisions/0011-annotation-import-formats.md) for format details and CSV chunk representation.

**Single direction:** JSON → Argilla (no sync, no bidirectional updates)

**Fan-out:** One canonical record produces records in all three datasets. Task 1 produces K records per input (one per chunk); Tasks 2 and 3 produce one record each.

## Source Adapter & Canonical Schema

The import pipeline operates exclusively against a canonical import schema. Format loaders (JSON, JSONL, CSV, HF Dataset, DataFrame → `list[dict]`) are baked into the API per [ADR-0007](../decisions/0007-packaging-invocation-surface.md) and [ADR-0011](../decisions/0011-annotation-import-formats.md). Source-system-specific adapters (e.g. transforming a chatbot's raw output into canonical records) remain out of scope — adding a new source system requires only a new adapter upstream of the import pipeline.

### Canonical record

>TODO: a separate PR with designs for overall package structure will add an independent `schema/` layer (add ref when done), this pydantic model will be the SSOT.

One record per RAG query-response cycle:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Original or LLM-refined user query |
| `answer` | string | yes | Generated answer |
| `chunks` | list[Chunk] | yes | Selected/Top-K retriever-level segments used during retrieval |
| `context_set` | string | yes | Prompt-inserted context string, as the model saw it; documents separated by `[CTX_SEP]` |
| `language` | string | no | Detected language code |

**Chunk schema:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Stable unique identifier for this chunk |
| `doc_id` | string | Source document/publication identifier |
| `chunk_rank` | int | Position in the flat (post-rerank) chunk list |
| `text` | string | Chunk text content |

>### chunk_rank
>**NB:**`chunk_rank` is not a concern of the initial import schema/contract, but rather it is a derived attribute from upstream transformation processes prior to analysis.
>
>Retrieval metrics (MRR@K, NDCG@K) require a flat, ordered list of K chunks. When a reranker scores at document level (not chunk level), chunk rank is derived as:
>
>1. Sort documents by reranker score (descending)
>2. For each document in rank order, enumerate its selected chunks in document order
>3. `chunk_rank` = 1-indexed position in the resulting flat list
>
>This treats all chunks from a higher-ranked document as more relevant than chunks from a lower-ranked document (=consistent with the reranker's signal). 
>
>**NB:** If the source system provides per-chunk scores directly, `chunk_rank` can be derived from those instead.

## Outputs

Three Argilla datasets, each receiving records from every import:

### `task_retrieval` — one record per chunk

**Fields (shown to annotators):**
- `query` ← canonical `query`
- `chunk` ← canonical `chunks[k].text`
- `answer` ← canonical `answer` (supporting context — positioned last per [Annotation Interface §Visibility contract](annotation-interface.md))

**Metadata (stored, not shown):**
- `record_uuid` — assigned at import; links same query across all three datasets
- `chunk_id` ← canonical `chunks[k].chunk_id`
- `doc_id` ← canonical `chunks[k].doc_id`
- `chunk_rank` ← canonical `chunks[k].chunk_rank`
- `language` ← canonical `language`

---

### `task_grounding` — one record per answer-context set pair

**Fields (shown to annotators):**
- `query` ← canonical `query`
- `answer` ← canonical `answer`
- `context_set` ← canonical `context_set`

**Metadata (stored, not shown):**
- `record_uuid`
- `language` ← canonical `language`

---

### `task_generation` — one record per query-answer pair

**Fields (shown to annotators):**
- `query` ← canonical `query`
- `answer` ← canonical `answer`
- `context_set` ← canonical `context_set` (auxiliary UI context, collapsed — not part of the annotation unit; see [Annotation Interface §Visibility contract](annotation-interface.md))

**Metadata (stored, not shown):**
- `record_uuid`
- `language` ← canonical `language`

## Calibration partitioning

Calibration vs production assignment is **per annotation item**, not per `record_uuid` (see [ADR-0012](../decisions/0012-annotation-per-item-calibration-partition.md)). The annotation item differs by task: `record_uuid` for grounding and generation (one item per record), `(record_uuid, chunk_id)` for retrieval (one item per chunk).

### Per-task knobs

`calibration_fraction` and `calibration_max_items` are inheritable across deployment / workspace / task scopes via the `Inherit` sentinel:

- `calibration_fraction` (deployment default 0.1): fraction of annotation items routed to the calibration dataset.
- `calibration_max_items` (deployment default `None`): absolute cap on calibration annotation items per task. Smaller of (fraction × N_items, cap) wins.

Cap unit is the annotation item: a cap of 200 on retrieval means 200 chunks; on grounding/generation, 200 records. Override via YAML config per-(workspace, task); the CLI `--calibration-fraction` and `--calibration-max-records` flags set deployment defaults only.

### Deterministic bucketing

Per-(task, unit) digest: `int(sha256(seed‖task.value‖unit_id)[:8], 16)`. Unit is calibration iff `digest < fraction × 2^32`. Mixing the task name into the hash makes per-task draws statistically independent - a chunk that lands in retrieval-calibration is not constrained to also land in grounding-calibration (cleaner under the naive Krippendorff bootstrap pragmata uses).

### Slot accounting under cap

For each task, per import:

1. Count existing calibration units in the manifest (for retrieval, sum across chunks; for grounding/generation, count records).
2. Compute `remaining = None if cap is None else max(0, cap - existing)`.
3. If existing already exceeds cap (config tightened post-hoc), log a warning and treat `remaining = 0` - never demote existing entries (manifest-lock invariant).
4. Bucket new units by fraction; sort eligible candidates by digest ascending; promote the first `remaining` (or all if uncapped). The rest demote to production.

### Order-dependence under binding cap

Because the manifest is append-only, the calibration set under a binding cap is a function of `(corpus, seed, import_order)`, not `(corpus, seed)` alone. If a corpus arrives across multiple imports and the cap binds, the specific set chosen depends on import order - but cardinality always honours the cap. This is documented in `assign_partitions` and the dedicated test in `tests/unit/core/annotation/test_partition.py::test_cap_under_split_imports_is_order_dependent_by_design`.

### Per-record manifest schema

`PartitionManifestEntry` carries:

- `grounding_generation_calibration: dict[Task, bool]` - keys GROUNDING and GENERATION
- `retrieval_chunk_calibration: dict[str, bool]` - keys are `chunk_id`; entries absent at fan-out time default to production
- `calibration_fraction_at_import: dict[Task, float]` and `calibration_max_items_at_import: dict[Task, int | None]` - per-task provenance stamped at import time

## Failure Modes

**Invalid/incomplete canonical record:**
- Validate required fields (`query`, `answer`, `chunks`, `context_set`) before processing
- Skip malformed records with error log; report summary at completion

**Missing chunks:**
- Import record anyway; log warning
- Annotators can flag retrieval failure via label questions

**Duplicate imports:**
- Check `record_uuid` / `query` against existing records
- Skip duplicates with warning logged

**Schema mismatch:**
- Validate field names match configured Argilla dataset schema on startup
- Fail early with expected vs. actual schema diff

## References

- [Export Pipeline](annotation-export-pipeline.md) — export schema and cross-dataset linking via `record_uuid`
- [Annotation Protocol](../methodology/annotation-protocol.md) — label definitions and annotation units
- [Annotation Interface](annotation-interface.md) — visibility contract and question wording
- [ADR-0012: Per-Item Calibration Partition](../decisions/0012-annotation-per-item-calibration-partition.md) - design review, statistical rationale, and consequences for the calibration partitioning section above
