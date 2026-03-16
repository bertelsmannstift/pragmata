# Import Pipeline

Data pipeline to load captured chatbot query-response data into Argilla annotation datasets. Operates against a source-agnostic canonical schema; source-specific logic lives in adapter modules which are out of scope.

## Responsibilities

**In scope:**
- Accept canonical import records (JSON)
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
│  chatboteval annotation import <file>│
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
**Entry point:** `chatboteval annotation import <file>` — accepts a JSON file conforming to the canonical import schema. JSON because the canonical record contains nested structures (chunk lists with sub-fields) that don't map cleanly to flat CSV rows.

**Single direction:** JSON → Argilla (no sync, no bidirectional updates)

**Fan-out:** One canonical record produces records in all three datasets. Task 1 produces K records per input (one per chunk); Tasks 2 and 3 produce one record each.

## Source Adapter & Canonical Schema

The import pipeline operates exclusively against a canonical import schema. Adapter modules (out of scope) transforms the source system's output into our canonical records; the pipeline never touches source-system internals. Adding a new source system requires only a new adapter.

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
- `generated_answer` ← canonical `answer` (supporting context, collapsible — per [Annotation Interface §Visibility contract](annotation-interface.md))

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

**Metadata (stored, not shown):**
- `record_uuid`
- `language` ← canonical `language`

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
