# Export Pipeline

Data pipeline to export completed annotations from Argilla records to structured CSV files for downstream use.

## Responsibilities

**In scope:**
- Fetch submitted annotations from all three Argilla datasets
- Output one flat CSV per task (primary export)
- Validate export schema before write
- Post-submission constraint validation (flag logical violations before metric computation)

**Out of scope:**
- Data import (see [Import Pipeline](annotation-import-pipeline.md))
- IAA calculation (consumes export output; separate concern)
- Annotation quality filtering (export all submitted records; filtering is downstream)

## Architecture

```
Argilla PostgreSQL
        │
        ├── task1_retrieval ──────────────────────────────► retrieval.csv
        │   (workspace_a)                    (query, chunk, chunk_id,
        │                                                    chunk_rank, labels...)
        │
        ├── task2_grounding ──────────────────────────────► grounding.csv
        │   (workspace_a)                    (answer, context_set,
        │                                                    labels...)
        │
        └── task3_generation ─────────────────────────────► generation.csv
            (workspace_b)                      (query, answer, labels...)
```
>**Pending** - exact API / CLI wrapper wording below are TBC, TODO update when decided.

**Entry point:** `chatboteval export <output_dir>` (CLI) or `chatboteval.export(...)` (Python API)

## Inputs

Three Argilla datasets, accessed via Argilla SDK. Filter: `status == "submitted"` only (exclude draft, discarded). 


| Dataset | Records |
|---------|---------|
| `task1_retrieval` | One record per query–chunk pair |
| `task2_grounding` | One record per answer-context set pair |
| `task3_generation` | One record per query–answer pair |

> NB: Workspace setup and associated task assignment are deployment configuration, not fixed architecture (see [Workspace & Task Distribution](annotation-workspace-task-distribution.md)).

## Export Format & Schema

> **Depends on:** [Annotation Protocol](../methodology/annotation-protocol.md) for label definitions

Export as human-readable CSV, flat format, one row per annotator response vector. Three task-specific CSVs with disjoint label columns and shared metadata.

> **Secondary format: HuggingFace Datasets** — Argilla v2 SDK natively supports `dataset.records.to_datasets()`. Deferred feature.
>
> **Alternative rejected: Nested JSON (one object per record with annotation array)** — Harder to compute inter-annotator agreement; requires unpacking before analysis.

Export schemas define what downstream pipelines need. They are not a mirror of what annotators see in the annotation interface (field naming and structure may differ).

### Shared metadata columns (all tasks)

| Column | Type | Description |
|--------|------|-------------|
| `record_uuid` | string | Cross-dataset record identifier (see [Import Pipeline](annotation-import-pipeline.md)) |
| `annotator_id` | string | Annotator username |
| `task` | string | Task identifier: `retrieval`, `grounding`, or `generation` |
| `language` | string | Language code (e.g. `de`, `en`) |
| `inserted_at` | datetime | When the record was loaded into Argilla, tracks batch provenance |
| `created_at` | datetime | Response submission timestamp |
| `record_status` | string | Argilla record status: `pending` or `completed` (whether the record met its `TaskDistribution` overlap target) |

**Not exported:**
- `response_status` — we filter to `submitted` only; column would be constant
- `_server_id` — Argilla-internal UUID; `record_uuid` already serves as the cross-dataset identifier

### Task 1: Retrieval — `retrieval.csv`

Unit: one row per `(query, chunk, annotator)` triple.

| Column | Type | Description |
|--------|------|-------------|
| `input_query` | string | Original user query as sent to the retriever |
| `chunk` | string | Retriever-level text segment |
| `chunk_id` | string | ID assigned to the chunk at ingestion |
| `doc_id` | string | Document ID linking the chunk to its source document |
| `chunk_rank` | int | Rank of chunk in (post-rerank) result set |
| `topically_relevant` | bool | Chunk contains information substantively related to the query |
| `evidence_sufficient` | bool | Chunk provides sufficient evidence to support answering the query |
| `misleading` | bool | Chunk could plausibly lead to an incorrect or distorted answer |
| `notes` | string | Optional annotator notes |
| *(shared metadata)* | | |

### Task 2: Grounding — `grounding.csv`

Unit: one row per `(answer, context_set, annotator)` triple.

| Column | Type | Description |
|--------|------|-------------|
| `answer` | string | Generated answer |
| `context_set` | string | Full retrieved context as injected in the prompt, concatenated as a single string with `[CTX_SEP]` separators |
| `support_present` | bool | At least one answer claim is supported by evidence in the context set |
| `unsupported_claim_present` | bool | Answer contains at least one claim not supported by the context set |
| `contradicted_claim_present` | bool | Context set contains information that contradicts at least one answer claim |
| `source_cited` | bool | Answer contains at least one citation marker in the expected format |
| `fabricated_source` | bool | Answer cites at least one source not present in the retrieved context set |
| `notes` | string | Optional annotator notes |
| *(shared metadata)* | | |

### Task 3: Generation — `generation.csv`

Unit: one row per `(query, answer, annotator)` triple.

| Column | Type | Description |
|--------|------|-------------|
| `query` | string | Input query |
| `answer` | string | Generated answer |
| `proper_action` | bool | Response selects the appropriate type (answer, refusal, clarification) given the query |
| `response_on_topic` | bool | Response substantively addresses the user's request |
| `helpful` | bool | Response would enable a typical user to make progress on their task |
| `incomplete` | bool | Response fails to cover one or more required parts of the query |
| `unsafe_content` | bool | Response contains content violating safety or policy constraints |
| `notes` | string | Optional annotator notes |
| *(shared metadata)* | | |

### Merged view 

Optional downstream merge joins all three task CSVs on `record_uuid`. Task 1 requires an aggregation step first, as multiple rows per query (one per chunk) must be reduced to a per-query summary before joining.

**Task 1 aggregation:** boolean OR across all chunks for a query (`_any` suffix). E.g.:

| record_uuid | chunk | topically_relevant | evidence_sufficient | misleading |
|-------------|-------|--------------------|---------------------|------------|
| abc-123 | c1 | true | false | false |
| abc-123 | c2 | false | false | false |
| abc-123 | c3 | true | true | false |
| abc-123 | c4 | false | false | true |

| record_uuid | topically_relevant_any | evidence_sufficient_any | misleading_any |
|-------------|------------------------|-------------------------|----------------|
| abc-123 | true | true | true |

**Final Merged View (all tasks):** 

| Group | Columns |
|-------|---------|
| Shared | `record_uuid`, `query`, `answer`, `context_set` |
| Task 1 (aggregated) | `topically_relevant_any`, `evidence_sufficient_any`, `misleading_any` |
| Task 2 | `support_present`, `unsupported_claim_present`, `contradicted_claim_present`, `source_cited`, `fabricated_source` |
| Task 3 | `proper_action`, `response_on_topic`, `helpful`, `incomplete`, `unsafe_content` |
| Meta | `annotator_id_t1`, `annotator_id_t2`, `annotator_id_t3`, ... |

NULLs for any task where no submitted annotation exists for the record.

## Constraint Validation

The [Annotation Protocol](../methodology/annotation-protocol.md) defines logical consistency constraints between labels (e.g. `evidence_sufficient = 1 ⇒ topically_relevant = 1`). The export pipeline validates these before metric computation and flags or rejects violating rows. See [Annotation Protocol](../methodology/annotation-protocol.md) for the full constraint list.

## Failure Modes

**Missing `record_uuid`:**
- Records without `record_uuid` cannot be joined across datasets
- Log warning; include in flat export, exclude from merged view

**Partial annotations:**
- A record may have annotations in some tasks but not others
- Include in flat export; include in merged view with NULLs for missing tasks

**Schema mismatch:**
- Validate expected column names and types before write
- Fail with diff of expected vs actual schema

**Argilla connection failure:**
- Fail with clear error including API URL
- No partial output written (atomic: write only on success)

## References

- [Annotation Protocol](../methodology/annotation-protocol.md) — Label definitions, units of annotation, and logical constraints
- [Import Pipeline](annotation-import-pipeline.md) — Upstream data flow
- Quality Assurance (forthcoming) — IAA calculation consumes export output (reads flat export directly, filtering by label column within each task CSV).
