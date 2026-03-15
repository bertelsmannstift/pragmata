# 0010: Annotation Multi-Dataset Architecture

Status: Draft


## Decision

Three separate Argilla datasets — one per annotation task:

| Dataset | Task | Record unit |
|---------|------|-------------|
| `task_retrieval` | Retrieval quality | One record per (query, chunk) |
| `task_grounding` | Grounding quality | One record per (answer, context set) |
| `task_generation` | Generation quality | One record per (query, answer) |

All datasets are assigned to workspaces (see [Workspace & Task Distribution](../design/annotation-workspace-task-distribution.md)). Records from the same input are linked across datasets via `record_uuid` metadata.


## Rationale

**Retrieval and Grounding have incompatible record multiplicities.** Retrieval produces K records per input (one per retrieved chunk); Grounding produces one record per input. This difference in cardinality is the fundamental constraint that prevents sharing a dataset:

- A unified schema would require either embedding K chunks as separate fields (bounded, inflexible) or repeating the (answer, context set) pair K times (creates duplicate annotation burden for grounding labels).
- Argilla datasets enforce a single fixed schema, so all records must have the same fields. A Retrieval record has a `chunk` field; a Grounding record has a `context` field. These cannot coexist cleanly in one schema.

**Generation is separated from Grounding to enable flexible workspace assignment.** Although Grounding and Generation share similar record structures, their question sets are entirely disjoint. Separate datasets allow operators to assign tasks to workspaces freely - e.g., grouping by annotator expertise, splitting across teams, or consolidating into a single workspace. Dataset-to-workspace mapping is deployment configuration (see [Workspace & Task Distribution](../design/annotation-workspace-task-distribution.md)).

**Three datasets is the minimum needed, no more.** The multiplicity constraint forces Retrieval and Grounding apart; the disjoint question sets make a Grounding + Generation merge pointless (annotators would see irrelevant questions). Three datasets keeps schema and task cleanly aligned while leaving workspace assignment unconstrained.


## Alternatives Considered

**Unified schema with optional/conditional fields**

One dataset with all fields (`query`, `chunk`, `answer`, `context_set`) and questions for all tasks. Annotators see all fields regardless of their task.

Rejected:
- Argilla v2 has no conditional question display — annotators would see irrelevant fields
- K Retrieval records (chunk-level) would exist alongside Grounding records (answer-level) in the same dataset, confusing annotators
- IAA computation becomes harder when record structure is heterogeneous

**Two datasets: one per annotator group**

`retrieval_grounding` dataset for Retrieval + Grounding combined; `generation` for Generation.

Rejected:
- Retrieval produces K records per input; Grounding produces 1. If loaded into the same dataset, the schema must accommodate both structures, reintroducing the multiplicity problem.
- Annotation UX is worse: annotators see questions for both Retrieval and Grounding on every record, but most records are relevant only to one.

**Dynamic dataset routing at import time only**

Create a single logical dataset per annotator group and route records at import, without encoding task separation in schema.

Rejected: same problem — Argilla schemas are static. You cannot route Retrieval and Grounding records into the same dataset without schema conflicts.


## Consequences

- Import pipeline must route each input into multiple datasets: K Retrieval records + 1 Grounding record + 1 Generation record per input
- Export pipeline exports each dataset independently; cross-dataset joining is a downstream concern
- `record_uuid` is a required metadata field on all three datasets — it is the only cross-dataset link
- IAA is computed per-dataset, not cross-dataset (each task converges independently)


## References

- [Workspace & Task Distribution](../design/annotation-workspace-task-distribution.md) — workspace assignment and dataset schemas
- [Import Pipeline](../design/annotation-import-pipeline.md) — routing logic
- [Export Pipeline](../design/annotation-export-pipeline.md) — per-dataset CSV export and merged view
- [Export Pipeline](../design/annotation-export-pipeline.md) — export schema and pipeline per task
- [Annotation Protocol](../methodology/annotation-protocol.md) — task definitions and record units
- [Annotation Interface](../design/annotation-interface.md) — question wording and UI visibility contract
