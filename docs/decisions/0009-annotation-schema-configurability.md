# 0009: Annotation Schema Configurability

Status: Accepted

## Decision

Annotation schemas are hardcoded for v1.0 and are not user-configurable in v1.x. Schemas are defined as Argilla `Settings` objects in a dedicated Python module, separated from setup orchestration logic. No config-file-driven schema editor.

Any future schema change is a breaking design decision requiring a new ADR and new major version, not a config layer.


## Rationale

**Argilla is code-first by design:** `Settings` objects are Python — adding a config layer re-implements the SDK in a different format with no benefit.

**Schemas encode the label set from [Annotation Protocol](../methodology/annotation-protocol.md):** user-configurability would undermine result comparability across annotators and datasets.


>## Alternatives considered
>
>**YAML/TOML config layer:** Rejected — re-implements the Argilla SDK in a different format; requires a config schema, parser, and validator with ongoing maintenance burden.
>
>**Argilla admin UI schema editor:** Does not exist in Argilla v2; would require a custom frontend (out of scope — see [ADR-0001](0001-annotation-argilla-platform.md)).
>
>**Dedicated schema module (fork-and-modify path):** Schema definitions in a dedicated Python module (e.g. `schemas.py`), separated from setup orchestration — the recommended extension path if schema changes are warranted.


## Consequences

- Changing annotation dimensions requires modifying setup code and re-creating Argilla datasets
- Schema definitions should live in a dedicated module, not inline with orchestration logic, to make the fork-and-modify path obvious
- Any schema change requires updating/reviewing the annotation-related ADRs first (metrics taxonomy + annotation tasks + presentation), then updating the code implementation
- See [Annotation Protocol](../methodology/annotation-protocol.md) and [Annotation Interface](../design/annotation-interface.md) for current schema definitions


## Addendum: rendering flags are operational, not schema

Rendering flags on existing fields — `use_markdown` on `rg.TextField`, `v-html` vs. text interpolation in CustomField templates — are operational, not schema. They do not affect the labels collected, the export format, or comparability across deployments; they only change how existing content is presented in the annotator UI.

They are therefore configurable via `AnnotationSettings.field_render_mode` (deployment / workspace / task scopes, sparse-dict overlay), in the same operational layer as `min_submitted`, `constraint_severity`, and `locale`. This does not relax the hardcoded constraint on which fields, questions, and labels exist.
