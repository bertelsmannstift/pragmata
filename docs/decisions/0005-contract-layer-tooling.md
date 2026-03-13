# 0005: Contract Layer Tooling

Status: Draft

## Decision

The `core/` contract layer uses following tooling choices:

**Pydantic primarily for boundary schemas** — input specs, LLM-structured outputs, and disk artifacts (CSV/JSON) are defined as Pydantic models in `core/schemas/`. Settings models also use Pydantic. Dataclasses are used for runtime in-memory objects, where validation and serialisation are not needed.

**Rationale:** 
- Pydantic at boundaries gives free validation, serialisation (`model_dump_json()`/`model_dump()`), and schema generation (i.e. where those properties matter). 
- not Pydantic for runtime-only objects: (i) avoids unnecessary validation/coercion spreading into places that don't cross any boundary, (ii) maintains distinction between contracts and helpers, and (iii) avoids coupling without benefit ->  dataclass for an internal return type.

**`core/schemas/` is the canonical source of truth** for all boundary contracts. CSV/JSON are downstream serialisations of these models, not parallel definitions.

**Contract types over raw dicts** for all inter-module data exchange. Schema changes are breaking changes.

**No business logic in the schema layer** — `core/schemas/` contains types and contracts only. It is the root of the dependency graph and must not import from any other internal modules. Other `core/` subpackages (settings, paths) may contain implementation logic but still must not import from `api/` or `cli/`.

**Dependency direction:** `core/ ← api/ ← cli/` (per [ADR-0007](0007-packaging-invocation-surface.md)).

**`StrEnum` for controlled vocabularies** — `Task` and other fixed enumerations. (i) catches typos at import time, (ii) enables IDE autocomplete, (iii) centralises renaming.

**CSV for all data exchange** (-> the `context_set` column uses JSON-serialised array). One format everywhere; the column is machine-consumed so readability is not a concern.

**Frozen boundary schemas** — Pydantic models in `core/schemas/` are frozen (`frozen=True`); these are contracts, not accumulators, and mutation post-construction is a bug. Runtime dataclasses may use frozen on a case-by-case basis.

**No per-file schema versioning.** Schema version is implicit from the package version. Pydantic validation at load time is the mismatch signal for CSV files.

**Per-tool settings with shared resolution** — each tool defines its own settings bundle with semantically scoped groups. A shared base provides deterministic precedence resolution (overrides > env > config file > defaults) so tools don't reimplement merge logic.


> ## Alternatives considered
>
> **Pydantic for all types, including runtime objects:** Rejected — spreads validation/coercion where not needed, blurs boundary between contracts vs internal helpers, increases coupling w/o benefit. Frozen dataclasses preferred for pure in-memory objects.
>
> **Dataclasses for all types, Pydantic only at serialisation call sites:** Rejected — loses the single definition point for boundary contracts. Separating the schema model from its validation and serialisation creates drift between what the code validates and what it documents.
>
> **TOML/JSON for config:** Rejected — Pydantic Settings handles YAML, env vars, and CLI flags with a single model definition.


## Consequences

- Pydantic becomes a core dependency (already required for Argilla integration)
- `core/schemas/` is the definitive contract definition — contributors look there first
- Schema changes in `core/schemas/` should be treated as potentially breaking and handled with care
- Runtime types (dataclasses) are free to evolve without triggering breaking-change discipline
- `core/` must remain free of business logic to preserve its role as dependency root