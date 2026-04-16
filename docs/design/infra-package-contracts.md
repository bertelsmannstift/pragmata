# Package Contracts Design

## Purpose

Scaffolds the contract layer for `pRAGmata` — canonical schemas, path conventions, and config structures that all internal modules depend on. Each section describes patterns to be followed during implementation; concrete schemas are defined in tool-specific issue specs and code PRs.

Tooling choices underpinning this layer (Pydantic @ boundaries, dataclasses for runtime types, dependency dir, CSV interchange) are captured in [ADR-0005](../decisions/0005-contract-layer-tooling.md).


---

## Contract scaffold

```
src/pragmata/core/
├── schemas/          # Boundary schemas (Pydantic SSOT)
├── types/            # Runtime types (dataclasses)
├── settings/         # Per-tool settings + shared resolution base
└── paths/            # Workspace path resolution
```

```
<base_dir>/                        # defaults to cwd
  <tool>/
    runs/
      <run_id>/                    # per-run artefacts

~/.pragmata/
  config.yaml                      # user config (optional)
```

Dependency direction: `core/ ← api/ ← cli/` (per [ADR-0007](../decisions/0007-packaging-invocation-surface.md)).

`core/schemas/` is dependency root — no internal imports. Tool implementations and `api/` import from it; it imports from nothing.


---

## 1. Boundary Schemas

Pydantic models in `core/schemas/` — the single source of truth for all inter-module contracts. CSV/JSON are downstream serialisations of these models, not parallel definitions.

Per-tool schemas are organised by tool prefix (e.g. `querygen_input.py`, `querygen_output.py`, `querygen_plan.py`, `querygen_realize.py`).

Boundary schema models are frozen by default (`model_config = ConfigDict(frozen=True)`). Schema changes should be treated as potentially breaking.

### LLM-boundary schemas

Schemas that double as LLM structured output contracts follow additional conventions:

- All fields include `Field(description=...)` — these serve as written instructions to the model
- A thin wrapper model (e.g. `QueryBlueprintList`) provides the top-level object for structured output
- Cross-stage join keys (e.g. `candidate_id`) couple related schemas; changes to join keys are breaking changes across all schemas that share them


---

## 2. Serialisation

> On-disk formats — downstream of the Pydantic schemas above.

Shared CSV serialisation/deserialisation helpers live in `core/` (whereas `core/schemas/` contains contract definitions only). This section documents only the differences between the in-memory schema and the on-disk representation.

### Tabular data (CSV)

CSV is the interchange format for all tabular data. One format everywhere.

Conventions:

- 1 CSV per record type, flat and string-typed
- List fields serialised as JSON arrays (e.g. `context_set`)
- Enums serialised as string values
- UUIDs and datetimes serialised as strings (ISO 8601 for dates)
- Column order derived from Pydantic model field definition order (model_fields.keys())

Serialisation/deserialisation logic lives in dedicated helpers, not on the schema models themselves.

### Run metadata (JSON sidecar)

Non-tabular, run-level metadata (e.g. `run_id`, `created_at`, model info) is written as a JSON file alongside the CSV (e.g. `synthetic_queries.meta.json`). This avoids repeating per-run fields as columns in every CSV row.


---

## 3. Runtime Types

Frozen dataclasses for internal ergonomic types that don't need Pydantic validation. These are not contracts, they're implementation conveniences free to evolve without triggering breaking-change discipline.

General-purpose runtime types live in `core/types/`. Domain-specific frozen dataclasses (e.g. path bundles) live alongside their domain module (e.g. `core/paths/`).

Examples: run result objects (run_id, output paths, record counts), path bundles. These may reference schema models but don't duplicate field definitions.


---

## 4. File Path Conventions

Canonical locations for data artefacts produced and consumed by the pipeline (not source code layout — see [ADR-0007](../decisions/0007-packaging-invocation-surface.md) for package structure).

```
<base_dir>/                       # defaults to cwd; overridable
  <tool>/
    runs/
      <run_id>/                   # per-run artefacts (CSV + JSON sidecar)

~/.pragmata/
  config.yaml                     # user config (optional — see Section 5)
```

A shared workspace path resolver in `core/paths/` provides deterministic construction of workspace root, tool root, and run root paths. Initialised once at the API/CLI entrypoint and threaded down — consuming code uses the resolver rather than constructing paths directly (single resolution point, easy to override in tests).

Tools with complex output structures may define their own path bundles that build on the shared resolver.


---

## 5. Config & Settings

### Global config

Optional user config at `~/.pragmata/config.yaml`, parsed by `core/settings/` using Pydantic Settings. Built-in defaults mean the tool works with zero config — the file is only needed when overriding defaults (e.g. credentials, custom output paths).

Full precedence chain (highest to lowest):

```
CLI flags / API call overrides          ← one-off overrides
        ↓
Environment variables
        ↓
Config file (~/.pragmata/config.yaml)
        ↓
Built-in defaults
```

### Secrets

Secrets (API keys, credentials) are resolved separately from settings — they do not appear in config files or settings models. `core/settings/settings_base.py` provides a `resolve_api_key()` helper that reads API keys from pre-known environment variables (e.g. `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `ARGILLA_API_KEY`). A `MissingSecretError` is raised when the required key is not set.

### Tool-specific settings

Each tool has a settings module in `core/settings/` containing:

- **Semantically scoped settings classes** for logical groupings (e.g. `LlmSettings` for model config)
- **A `RunSettings` class** that bundles the above plus run-level fields and inherits from `ResolvableSettings`

A shared base in `core/settings/` provides deterministic precedence resolution. Each tool's settings bundle inherits this — no reimplementation per tool. The API entrypoint resolves settings and passes them to the implementation layer.


---

## References

- [ADR-0005: Contract Layer Tooling](../decisions/0005-contract-layer-tooling.md) — tooling choices underpinning this layer
- [ADR-0007: Packaging and Invocation Surface](../decisions/0007-packaging-invocation-surface.md) — module dependency direction
