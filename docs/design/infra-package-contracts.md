# Package Contracts Design

## Purpose

Scaffolds the contract layer for `chatboteval` — canonical schemas, path conventions, and config structures that all internal modules depend on. Each section describes patterns to be followed during implementation; concrete schemas will be later defined in tool-specific issue specs and code PRs.

Tooling choices underpinning this layer (Pydantic @ boundaries, dataclasses for runtime types, dependency dir, CSV interchange) are captured in [ADR-0005](../decisions/0005-contract-layer-tooling.md).


---

## Contract scaffold

```
src/chatboteval/
├── core/
│   ├── schemas/                   # Boundary schemas (Pydantic SSOT)
|   |   ├── __init__.py
│   │   ├── base.py                # Shared StrEnums and common types
│   │   ├── csv_io.py              # CSV serialisation helpers
│   │   └── <tool>_*.py
│   ├── types/                     # Runtime types (frozen dataclasses)
│   │   └── __init__.py
│   ├── settings/
│   │   ├── settings_base.py       # ResolveSettings base class
│   │   └── <tool>_settings.py     
│   └── paths.py                   # PathResolver
├── api/
│   └── <tool>.py                  # Entrypoint: resolve(), PathResolver init
└── cli/
    └── commands/

<workspace_dir>/
  data/                            # datasets
  outputs/                         # computed results

~/.chatboteval/
  config.yaml                      # user config (optional)
```

Dependency direction: `core/ ← api/ ← cli/` (per [ADR-0007](../decisions/0007-packaging-invocation-surface.md)).

`core/schemas/` is dependency root — no internal imports. Tool implementations and `api/` import from it; it imports from nothing.


---

## 1. Boundary Schemas

Pydantic models in `core/schemas/` — the single source of truth for all inter-module contracts. CSV/JSON are downstream serialisations of these models, not parallel definitions.

Per-tool schemas are organised by tool prefix (e.g. `querygen_input.py`, `querygen_output.py`, `querygen_plan.py`).

All schema models are frozen by default. Schema changes are breaking changes requiring version bumps.


---

## 2. CSV Serialisation

> On-disk format — downstream of the Pydantic schemas above. Flat, string-typed, one row per record.

`core/schemas/csv_io.py` handles serialisation/deserialisation. This section documents only the differences between the in-memory schema and the on-disk representation.


CSV is the interchange format for all tabular data. One format everywhere.

Conventions:

- 1 CSV per record type, flat and string-typed
- List fields serialised as JSON arrays (e.g. `context_set`)
- Enums serialised as string values
- UUIDs and datetimes serialised as strings (ISO 8601 for dates)
- Canonical column order defined per schema as a module-level constant

Serialisation/deserialisation logic lives in dedicated helpers, not on the schema models themselves.


---

## 3. Runtime Types

Frozen dataclasses in `core/types/` — internal ergonomic types that don't cross module boundaries and don't need Pydantic validation.

Examples: run result objects (run_id, output paths, record counts). These may reference schema models but don't duplicate field definitions. They are not contracts — they're implementation conveniences free to evolve without triggering breaking-change discipline.


---

## 4. File Path Conventions

Canonical locations for data artefacts produced and consumed by the pipeline (not source code layout — see [ADR-0007](../decisions/0007-packaging-invocation-surface.md) for package structure).

```
<workspace_dir>/
  data/                 # datasets (versionable, shareable)
  outputs/              # computed results (reproducible from data)

~/.chatboteval/
  config.yaml           # user config (optional — see Section 5)
```

A `PathResolver` dataclass in `core/paths.py` exposes these locations. Initialised once at the API/CLI entrypoint and threaded down — consuming code uses the resolver rather than constructing paths directly (single resolution point, easy to override in tests).

Tools that need a tool-specific output subdirectory compute it at the entrypoint (e.g. `paths.outputs / "querygen"`) — no subclassing needed. Directory structure is fixed by convention; only `workspace_dir` itself is overridable, which shifts the entire tree.


---

## 5. Config & Settings

### Global config

Optional user config at `~/.chatboteval/config.yaml`, parsed by `core/settings/` using Pydantic Settings. Built-in defaults mean the tool works with zero config — the file is only needed when overriding defaults (e.g. credentials, custom output paths).

Full precedence chain (highest to lowest):

```
CLI flags / API call overrides          ← one-off overrides
        ↓
CHATBOTEVAL_* env vars
        ↓
Config file (~/.chatboteval/config.yaml)
        ↓
Built-in defaults
```

### Tool-specific settings

Each tool has a settings module in `core/settings/` containing:

- **Semantically scoped settings classes** for logical groupings (e.g. LLM config, sampling parameters)
- **A `RunSettings` class** that bundles the above plus run-level fields and inherits from `ResolveSettings`

`ResolveSettings` in `core/settings/settings_base.py` provides a shared `resolve()` classmethod handling the precedence merge (recursive deep-merge, so nested dicts don't clobber siblings). Each tool's `RunSettings` inherits this — no reimplementation per tool.

The API entrypoint (`api/<tool>.py`) calls `RunSettings.resolve()` with caller-supplied overrides and config, then passes the resolved settings to the implementation layer.


---

## References

- [ADR-0005: Contract Layer Tooling](../decisions/0005-contract-layer-tooling.md) — tooling choices underpinning this layer
- [ADR-0007: Packaging and Invocation Surface](../decisions/0007-packaging-invocation-surface.md) — module dependency direction