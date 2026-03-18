# Packaging and invocation surface


## Purpose

Specifies the concrete package structure, module boundaries, and invocation bindings that realize the packaging and access model defined in ADR-0007. This document serves as the reference for how code must be placed and exposed within the package.


## Responsibilities and boundaries

**In scope**
- Package structure
- Surface exposure
- Package/module boundaries
- Public API curation and stability rules

**Out of scope**
- Runtime configuration model
- Detailed `pyproject.toml` field-by-field documentation
- Feature-specific flows


## Package Layout and Module Boundaries

**Package scaffold**

```
src/
└── pragmata/
    ├── __init__.py              # curated public re-exports only
    ├── api/
    │   ├── __init__.py          # stable, high-level Python entrypoints
    │   └── ...                  # orchestration-facing functions
    ├── cli/
    │   ├── __init__.py          # exposes Typer app as `app`
    │   ├── app.py               # Typer root + subcommand registration
    │   └── commands/            
    │       └── ...              # individual command implementations
    └── core/
        └── ...                  # internal implementation (non-public)
```

**Boundary rules**

- `pragmata.cli` is a thin execution layer over the API. May import `api`, but must not import `core` directly.
- `pragmata.api` contains internal orchestration. May import `core`, but must not depend on `cli`.
- `pragmata.core` contains implementation details and is explicitly not public. Must not depend on `cli` or leak into the public API surface.

Allowed dependency direction:

cli ⟹ api ⟹ core


## Invocation Surfaces

**Python import surface**

- `api/` provides the application service layer (internal orchestration):
  - imports and orchestrates functions from `core/`
  - defines internal entrypoints used by the CLI and re-exported selectively at the top-level
  - isolates the CLI and top-level API from internal structure
- curated API re-exported at the top-level via `pragmata/__init__.py`
- Only the top-level `pragmata` namespace is a supported stable import surface.
  Imports from `pragmata.api` are considered internal.

**CLI surface**

- Each CLI command is implemented in `cli/commands/` and registered in `pragmata/cli/app.py`
- CLI is exposed via a single console-script entry defined in `pyproject.toml`


## Packaging configuration

- Build and packaging configuration are defined solely in `pyproject.toml`.


## Extension points

**Adding new user-facing functionality**

- Implement in `core/` first.
- Expose orchestration in `api/`
- Add a CLI subcommand that calls the API function.
- Add curated top-level re-exports
- Must not add new console scripts.


## Architectural guardrails

- **Logic drift**: prevented by the enforced dependency direction and review rule “CLI delegates to API only.”
- **Accidental public API expansion**: prevented by curated re-exports
- **Import/path surprises in dev vs installed**: mitigated by src-layout.
