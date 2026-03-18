# 0007: Packaging and invocation surfaces

Status: Accepted


## Decision

This project is delivered as a standard distributable Python package (wheel/sdist), installable via `pip` compatible installers (e.g., `uv`).

The system exposes two supported invocation surfaces:

1. Python invocation surface: importing and calling functions/classes from `pragmata`, i.e., `import pragmata ...`

2. CLI invocation surface: a single canonical command via a `console_scripts` entry point, i.e., `pragmata ...`

The following architectural constraints apply:

- **CLI framework**
  - The CLI is implemented using Typer.
- **Library-first architecture**
  - All core logic and orchestration lives in library modules 
  - The CLI is a thin wrapper that parses inputs and delegates execution to the same library functions. No business logic is duplicated in the CLI layer.
- **Single canonical CLI entry point**
  - Exactly one top-level command (`pragmata`) is exposed. 
  - Subcommands may exist, but no additional console entry points are defined.
- **Src-layout repository structure**
  - The package uses a `src/` layout (`src/pragmata/`) to enforce clean import boundaries and avoid accidental reliance on repository root paths.
- **Curated public API surface**:
  - A small, intentionally defined public API is exposed for users, designed to be pleasant from an IDE.
  - Public symbols are documented and re-exported, intentionally concentrated in a small number of modules, and considered stable under semantic versioning.

## Rationale

- **Standard packaging lowers adoption friction** A normal pip-installable package aligns with user expectations and modern Python tooling.
- **Two invocation modes support different workflows** The Python API enables integration into notebooks, scripts, and pipelines. The CLI enables operational usage and reproducible runs from the shell.
- **Typer CLI framework** This supports a thin, typed CLI It enables declarative command definitions driven by type hints, reinforcing the library-first layering.
- **Library-first prevents logic drift** Centralizing business logic in importable modules ensures consistent behavior between CLI and Python usage, simplifies testing, and reduces maintenance risk.
- **Single entry point reduces cognitive load** One canonical command simplifies documentation, onboarding, and support.
- **Src-layout enforces discipline** It prevents accidental reliance on local paths and ensures imports behave the same in development and after installation.
- **Explicit public surface enables stability** Defining what is "public" allows internal refactoring without breaking UX, supporting long-term maintainability.

## Consequences

- The repository must maintain a clear separation between:
  - CLI interface layer
  - Core library modules
- Tests should target the library layer directly; CLI tests focus on argument parsing and wiring.
- Typer is a runtime dependency for the CLI surface.
- Changes to the public API require explicit versioning consideration.
- Core workflows must be invocable via both the Python API and the CLI. Documentation must provide parallel examples for each.
