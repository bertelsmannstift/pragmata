# Package Contracts Design

## Purpose

Scaffolds the contract layer for `chatboteval` — canonical schemas, path conventions, and config structures that all internal modules depend on. Each section is a stub to be developed during implementation. Together they form the shared vocabulary and structural conventions for the whole package.

Architectural choices underpinning this layer (Pydantic at boundaries, frozen dataclasses for runtime types, dependency direction, CSV interchange) are captured in [ADR-0005](../decisions/0005-contract-layer-tooling.md).


---

## Directory structure

```
src/chatboteval/core/
├── schema/               # Boundary contracts — Pydantic SSOT
│   ├── __init__.py         # Re-exports for clean imports
│   ├── base.py             # Task enum and shared schema types
│   ├── annotations.py      # Import record + annotation schemas
│   └── csv_io.py           # CSV read/write & serialisation logic
├── types/                # Runtime types (frozen dataclasses)
│   └── __init__.py         # e.g. run result types, internal helpers
├── paths.py              # PathResolver
├── settings_base.py      # ResolvableSettings base class
└── settings.py           # Global config (Pydantic Settings)
```

---

## 1. Boundary Schemas

> Pydantic models in `core/schema/` — the single source of truth for all inter-module contracts. CSV/JSON are downstream serialisations of these models.

### Controlled vocabulary

```python
class Task(StrEnum):
    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    GENERATION = "generation"
```

### Import record

Input format to `generate` and `annotation import`. Canonical contract for data arriving from a source adapter:

```
QueryResponsePair
  query: str
  response: str
  context_set: list[str]       # one entry per retrieved chunk
  source_id: str               # opaque identifier from source system
  metadata: dict               # pass-through from source (timestamps, model id, etc.)
```

### Annotation schemas

**Annotation base:**

```python
class AnnotationBase(BaseModel, frozen=True):
    record_uuid: UUID
    annotator_id: str
    language: str
    inserted_at: datetime            # when record was loaded into Argilla (batch provenance)
    created_at: datetime             # response submission timestamp
    record_status: str               # Argilla record status: pending | completed
    notes: str | None
```

**Task-specific annotation schemas** — inherit from `AnnotationBase`, add task-specific labels. Match the [export schema](annotation-export-schema.md).

```
RetrievalAnnotation(AnnotationBase)        # unit: (query, chunk)
  input_query: str
  chunk: str
  chunk_id: str
  doc_id: str
  chunk_rank: int
  topically_relevant: bool
  evidence_sufficient: bool
  misleading: bool
```

```
GroundingAnnotation(AnnotationBase)        # unit: (answer, context_set)
  answer: str
  context_set: str                         # concatenated with [CTX_SEP]
  support_present: bool
  unsupported_claim_present: bool
  contradicted_claim_present: bool
  source_cited: bool
  fabricated_source: bool
```

```
GenerationAnnotation(AnnotationBase)       # unit: (query, answer)
  query: str
  answer: str
  proper_action: bool
  response_on_topic: bool
  helpful: bool
  incomplete: bool
  unsafe_content: bool
```


---

## 2. CSV Serialisation

> On-disk format — downstream of the Pydantic schemas above. Flat, string-typed, one row per record.

`core/schema/csv_io.py` handles serialisation/deserialisation. This section documents only the differences between the in-memory schema and the on-disk representation.

### Import record CSV

Flat serialisation of `QueryResponsePair`, plus an `id` column:

| Column | Type | Notes |
|---|---|---|
| `id` | str | Unique identifier for the QR pair (not on the schema model) |
| `query` | str | |
| `response` | str | |
| `context_set` | str (JSON array) | `list[str]` serialised as JSON array |
| `source` | str | `source_id` on the schema model |

### Annotation export CSVs

Three task-specific CSVs, one per task. Each is a flat serialisation of the corresponding annotation schema. See [Annotation Export Schema](annotation-export-schema.md) for full column definitions.

Serialisation differences from the schema models:

| Schema field | CSV column | Difference |
|---|---|---|
| `Task` enum | `task` (str) | Enum serialised as string value |
| `UUID` | `record_uuid` (str) | UUID serialised as string |
| `context_set: list[str]` | `context_set` (str) | Chunks concatenated with `[CTX_SEP]` separator |
| `datetime` fields | str | ISO 8601 formatted |

File naming: `retrieval.csv`, `grounding.csv`, `generation.csv`.


---

## 3. Runtime Types

> Frozen dataclasses in `core/types/` — internal ergonomic types that don't cross module boundaries and don't need Pydantic validation.

Examples: run result objects (run_id, output paths, counts, summary stats). These may reference schema models (e.g. `spec: QueryGenerationSpec`) but do not duplicate field definitions or add validation constraints. They are not contracts — they're implementation conveniences.

```python
@dataclass(frozen=True)
class QueryGenRunResult:
    run_id: str
    output_path: Path
    spec: QueryGenerationSpec       # references the boundary schema
    row_count: int
```


---

## 4. File Path Conventions

Canonical locations for data artefacts produced and consumed by the pipeline (not source code layout — see [ADR-0007](../decisions/0007-packaging-invocation-surface.md) for package structure).

```
<workspace_dir>/
  data/                 # datasets (versionable, shareable)
  outputs/              # computed results (reproducible from data)

~/.chatboteval/
  config.yaml           # user config (optional — see Section 5)

./apps/
  annotation/
    docker-compose.yml  # Argilla stack (local mode only)
    .env
```

Paths are exposed via a `PathResolver` dataclass in `src/chatboteval/core/paths.py`:

```python
@dataclass
class PathResolver:
    workspace_dir: Path

    @property
    def data(self) -> Path:
        return self.workspace_dir / "data"

    @property
    def outputs(self) -> Path:
        return self.workspace_dir / "outputs"
```

The resolver is initialised at the API/CLI entrypoint using `Path.cwd()` (or a user-provided path). Consuming code imports the resolver rather than constructing paths — single resolution point, easy to override in tests or via config.

> Directory structure is fixed by convention. Only override is `workspace_dir` itself, which shifts the entire tree.


---

## 5. Config & Settings

### Global config

Structure of `~/.chatboteval/config.yaml`, parsed at startup by `core/settings.py` using Pydantic Settings.

**The config file is optional.** Built-in defaults mean the tool works with zero config — the file is only needed when overriding defaults (e.g. Argilla credentials, custom output paths).

```yaml
argilla:
  mode: local              # local | hosted
  url: http://localhost:6900
  api_key: owner.apikey

output:
  workspace_dir: ./my_workspace
```

Full precedence chain (highest to lowest):
```
CLI flags / API call overrides          ← one-off overrides
        ↓
CHATBOTEVAL_* env vars
        ↓
./my_project/querygen.yaml              ← optional project-level tool config
        ↓
~/.chatboteval/config.yaml              ← user-global (argilla creds, output paths)
        ↓
built-in defaults
```

`resolve()` accepts a `config: dict | None` — the caller loads whichever config file is relevant (global, project-level, or none) and passes it in. The merge is the same regardless of source.

> Global config only at `~/.chatboteval/config.yaml` for v1.0. Follows the dbt/AWS CLI pattern, appropriate for a tool with credentials. Project-level tool configs are optional and caller-supplied.

### Tool-specific settings

Settings modules live alongside their tool, not in `core/`:

```
src/chatboteval/
├── core/
│   ├── settings_base.py      # ResolvableSettings base class (shared)
│   └── settings.py           # Global config (argilla, output paths)
├── querygen/
│   ├── querygen_settings.py  # QueryGenRunSettings
│   └── ...
└── api/
    ├── querygen.py           # calls QueryGenRunSettings.resolve()
    └── ...
```

Each tool's settings module contains:

- **Semantically scoped settings classes** for logical groupings (e.g. `QueryGenSettings`, `LangChainModelSettings`)
- **A `RunSettings` class** that bundles the above plus tool-level fields (`work_dir`, `run_id`, etc.) and exposes a `resolve()` class method

The `resolve()` merge logic lives once in `ResolvableSettings` in `core/settings_base.py`; each `RunSettings` class inherits it:

```python
# core/settings_base.py
class ResolvableSettings(BaseModel):
    @classmethod
    def resolve(cls, *, config: dict | None, overrides: dict) -> "ResolvableSettings":
        # Precedence: overrides > config > defaults
        # Recursive merge so nested dicts don't clobber each other
        merged = deep_merge(config or {}, strip_none(overrides))
        return cls.model_validate(merged)
```

```python
# querygen_settings.py
class QueryGenSettings(BaseModel):
    diversity: float = 0.7
    n_candidates: int = 5

class QueryGenRunSettings(ResolvableSettings):
    work_dir: Path = Field(default_factory=lambda: Path(".").resolve())
    run_id: str | None = None
    gen: QueryGenSettings = Field(default_factory=QueryGenSettings)
```

```python
# api/querygen.py
def run_querygen(raw_csv, *, work_dir=None, run_id=None, diversity=None, config_path=None):
    config = load_yaml(config_path) if config_path else None
    cfg = QueryGenRunSettings.resolve(
        config=config,
        overrides={"work_dir": work_dir, "run_id": run_id, "gen": {"diversity": diversity}},
    )
    # orchestrator uses cfg.gen.diversity, cfg.work_dir, etc.
```

> Task-specific configuration (active tasks, stratification rules, distribution overlap targets) is not runtime config — it is applied at dataset creation time during `chatboteval annotation import`. See [Workspace & Task Distribution](annotation-workspace-task-distribution.md).


---

## References

- [ADR-0005: Contract Layer Tooling](../decisions/0005-contract-layer-tooling.md) — tooling choices underpinning this layer
- [ADR-0007: Packaging and Invocation Surface](../decisions/0007-packaging-invocation-surface.md) — module dependency direction
- [Annotation Export Schema](annotation-export-schema.md) — full column definitions for export CSVs
