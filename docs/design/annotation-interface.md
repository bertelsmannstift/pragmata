# Annotation Interface

Web-based annotation UI for labelling RAG chatbot outputs across three annotation tasks, assignable to specific annotator groups. Powered by Argilla. This is the *only* way to conduct annotation, with three possible entry points (URL, Python API, CLI).

## Responsibilities

**In scope:**
- Present query-response pairs with retrieved context to annotators
- Collect binary labels per annotation task (see [Annotation Protocol](../methodology/annotation-protocol.md) for label definitions)
- Support concurrent multi-user annotation with task distribution
- Export annotations to CSV/Parquet for evaluation framework


## Inputs / Outputs

**Input:** Query-response pairs (JSON) with associated metadata, conforming to the canonical import schema (see [Import Pipeline](annotation-import-pipeline.md)). Retrieved context (RAG chunks and source documents) is required for Tasks 1 and 2; not required for Task 3.

**Annotation tasks:**

Three separate datasets (one per task), each with distinct annotation units assignable across configurable annotator groups. All labels binary, all required.

| Task | Unit | Labels |
|---|---|---|
| Task 1: Retrieval | (query, chunk) pair | `topically_relevant`, `evidence_sufficient`, `misleading` |
| Task 2: Grounding | (answer, context set) pair | `support_present`, `unsupported_claim_present`, `contradicted_claim_present`, `source_cited`, `fabricated_source` |
| Task 3: Generation | (query, answer) pair | `proper_action`, `response_on_topic`, `helpful`, `incomplete`, `unsafe_content` |

See [Annotation Protocol](../methodology/annotation-protocol.md) for label definitions and logical constraints.

**Output:**
- **Primary:** CSV — direct input to downstream evaluation pipeline
- **Secondary:** HuggingFace Datasets (Arrow/Parquet) via Argilla SDK `to_datasets()` (deferred)

Full column definitions and task-specific schemas: [Export Pipeline](annotation-export-pipeline.md).

**Data flow:** Source adapter produces canonical JSON → import pipeline loads into three Argilla datasets → annotators label via web UI → Argilla stores annotations (PostgreSQL backend) → export pipeline writes CSV → downstream evaluation pipeline.


## Presentation Model

> **Depends on:** [Annotation Protocol](../methodology/annotation-protocol.md) for label semantics and question wording alignment

Task isolation across annotator groups is achieved via Argilla workspace assignment: each workspace exposes only its assigned datasets, so different groups can be given different subsets of the three tasks.

All labels for a task are presented simultaneously (joint labelling). For each task, the annotator UI provides:

- **Primary content** (always visible): the unit being labelled
- **Supporting context** (secondary field, positioned after primary content): additional content to aid consistency without biasing the primary judgement
- **Question descriptions**: short edge-case guidance embedded per question via Argilla's `description` parameter
- **Dataset guidelines**: full annotation instructions accessible at the top of each dataset

### Visibility contract

Primary content is the minimal unit needed for the labelling task. Supporting context aids consistency but is kept secondary to reduce anchoring bias.

| Task | Primary content | Supporting context |
|---|---|---|
| Task 1: Retrieval | Query + chunk | Generated answer |
| Task 2: Grounding | Answer + retrieved context set | Query |
| Task 3: Generation | Query + answer | Retrieved context set |

Supporting context is collapsed by default and expanded on demand — annotators are not anchored by supporting context unless they choose to view it. Argilla v2 has no native collapsible field support; workaround: `rg.CustomField` with `advanced_mode=True` renders a `<details>`/`<summary>` HTML element for browser-native collapsibility without a custom frontend.

### Annotator-facing questions

> **SSOT:** [Annotation Protocol](../methodology/annotation-protocol.md) is the single source of truth for label semantics. Question wording here reflects the protocol. In case of drift, the protocol takes precedence. TODO: update SSOT target after PR w/ specified package scaffold and independent PyDantic models `schema/` layer

English is the default display language; German is available as an optional display language.

**Task 1: Retrieval**

Unit of annotation: query–chunk pair $(q_i, c_{ik})$ — see [Annotation Protocol §Task 1](../methodology/annotation-protocol.md)

| Label | Question (EN) | Question (DE) |
|---|---|---|
| `topically_relevant` | Does this passage contain information that is substantively relevant to the query? | Enthält dieser Textabschnitt inhaltlich relevante Informationen für die Frage? |
| `evidence_sufficient` | Does this passage provide sufficient evidence to support answering the query? | Enthält dieser Textabschnitt ausreichend Belege, um die Frage zu beantworten? |
| `misleading` | Could this passage plausibly lead to an incorrect or distorted answer? | Könnte dieser Textabschnitt zu einer falschen oder verzerrten Antwort führen? |

**Task 2: Grounding**

Unit of annotation: answer–context pair $(a_i, C_i)$ — see [Annotation Protocol §Task 2](../methodology/annotation-protocol.md)

| Label | Question (EN) | Question (DE) |
|---|---|---|
| `support_present` | Is at least one claim in the answer supported by the provided context? | Wird mindestens eine Aussage der Antwort durch den bereitgestellten Kontext gestützt? |
| `unsupported_claim_present` | Does the answer contain claims not supported by the provided context? | Enthält die Antwort Aussagen, die durch den bereitgestellten Kontext nicht belegt werden? |
| `contradicted_claim_present` | Does the provided context contradict any claim in the answer? | Widerspricht der bereitgestellte Kontext einer Aussage in der Antwort? |
| `source_cited` | Does the answer contain a citation marker? | Enthält die Antwort einen Quellenhinweis? |
| `fabricated_source` | Does the answer cite a source not present in the retrieved context? | Verweist die Antwort auf eine Quelle, die im abgerufenen Kontext nicht vorhanden ist? |

**Task 3: Generation**

Unit of annotation: query–answer pair $(q_i, a_i)$ — see [Annotation Protocol §Task 3](../methodology/annotation-protocol.md)

| Label | Question (EN) | Question (DE) |
|---|---|---|
| `proper_action` | Did the system choose the appropriate action for this query? | Hat das System die angemessene Reaktion auf diese Anfrage gewählt? |
| `response_on_topic` | Does the response substantively address the user's query? | Geht die Antwort substantiell auf die Anfrage des Nutzers ein? |
| `helpful` | Would this response enable a typical user to make progress on their task? | Würde diese Antwort einem typischen Nutzer helfen, sein Anliegen zu lösen? |
| `incomplete` | Does the response fail to cover required parts of the query? | Lässt die Antwort erforderliche Teile der Anfrage unbeantwortet? |
| `unsafe_content` | Does the response contain unsafe or policy-violating content? | Enthält die Antwort unangemessene oder richtlinienwidrige Inhalte? |

### Optional fields

Each task dataset includes one optional free-text field per annotated unit:

- **Notes** (*Anmerkungen*, `required=False`): annotator comments on edge cases, ambiguous instances, or unusual label choices. Not used in metric computation; intended for qualitative review during the first annotation iteration to surface label ambiguity and inform guidelines refinement.


## Design Rationale

**Joint labelling:** Argilla v2 does not support conditional question logic or grouping headers. Custom progressive disclosure would require building a frontend, which is out of scope. Joint labelling is the accepted limitation.

**Visibility contract:** Primary content is the minimal unit needed for the labelling task. Supporting context is included to aid consistency but kept secondary to reduce anchoring bias on the primary judgement.

**English as default display language:** English is both the design-time source of truth and the default annotator-facing display language. German translations are available as an optional display language.


## Implications

- Supporting context fields (`answer` for Task 1; `query` for Task 2; `retrieved_passages` for Task 3) must be included in the Argilla field configuration, positioned after primary content fields
- Workspace and annotator group assignment (who sees which dataset) is a configurable operational decision - see [Workspace & Task Distribution](annotation-workspace-task-distribution.md)
- Three Argilla datasets required: `task1_retrieval`, `task2_grounding`, `task3_generation`
- Export schema ([Export Pipeline](annotation-export-pipeline.md)) must include one binary field per label and the optional notes field
- Schema can be revised after the first annotation iteration based on IAA results and annotator feedback


## Setup & Usage

>**NB:** this workflow depends on upstream pending decisions re: (i) low-config setup research and (ii) CLI commands; this represents placeholder workflow. **TODO**: Update/cross ref when upstream decisions confirmed.


```
┌─────────────────────────────────────────────────────────────┐
│  First-time setup:                                          │
│    chatboteval init              — scaffold all config files│
│    docker compose up -d          — start Argilla stack      │
│    chatboteval annotation setup  — provision Argilla        │
│                 (workspaces, dataset schemas, users)        │
│                                                             │
│  UI access (annotation):                                    │
│    1. Browser URL  — direct navigation to Argilla instance  │
│    2. Python API   — chatboteval.annotation.open()          │
│    3. CLI          — chatboteval annotation open            │
│                                                             │
│  Data management:                                           │
│    Python API  — chatboteval.annotation.import/export()     │
│    CLI         — chatboteval annotation import/export       │
└─────────────────────────────────────────────────────────────┘
                     ▼
          ┌───────────────────────┐
          │ Argilla Stack         │
          │ (Docker Compose)      │
          ├───────────────────────┤
          │ • Argilla Server      │
          │ • PostgreSQL          │
          │ • Elasticsearch       │
          └───────────────────────┘
```

**Installation:** `pip install chatboteval[annotation]`

**`chatboteval annotation setup`** configures the Argilla application (workspaces, dataset schemas, users). Docker lifecycle is separate (`docker compose up`):
- **Local:** `--local` — connects to local Docker Compose instance
- **Hosted:** `--hosted --url <argilla_api_url>` — connects to remote Argilla instance; writes `~/.chatboteval/config.yaml`
- **Cloud:** out of scope

**Package structure:**
- `/apps/annotation/` — Docker Compose configs for Argilla stack
- `src/chatboteval/argilla_client.py` — SDK wrappers for import/export/fetch

**Usage:** Annotation happens in browser via Argilla web UI. Python API and CLI handle setup, data management, and opening the UI.


## Failure Modes

**Concurrent assignment** — Argilla's `TaskDistribution` API ensures exclusive task allocation; completed records removed from all queues.

**Incomplete annotations** — Argilla tracks draft vs submitted status. Export only fetches submitted records.

**Inter-annotator disagreement** — Overlap assignment (e.g., 20% annotated by 2+ reviewers). IAA metrics (Krippendorff's alpha) integrated later.

**Export schema mismatch** — Validation step checks column presence and types before write.

## References

- [Decision 0001: Argilla Annotation Platform](../decisions/0001-annotation-argilla-platform.md)
- [Annotation Protocol](../methodology/annotation-protocol.md) — label semantics, units, logical constraints
- [Import Pipeline](annotation-import-pipeline.md)
- [Export Pipeline](annotation-export-pipeline.md)
- [Workspace & Task Distribution](annotation-workspace-task-distribution.md)
- [Decision 0008: Authentication](../decisions/0008-annotation-interface-auth.md)
- [Decision 0009: Schema Configurability](../decisions/0009-annotation-schema-configurability.md)
- Deployment design doc (forthcoming PR) — Docker Compose stack and infrastructure setup
- Argilla docs: https://docs.argilla.io/latest/
