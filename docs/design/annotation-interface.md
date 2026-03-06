# Annotation Interface

Web-based annotation interface for labelling RAG chatbot outputs across three annotation tasks, assignable to specific annotator groups. Powered by Argilla.

## Responsibilities

**In scope:**
- Present query-response pairs with retrieved context to annotators
- Collect binary labels per annotation task (see [Annotation Tasks](../methodology/annotation-tasks.md) for label definitions)
- Support concurrent multi-user annotation with task distribution
- Export annotations to CSV/Parquet for evaluation framework


## Architecture

>**NB:** this workflow depends on further decisions to be made re: CLI commands and low-config setup; this represents the latest thinking but is likely to further changes.

```
┌──────────────────────────────────────────────────────────────┐
│  One-time setup:                                             │
│    docker compose up                — start Argilla stack    │
│    chatboteval annotation init      — configure Argilla      │
│                   (workspaces, dataset schemas, users)       │
│                                                              │
│  UI access (annotation):                                     │
│    1. Browser URL  — direct navigation to Argilla instance   │
│    2. Python API   — chatboteval.annotation.open()           │
│    3. CLI          — chatboteval annotation open             │
│                                                              │
│  Data management:                                            │
│    Python API  — chatboteval.annotation.import/export()      │
│    CLI         — chatboteval annotation import/export        │
└──────────────────────────────────────────────────────────────┘
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

**`chatboteval annotation init`** configures the Argilla application (workspaces, dataset schemas, users). Docker lifecycle is separate (`docker compose up`):
- **Local:** `--local` — connects to local Docker Compose instance
- **Hosted:** `--hosted --url <argilla_api_url>` — connects to remote Argilla instance; writes `~/.chatboteval/config.yaml`
- **Cloud:** out of scope

> **NB:** as above, exact command name (`init` vs `setup`), flag design, and zero-config default behaviour (guided wizard vs auto-detect) are still under discussion. Current intent: `init` (no flags) defaults to a guided setup; `--local`/`--hosted` are explicit overrides.

**Package structure:**
- `/apps/annotation/` — Docker Compose configs for Argilla stack
- `src/chatboteval/argilla_client.py` — SDK wrappers for import/export/fetch

**Usage:** Annotation happens in browser via Argilla web UI. Python API and CLI handle setup, data management, and opening the UI.

## Inputs / Outputs

**Input:** Query-response pairs (CSV/JSON) with associated metadata. Retrieved context (RAG chunks and source documents) is required for Tasks 1 and 2; not required for Task 3.

**Annotation tasks:**

Three separate datasets are used (one per task), each task has distinct units of annotation and can be assigned across configurable annotator groups:

| Task | Unit | Labels | 
|---|---|---|
| Task 1: Retrieval | (query, chunk) pair | `topically_relevant`, `evidence_sufficient`, `misleading` | 
| Task 2: Grounding | (answer, context set) pair | `support_present`, `unsupported_claim_present`, `contradicted_claim_present`, `source_cited`, `fabricated_source` |
| Task 3: Generation | (query, answer) pair | `proper_action`, `response_on_topic`, `helpful`, `incomplete`, `unsafe_content` |

All labels binary, all required. Questions presented simultaneously within each task. Fields ordered per visibility contract below (primary content first, supporting context appended).

See [Annotation Tasks](../methodology/annotation-tasks.md) for label definitions and logical constraints.

### Visibility contract

Primary content is the minimal unit needed for the labelling task. Supporting context aids consistency but is kept secondary to reduce anchoring bias. Rationale: [Annotation UI Presentation](annotation-presentation.md).

```
Task 1: Retrieval
┌──────────────────────────────────┐
│ PRIMARY: Query + chunk           │  ← annotator focuses here
├──────────────────────────────────┤
│ SUPPORTING: Answer               │  ← collapsed by default
└──────────────────────────────────┘

Task 2: Grounding
┌──────────────────────────────────┐
│ PRIMARY: Answer + context set    │  ← annotator focuses here
├──────────────────────────────────┤
│ SUPPORTING: Query                │  ← collapsed by default
└──────────────────────────────────┘

Task 3: Generation
┌──────────────────────────────────┐
│ PRIMARY: Query + answer          │  ← annotator focuses here
├──────────────────────────────────┤
│ SUPPORTING: Retrieved passages   │  ← collapsed by default
└──────────────────────────────────┘
```

#### Collapsible rendering

Supporting context is collapsed by default and expanded on demand. This preserves the primary/supporting hierarchy in the UI — annotators are not anchored by supporting context unless they choose to view it.

Argilla v2 has no native collapsible field support. Approach: `rg.CustomField` with `advanced_mode=True` renders arbitrary HTML, enabling a `<details>`/`<summary>` element for native browser collapsibility without a custom frontend.

### Annotator-facing questions

Question wording reflects label semantics from [Annotation Tasks](../methodology/annotation-tasks.md). English is the default display language; German is available as an optional display language.

**Task 1: Retrieval**

| Label | Question (EN) | Question (DE) |
|---|---|---|
| `topically_relevant` | Does this passage contain information that is substantively relevant to the query? | Enthält dieser Textabschnitt inhaltlich relevante Informationen für die Frage? |
| `evidence_sufficient` | Does this passage contain sufficient evidence to support answering the query? | Enthält dieser Textabschnitt ausreichend Belege, um die Frage zu beantworten? |
| `misleading` | Could this passage plausibly lead to an incorrect or distorted answer? | Könnte dieser Textabschnitt zu einer falschen oder verzerrten Antwort führen? |

**Task 2: Grounding**

| Label | Question (EN) | Question (DE) |
|---|---|---|
| `support_present` | Is at least one claim in the answer supported by the provided context? | Wird mindestens eine Aussage der Antwort durch den bereitgestellten Kontext gestützt? |
| `unsupported_claim_present` | Does the answer contain claims not supported by the provided context? | Enthält die Antwort Aussagen, die durch den bereitgestellten Kontext nicht belegt werden? |
| `contradicted_claim_present` | Does the provided context contradict any claim in the answer? | Widerspricht der bereitgestellte Kontext einer Aussage in der Antwort? |
| `source_cited` | Does the answer contain a citation marker? | Enthält die Antwort einen Quellenhinweis? |
| `fabricated_source` | Does the answer cite a source not present in the retrieved context? | Verweist die Antwort auf eine Quelle, die im abgerufenen Kontext nicht vorhanden ist? |

**Task 3: Generation**

| Label | Question (EN) | Question (DE) |
|---|---|---|
| `proper_action` | Did the system choose the appropriate action for this query? | Hat das System die angemessene Reaktion auf diese Anfrage gewählt? |
| `response_on_topic` | Does the response substantively address the user's query? | Geht die Antwort substantiell auf die Anfrage des Nutzers ein? |
| `helpful` | Would this response enable a typical user to make progress on their task? | Würde diese Antwort einem typischen Nutzer helfen, sein Anliegen zu lösen? |
| `incomplete` | Does the response fail to cover required parts of the query? | Lässt die Antwort erforderliche Teile der Anfrage unbeantwortet? |
| `unsafe_content` | Does the response contain unsafe or policy-violating content? | Enthält die Antwort unangemessene oder richtlinienwidrige Inhalte? |

### Optional fields

Each task dataset includes one optional free-text field per annotated unit:

- **Notes** (*Anmerkungen*, `required=False`): annotator comments on edge cases, ambiguous instances, or unusual label choices.

### Output

- **Primary:** CSV — direct input to downstream evaluation pipeline
- **Secondary:** HuggingFace Datasets (Arrow/Parquet) via Argilla SDK `to_datasets()` (deferred)

Full column definitions and task-specific schemas: [Annotation Export Schema](annotation-export-schema.md).

**Data flow:** Annotations stored in Argilla (PostgreSQL backend) → chatboteval exports to CSV (primary) / HuggingFace Datasets (deferred) via Argilla SDK → downstream evaluation pipeline reads CSV.

## Failure Modes

**Concurrent assignment** — Argilla's `TaskDistribution` API ensures exclusive task allocation; completed records removed from all queues.

**Incomplete annotations** — Argilla tracks draft vs submitted status. Export only fetches submitted records.

**Inter-annotator disagreement** — Overlap assignment (e.g., 20% annotated by 2+ reviewers). IAA metrics (Krippendorff's alpha) integrated later.

**Export schema mismatch** — Validation step checks column presence and types before write.

## References

- [Decision 0001: Argilla Annotation Platform](../decisions/0001-annotation-argilla-platform.md)
- [Annotation Export Schema](annotation-export-schema.md)
- [Annotation UI Presentation](annotation-presentation.md) — rationale for joint labelling and visibility contract design
- [Annotation Tasks](../methodology/annotation-tasks.md)
- [Decision 0008: Authentication](../decisions/0008-annotation-interface-auth.md)
- Deployment design doc (forthcoming PR) — Docker Compose stack and infrastructure setup
- Argilla docs: https://docs.argilla.io/latest/
