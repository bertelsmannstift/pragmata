# `deploy/annotation/` — annotation stack & per-deployment data

This directory holds everything needed to run the Argilla annotation stack
for a Pragmata deployment: the Docker Compose definition, environment
defaults, and three categories of deployment-instance data (workspace
configs, user specs, input records).

## Files at a glance

| Path | Tracked? | Purpose |
|---|---|---|
| `docker-compose.dev.yml` | o | Argilla stack definition (Argilla + Postgres + Elasticsearch + Redis). |
| `.env.dev.example` | o | Template for the env file consumed by Docker Compose. |
| `.env` | x (gitignored) | Real env for this deployment. Copied from `.env.dev.example` on first `make docker-up`. |
| `ws_setup_configs/topic.example.yaml` | o | Minimal copyable skeleton for a per-topic setup config. |
| `ws_setup_configs/*.yaml` | x (gitignored) | Real per-topic configs (e.g. `team-a.yaml`). |
| `users.example.json` | o | Schema example for the annotator/owner roster. |
| `users.json` | x (gitignored) | Real roster for this deployment. **May contain passwords — never commit.** |
| `input_records/topic.example.jsonl` | o | Five-record fixture demonstrating the `QueryResponsePair` JSONL shape. |
| `input_records/*.jsonl` | x (gitignored) | Real per-topic corpora fed to `pragmata annotation import`. |

The gitignore convention is: anything that varies per deployment stays
local; an `*.example.*` sibling documents the schema for newcomers.

## Operator workflow

### 1. Start the stack

```bash
make docker-up                  # all services bundled (default)
make docker-up-external-pg      # bring your own Postgres
make docker-up-external-es      # bring your own Elasticsearch
make docker-up-external         # all backing services external
```

The first invocation copies `.env.dev.example` → `.env`. Edit `.env` to
change ports, credentials, or external service URLs. See the comments in
`.env.dev.example` for which variables each profile requires.

Argilla becomes available at `http://localhost:6900`.

### 2. Provision workspaces and users

For each topic, create a `ws_setup_configs/<topic>.yaml` (see the section
[Workspace configuration](#workspace-configuration) below for the
minimum, common, and full forms). Define annotators and owners in
`users.json` (see `users.example.json` for the shape). The `password`
field is optional — omit it and the API auto-generates one and returns
it in `SetupResult.generated_passwords`.

```bash
pragmata annotation setup \
  --config deploy/annotation/ws_setup_configs/<topic>.yaml \
  --users  deploy/annotation/users.json
```

Setup is idempotent: re-running skips existing workspaces and users.

### 3. Import records

Each topic's corpus lives in `input_records/<topic>.jsonl`, one
`QueryResponsePair` per line. See `topic.example.jsonl` for the exact
shape (query, answer, context_set, chunks, optional language).

```bash
pragmata annotation import deploy/annotation/input_records/<topic>.jsonl \
  --config deploy/annotation/ws_setup_configs/<topic>.yaml \
  --dataset-id <topic>
```

Datasets are auto-created on first import. Re-imports are deterministic:
the same input pair always lands on the same Argilla record id.

## Workspace configuration

`workspace_dataset_map` is a nested structure
`{workspace_name: {task: TaskOverlap}}`. It supports two degrees of
freedom — naming workspaces however you like, and grouping multiple
tasks under one workspace — but most deployments use it minimally.
Three progressive forms:

### Minimal — no config at all

If you're happy with the built-in defaults (one workspace per task,
literally named `retrieval`, `grounding`, `generation`, with overlap
defaults), don't write a config file. Built-in defaults handle it:

```bash
pragmata annotation setup --users deploy/annotation/users.json
```

### Common — one topic, named workspaces

To namespace workspaces under a topic (typical for multi-team
deployments) but otherwise take all defaults, write a small
`<topic>.yaml`:

```yaml
workspace_dataset_map:
  team-a-retrieval:  { retrieval:  {} }
  team-a-grounding:  { grounding:  {} }
  team-a-generation: { generation: {} }
```

Each `{}` accepts default `TaskOverlap` values
(`production_min_submitted: 1`, `calibration_min_submitted: 3`).

### Full schema reference

All tunable `AnnotationSettings` fields with their defaults:

```yaml
workspace_dataset_map:
  topic-retrieval:
    retrieval:
      production_min_submitted: 1   # responses needed to mark a record done in production
      calibration_min_submitted: 3  # responses needed in the calibration dataset
  topic-grounding:
    grounding:
      production_min_submitted: 1
      calibration_min_submitted: 3
  topic-generation:
    generation:
      production_min_submitted: 1
      calibration_min_submitted: 3

calibration_fraction: 0.1        # fraction of imported records routed to calibration
calibration_partition_seed: 0    # RNG seed for deterministic calibration partitioning

# Usually set via --dataset-id on the CLI rather than here.
# Appended to dataset names for run-scoping (e.g. "pilot1" → retrieval_production_pilot1).
dataset_id: ""

# Workspace base directory for run artifacts (logs, manifests, export CSVs).
# Defaults to cwd at load time.
base_dir: "."

# When true, exported CSVs include responses the annotator marked as discarded.
# Overridden per-invocation by --include-discarded on `pragmata annotation export`.
include_discarded: false
```

Connection settings (`argilla.api_url`, `api_key`) belong on the env/CLI layer — see [Secrets](#secrets) below.

## Configuration layers

Pragmata distinguishes three configuration layers, each with a different
home. Conflating them (e.g. dropping API keys into a tracked YAML file,
or dropping per-topic settings into the Argilla stack's `.env`) is a
common source of operational confusion.

| Layer | Where it lives | Example values |
|---|---|---|
| **Argilla stack config** | `deploy/annotation/.env` (copied from `.env.dev.example`) | `ARGILLA_PORT`, `POSTGRES_PASSWORD`, `ARGILLA_DATABASE_URL` |
| **Pragmata settings** | Per-topic YAML in `ws_setup_configs/` passed via `--config`, plus env vars and CLI flags | `locale`, `workspace_dataset_map`, `calibration_fraction`, `dataset_id` |
| **Secrets** | Environment variables only — never in any file | `ARGILLA_API_KEY`, `OPENAI_API_KEY` |

### Pragmata settings resolution

For Pragmata settings, the precedence chain is (highest wins):

```
CLI flags / API call overrides   ← one-off overrides
        ↓
Environment variables            ← e.g. PRAGMATA_ANNOTATION_LOCALE, ARGILLA_API_URL
        ↓
Per-call config (--config)       ← deploy/annotation/ws_setup_configs/<topic>.yaml
        ↓
Built-in defaults
```

See [docs/design/infra-package-contracts.md](../../docs/design/infra-package-contracts.md)
for the contracts spec. The design doc additionally describes a
per-user `~/.pragmata/config.yaml` layer between `--config` and
defaults; this is not yet implemented (no caller of `ResolveSettings`
loads it). Per-topic YAML via `--config` covers current needs.

### Secrets

Secrets are resolved separately from settings and **never appear in any
config file** (per [ADR-0008](../../docs/decisions/0008-annotation-interface-auth.md)
and the contracts doc). Set them in your shell, systemd unit, or CI
secret store:

```bash
export ARGILLA_API_KEY=...
```

`core/settings/settings_base.py::resolve_api_key()` raises
`MissingSecretError` if a required key is unset.

## Localisation

Three layers of UI strings are visible to an annotator, with different
locale behaviour. Understanding which is which is the key to
configuring localisation correctly.

| Layer | What it is | Locale source | Live toggle? |
|---|---|---|---|
| **Argilla chrome** | UI Argilla itself ships (navigation, settings menu, button labels like "Submit", "Filters") | Argilla's nuxt-i18n catalog | Yes — user's language menu inside Argilla |
| **Argilla data** | Strings Pragmata pushes into a dataset via the SDK: field titles, question titles, guidelines, label option text | Locked at dataset creation by `--locale` / `PRAGMATA_ANNOTATION_LOCALE` | No — re-create the dataset to change |
| **Pragmata widgets** | HTML/JS Pragmata ships inside Argilla `CustomField` iframes (currently: the discard widget) | Every supported locale is bundled at dataset creation; widget JS picks the active one at runtime by reading Argilla's chrome locale | Yes — listens to Argilla's chrome toggle |

The boundary that matters: **data is server-stored, chrome and our
widgets are not**. Argilla's frontend renders dataset strings verbatim
without knowing about locales, so changing the chrome language doesn't
change them — they were baked in when the dataset was created. Widget
strings are bundled with all locales because we control the JS.

### Configuring the dataset locale

Set the env var for a deploy-wide default:

```bash
export PRAGMATA_ANNOTATION_LOCALE=en
```

…or pass `--locale` per-invocation to `setup` and `import`:

```bash
pragmata annotation setup --locale en --config <topic>.yaml
pragmata annotation import <records> --locale en --config <topic>.yaml
```

Precedence is CLI flag > env > config (`locale: en` in YAML) >
built-in default (`en`). Field/question `name=` identifiers and stored
label *values* (`yes`/`no`, `DiscardReason.*.value`) stay stable
regardless of locale, so exports from differently-localised datasets
merge cleanly.

Supported locales: `en`, `de`. To add another, see
[`core/annotation/locales/`](../../src/pragmata/core/annotation/locales/);
unit tests enforce key completeness against the English source-of-truth
catalog.

## Tearing down

```bash
make docker-down                              # stop stack and drop volumes
pragmata annotation teardown                  # remove datasets + workspaces
pragmata annotation teardown --dataset-id <topic>  # remove only that topic's datasets
```
