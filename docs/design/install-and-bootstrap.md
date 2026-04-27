<!-- TODO: review change -->
# Install, Bootstrap & Config UX

Status: Draft (PR #162 review applied; one open item remaining - see Open questions)
Related: 
- ADR-0007 (packaging & invocation surface), 
- ADR-0012 (install & bootstrap UX) [draft]

## Purpose

- define how users install, configure, and first-run `pragmata`. 
- holistically cover install model, bootstrap commands, config system, and error UX - across all three tools (`annotation`, `querygen`, `eval`)

**Top-level shape** - two buckets w/ orthogonal concerns:

- **§1 API UX** - how the Python/CLI entrypoints accept settings; how config resolves; what users see on error. Applies to all three tools
- **§2 Infra UX** - how `annotation` starts, stops, upgrades, and wires its Docker stack. Applies to `annotation` only

## Guiding principles

1. **Modular install.** Users can install and use any single tool w/o the others. `pip install pragmata[annotation]` must not require querygen or eval deps
2. **Install is side-effect free.** No prompts, no Docker, no network calls at `pip install` time -> configuration happens on first explicit invocation (if at all, see principle 4). follows PyPI policy and `gh auth login` / `supabase` convention.
3. **Fail clearly -> point to fix** - e.g. if optional extras not installed. Never raw traceback on a first-use error. Always: "X is not Y. Run `pragmata ... ` to fix." Follows `gh`, `supabase`, `vercel`, `dbt`, `fly`, `railway`.
<!-- TODO: review change -->
4. **Zero config OOTB.** Defaults work out of the box. Each tool must be usable immediately after install via opinionated defaults. Settings resolution (defaults + env + config + explicit args) is the same on every run - there is no separate "first-run config synthesis" path. Follows `ruff` / `black` (zero-config by design) and `supabase start` (sensible local defaults).
<!-- TODO: review change -->
5. **Run commands are deterministic; interactivity is opt-in.** Normal execution commands (`import`, `export`, `gen-queries`, `up`, `down`, etc.) are non-interactive by default - they read settings from kwargs/flags + env + config and either succeed or fail clearly. Interactivity is reserved for explicit setup/init flows that bootstrap infrastructure or provision external state (e.g. `annotation setup` against a running Argilla server). No general "config wizard" overlay on normal commands.
6. **Tool-scoped complexity.** Per-tool setup commands (`pragmata annotation setup`, `pragmata querygen setup`, `pragmata eval setup`). No global `pragmata setup`. Setup is diff per tool (annotation needs Docker + Argilla creds; querygen needs LLM provider creds; eval tbc).
7. **Dev tooling ≠ production tooling.** `Makefile` is dev-only, prod install path needs proper shell scripts, documented separately.

## Install model

<!-- TODO: review change -->
### Optional extras

```
pip install pragmata                     # bare install - CLI skeleton only
pip install pragmata[annotation]         
pip install pragmata[querygen]           
pip install pragmata[eval]               
```

Extras capture **heavy, optional, provider-specific, or runtime-sensitive** dependencies (e.g. `argilla` SDK, `langchain`, provider clients). Tool ownership alone does not justify an extra: lightweight shared dependencies that support a first-class capability across tools can stay regular core deps. Each extra is audited dependency-by-dependency, not by blanket "if querygen-only → extra."

<!-- TODO: review change -->
### Lazy imports at the package boundary

- optional-extra packages (`argilla`, `langchain`, etc.) must not be imported at module scope on the root import path
- guard the import **narrowly at the actual import site** (typically inside `core/` modules where the dep is used), not by blanket-wrapping entrypoints. Blanket wrappers mask unrelated import errors inside the optional dep itself
- the `ImportError` handler names the missing **package** and points to the **extra** that provides it. Pattern borrowed from [`transformers` error handling](https://github.com/huggingface/transformers/issues/24147):

> `ImportError: 'argilla' is required for pragmata.annotation. Install with: pip install 'pragmata[annotation]'`
> ^^^ this is the "fail loudly and point to fix" principle

Failure modes must be distinguished:

<!-- TODO: review change -->
| Failure | Cause | Error message hint |
|---|---|---|
| **Extra not installed** | `pip install pragmata` (bare) -> user calls `pragmata annotation ...` | "Install with: `pip install pragmata[annotation]`" |
| **Docker missing (annotation only)** | Has extra -> no Docker daemon | "Docker is required for `pragmata annotation`. Install Docker" |

Under the zero-config principle, "installed but unconfigured" is **not** a normal failure mode - defaults are sufficient and resolution is consistent on every run.

> NB: new deps proposed in this doc
>
> | Package | Where it goes | Why | Notes |
> |---|---|---|---|
> | [`platformdirs`](https://platformdirs.readthedocs.io/) | core deps | OS-appropriate user config dir (`~/.config/pragmata/` etc.) | ~50KB, zero deps, PyPA-endorsed (vendored in `pip`) |
>
> **`platformdirs`** - resolves config/cache/data dirs correctly across Linux/macOS/Windows from a single call, so we don't hardcode `~/.config/` (Linux-only). Full rationale and precedent in §1.1.1.
>
> Deferred: **`questionary`** (or any interactive prompt library). Not added for v0.1: there is no general config wizard (principle #5). Existing `pragmata annotation setup` (Argilla provisioning) is headless-flag-driven and stays that way. Revisit only if a concrete interactive flow appears (e.g. annotation provisioning that validates against a running server and benefits from masked-secret prompts), and even then scope to the `[annotation]` extra - never core, never querygen/eval.

---

# §1 API UX

How pragmata's Python/CLI entrypoints accept settings, resolve config, and present errors. Applies to all three tools uniformly.

## 1.1 Config system

TODO confirm this as open question: 
  - propose to add layer to existing `ResolveSettings.resolve` chain 
  - layer order, merge mechanics (`deep_merge`, `prune_unset`, `UNSET`), and `resolve_api_key` contract would stay unchanged

*Existing chain kept as-is:*
`overrides > env > config > defaults`
- `overrides`: call-site kwargs (CLI flags land here as kwargs)
- `env`: environment-derived layer
- `config`: YAML loaded via `load_config_file(config_path)`; caller passes `config_path` explicitly
- `defaults`: pydantic model defaults
- Secrets resolved separately via `resolve_api_key()` (env-only, never read from `config`)

<!-- TODO: review change -->
*Proposed addition (non-breaking - same layer order, adds auto-discovery where the layer is currently empty):*

**Auto-discover `config_path`** when the caller doesn't pass one. Resolution becomes:

1. If the caller passes an explicit `config_path` → use that, skip auto-discovery entirely
2. Otherwise → walk up from cwd for a project-local `./pragmata.yaml` or `pyproject.toml [tool.pragmata]` (first match wins)
3. Otherwise → fall back to `platformdirs.user_config_dir("pragmata") / "config.yaml"`

**Explicit beats implicit:** an explicit `config_path` overrides *all* auto-discovered config (project + user). Project-only auto-discovery only fires when no explicit path is passed. See §1.1.3, §1.1.4.

Resulting effective chain (additions in *italics*):

```
overrides
  > env
  > explicit config_path (if provided)
  > auto-discovered project config (./pragmata.yaml or pyproject.toml [tool.pragmata])
  > auto-discovered user config (~/.config/pragmata/config.yaml)
  > defaults
```

### 1.1.1 Location

Uses [`platformdirs`](https://platformdirs.readthedocs.io/) to resolve OS-appropriate user directories:

`platformdirs.user_config_dir("pragmata")`:

- Linux: `~/.config/pragmata/`
- macOS: `~/Library/Application Support/pragmata/`
- Windows: `%APPDATA%\pragmata\`

Then `PRAGMATA_CONFIG_DIR` env var overrides. Follows `poetry`, `wandb`, `huggingface_hub`.

>Rejected: `~/.pragmata/` (legacy single-dot convention of `dbt`, `aws`, `kubectl`). Fine, but greenfield 2026 Python tools use XDG.

**Reasoning:**

*Why user-level config dir as the base layer (not cwd):*
- pragmata is library-called-from-user-scripts + CLI - users run it from arbitrary directories, so making cwd the *only* config location breaks reproducibility
- Credentials (Argilla API keys, provider keys) should persist across projects - device-wide config avoids re-auth per repo
- Project-level overrides sit *above* the user-level config in the precedence chain (§1.1.3), so repos can still pin their own settings when needed - we get both persistent device-wide defaults and repo-scoped overrides.

*Why `platformdirs` (not hardcoded `~/.config/pragmata/`):*
- Hardcoding `~/.config/` is Linux-correct but wrong on macOS (`~/Library/Application Support/`) and Windows (`%APPDATA%\`). `platformdirs` resolves all three from one call.
- PyPA-endorsed and vendored in `pip`- don't reinvent the wheel here.
- Respects `XDG_CONFIG_HOME` on Linux for free

*Why XDG over legacy dotfile:*
- `~/.pragmata/` is the pre-XDG convention (`dbt`, `aws`, `kubectl`) (= fine but dated)
- greenfield 2026 Python tools (`poetry`, `wandb`, `huggingface_hub`, `ruff`) all use XDG, aligning pragmata with the ecosystem users already have in `~/.config/`
- XDG separates config from cache from data - future-proofs a cache dir (`~/.cache/pragmata/`) without polluting the same tree

*Why `PRAGMATA_CONFIG_DIR` override:*
- standard escape hatch for CI, containers, multi-user machines, testing - every mature CLI exposes one
- Cheap, zero downside, unblocks unpredictable use cases

<!-- TODO: review change -->
### 1.1.2 Files

Single non-secret config file under the resolved user config dir:

```
~/.config/pragmata/
└── config.yaml         # Non-secret settings, per-tool sections. Safe to share.
```

Secrets are not stored in any pragmata-owned file (§1.1.5). LLM provider keys go through env vars; Argilla delegates to its own credential store.

`config.yaml` shape - single file, per-tool top-level sections (gcloud/kubectl/dbt pattern):

```yaml
annotation:
  argilla_url: http://localhost:6900
  workspace: default
  # ...
querygen:
  provider: openai
  model: gpt-4o
  # ...
eval:
  # ...
```

Rationale: multi-service CLIs (aws, gcloud, dbt, kubectl) use one-file-with-sections (per-tool files = anti-pattern). Modularity preserved because each tool reads only its own section.

### 1.1.3 Project-level config override

Users can pin per-repo settings in either:

- `./pragmata.yaml` (same shape as the user-level `config.yaml` - per-tool top-level sections), or
- `pyproject.toml` under `[tool.pragmata]` (same shape, nested under the TOML table)

Resolution walks up from cwd until it finds one of these (or hits the filesystem root). First match wins; the two sources are not merged with each other. Project-level values override user-level `~/.config/pragmata/config.yaml`; everything above in the precedence chain (CLI flags, env vars, credentials file) still wins over project-level.

**TODO: open question - both formats?** `pragmata.yaml` mirrors user-level file shape so snippets copy 1:1 between device and repo. `pyproject.toml` is the PEP 518 standard home for Python tool config - repos that already centralise config there (ruff, mypy, pytest) can keep pragmata in the same file rather than adding another dotfile.

**Why not merge the two.** Merging a `pragmata.yaml` with a `pyproject.toml` in the same repo -> two sources of truth, ambiguous precedence. First match wins is the same rule `ruff` uses.

<!-- TODO: review change -->
**Secrets still excluded.** Same rule as `config.yaml` - project-level files are commit-safe; secrets only flow via kwargs/flags or env vars (§1.1.5).

Precedent: [ruff](https://docs.astral.sh/ruff/configuration/) (supports both `ruff.toml` and `pyproject.toml [tool.ruff]`, first match wins), [dbt](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml) (project `dbt_project.yml` + user `~/.dbt/profiles.yml`), [black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#configuration-via-a-file) (walks up for `pyproject.toml`).

<!-- TODO: review change -->
### 1.1.4 Precedence chain

Universal pattern (aws/terraform/kubectl/dbt/pip), extended with a project layer (§1.1.3). **Explicit beats implicit at every step**, including the explicit-config-path slot:

```
CLI flag / kwarg
  >  env var
  >  explicit config_path (if passed)
  >  auto-discovered project config (./pragmata.yaml or pyproject.toml [tool.pragmata])
  >  auto-discovered user config (~/.config/pragmata/config.yaml)
  >  built-in defaults
```

Secrets follow a separate, narrower chain (env-only for LLM providers; Argilla delegates to `~/.cache/argilla/credentials` after env). No pragmata-owned credential store for v0.1. See §1.1.5.

**Proposed** env var prefix for non-secret tool settings: `PRAGMATA_<TOOL>_<KEY>`, e.g. `PRAGMATA_ANNOTATION_ARGILLA_URL`.
- follows [gcloud's `CLOUDSDK_SECTION_PROPERTY`](https://docs.cloud.google.com/sdk/docs/properties).
- rejected: pydantic-settings' `PRAGMATA__ANNOTATION__URL` double-underscore - less shell-friendly, harder to type.

> *Current state*: no systematic `PRAGMATA_*` prefix exists yet. Today the codebase reads a few hardcoded env vars directly (notably `ARGILLA_API_URL` in [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py)). The proposal is to add prefix-based resolution *alongside* canonical provider env vars (secrets), not to replace the latter.

**Secrets are the exception - they use canonical provider env vars, not the `PRAGMATA_*` prefix.** See §1.1.5; already implemented in `API_KEY_ENV_VARS` / `resolve_api_key()`.

<!-- TODO: review change -->
Small set of shared top-level vars (outside per-tool scoping):
- `PRAGMATA_CONFIG_DIR` - escape hatch overriding the `platformdirs` config location (CI, containers, multi-user machines, testing)

> Rejected: `PRAGMATA_WORKSPACE_DIR`. Workspace/base dir is already an explicit kwarg/flag with cwd default; a global env override would create hidden state without clear benefit, and is structurally inconsistent with the per-tool `PRAGMATA_<TOOL>_<KEY>` scheme.

<!-- TODO: review change -->
### 1.1.5 Secrets

Aligns with the existing `resolve_api_key()` contract (`core/settings/settings_base.py`). Canonical provider env vars are the source of truth - not a `PRAGMATA_*`-prefixed name. **No pragmata-owned credential store for v0.1**: a persistent local secret store is a much larger product/security surface (perms, redaction, rotation, cross-platform handling) than non-secret config discovery, and is deferred until concrete demand exists.

**Resolution chain for LLM providers** (OpenAI, Anthropic, Mistral, Cohere, DeepSeek, Google):

```
kwarg  >  canonical env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)  >  MissingSecretError
```

**Resolution chain for Argilla** (delegates to Argilla's own credential store after env, no pragmata-owned fallback):

```
kwarg  >  ARGILLA_API_KEY env  >  ~/.cache/argilla/credentials (Argilla's own store)  >  MissingSecretError
```

Argilla's client already writes to `~/.cache/argilla/credentials` on `argilla login`. We delegate to it rather than maintaining a parallel store, which would create rotation and precedence confusion for the same service.

Secrets are **never read from `config.yaml`** or any project/user pragmata config file: enforced by shape (e.g. `ArgillaSettings` holds `api_url` only; no `api_key` field exists on settings models).

> Deferred (not v0.1): a pragmata-owned `~/.config/pragmata/credentials` file. If demand materialises, it would need explicit decisions on permissions, redaction, rotation/update behaviour, cross-platform handling, and precedence relative to provider-native stores.

<!-- TODO: review change -->
### 1.1.6 Config templates (deferred)

No interactive config editor for v0.1. If users want to materialise a config file, the lean option (deferred until demand) is a non-interactive command that emits a commented template:

```
pragmata <tool> configure --write-template > pragmata.yaml
```

This is a string template the CLI owns - documented defaults plus comments naming each env var equivalent. No prompts, no idempotent merging, no `aws configure`-style "current value" detection. Users edit the file in their editor; the precedence chain (§1.1.4) does the rest.

**Deferred entirely for v0.1:** the template-writer command itself, named profiles (`--profile staging`), and any interactive config flow.

## 1.2 Entrypoint signatures

The Python API entrypoints and the CLI commands share one contract: kwargs/flags + env + config, resolved identically.

shape:
> For `annotation_setup`, `annotation_import`, `annotation_export`: Argilla URL / API key / workspace / dataset settings / other client options, injected via args, env, and config (secrets only via args and env).


from issue #39, already implemented:

- Public API signatures accept tool-specific settings as kwargs with `UNSET` defaults
- `config_path` kwarg for config file override
- Internal resolution: `ToolSettings.resolve(config=..., overrides={...})` (§1.1)
- Secrets via args or env only; never in config files (§1.1.5)

**CLI ↔ API coupling.** CLI commands are thin wrappers over the API - flags land as kwargs with `UNSET` for un-passed; [ADR-0007](../decisions/0007-packaging-invocation-surface.md).

**What this means for external-service wiring.** Users should be able to pass anything relevant for the client, including external service URLs (e.g. an external Postgres connection string) - this is the Infra UX side of the same contract (§2.2.3: `--external-postgres <url>` etc.). The API entrypoint accepts them as kwargs; CLI surfaces them as flags; env-var and config-file fallbacks work the same way as for every other setting.

<!-- TODO: review change -->
## 1.3 CLI surface

- `annotation setup` keeps its **existing semantics**: provisioning Argilla workspaces/users/datasets against a running server. Headless-flag-driven, not a config wizard
- `annotation` adds stack lifecycle verbs (`up`/`down`) because it orchestrates Docker - other tools don't
- `querygen` and `eval` have no `setup` verb. Credentials and provider selection flow through args/env/config like every other setting; no per-tool setup proliferation

Full surface for v0.1:

```
pragmata annotation up          # starts local Docker stack (zero-config defaults) (NEW)
pragmata annotation down        # stops local Docker stack (NEW)
pragmata annotation setup       # CURRENT: Argilla workspace/user/dataset provisioning (unchanged)
pragmata annotation teardown    # CURRENT: tear down provisioned Argilla state (unchanged)
pragmata annotation import      
pragmata annotation export     
pragmata annotation iaa         

pragmata querygen gen-queries    # runtime - works OOTB if provider env var present

pragmata eval run (tbc)              
```

> *Current implementation state* 
>
> - `pragmata --version`, `--verbose`
> - `pragmata querygen gen-queries`
> - `pragmata annotation setup/teardown/import/export/iaa` - `setup`/`teardown` target Argilla workspace provisioning (already as documented above)

Naming rationale:

- `setup` / `teardown` - kept as-is. Provisioning/de-provisioning verbs against an already-running Argilla server. Analogue: `aws iam create-user`, `kubectl create namespace`. Headless and flag-driven; not interactive.
- `up`/`down` - Docker stack lifecycle. `docker compose up` inheritance is clear. Chosen over `start`/`stop` (supabase) because `up` preserves the Compose mental model.
- No `pragmata init`, no global setup, no per-tool config wizards. Settings flow: kwargs/flags + env + config (§1.1.4).

<!-- TODO: review change -->
### 1.3.1 Headless by default

All commands - including `annotation setup` (Argilla provisioning) - are non-interactive: required values come from flags, env vars, or config. Missing required values fail fast with a clear error that names the missing setting and the flag/env var that supplies it.

```
pragmata annotation setup --url http://... --api-key ...   # headless, no prompts
pragmata annotation setup --url http://...                 # missing api-key → fail fast with clear error
```

No prompts, no TTY-dependent branching, no `--no-prompt` mode flag (none needed - nothing prompts). Same behaviour in interactive shells, CI, and Python API callers.

## 1.4 First-use error UX

All runtime commands (`import`, `export`, `gen-queries/generate`, `run`, etc.) validate preconditions before doing work and fail clearly if any are missing. No raw tracebacks.

NB this mostly covers `annotation` tool as this has most complex failure mode chain. Other tools just check extra-installed.

```
$ pragmata annotation import foo.json      # bare pip install pragmata
Error: pragmata.annotation requires the 'annotation' extra.
  Install with: pip install 'pragmata[annotation]'

$ pragmata annotation up                    # no Docker daemon
Error: Docker daemon not reachable.
  Start your Docker runtime and try again.
  See: https://docs.docker.com/get-docker/

<!-- TODO: review change -->
$ pragmata annotation import foo.json
Error: Argilla stack is not running.
  Run: pragmata annotation up
```

Under zero-config (principle 4), "annotation is not configured" is **not** a failure mode - defaults always resolve to a working config.

Failure mode is checked in order: *extra-installed → Docker-running → stack-up.* Each check is a strict prerequisite for the next - stack-up can't succeed without Docker-running. We fail at the first missing prerequisite with the corresponding fix.

---

# §2 Infra UX

<!-- TODO: review change -->
Covers `pragmata annotation` only (other tools don't require same infra):
- stack composition (incl Docker Compose profiles & file distribution)
- default is bundled and automatic; customisation flows through CLI flags / env / config (§2.2.3) - no user-edited compose, no shell helpers
- first-run, upgrade, uninstall
- prod bootstrap


> NB: The `Makefile` targets (`docker-up`, `docker-down`, `test-stack`) are **dev-only** - they bind to `deploy/annotation/docker-compose.dev.yml` and assume a cloned repo + `make` (= non-starter for Windows, PyPI installs, and unattended/prod environments). The CLI command `pragmata annotation up` is the single end-user entry point and must work identically across all three.

SOTA for PyPI-distributed CLIs that wrap Docker stacks (Supabase CLI, Airbyte `abctl`, Dagster, Prefect, MLflow, Argilla itself) converges on a few points: 
- install is side-effect free, 
- first-run bootstrap is an idempotent single command, 
- upgrade is the #1 pain point (we mostly skip this), 
- dev ≠ prod.

<!-- TODO: review change -->
## 2.1 Stack composition

The runtime compose file ships as **package data under `src/pragmata/annotation/docker-compose.yml`** and is resolved at runtime via `importlib.resources`. Users never see or edit the YAML - all supported customisation flows through CLI flags / env / config (§2.2.3). This is the "locked compose" model (option Y in §2.2.1); see that section for the full rationale and rejected alternatives.

All services bundled by default (zero-config principle). Each backing service (postgres/elasticsearch/redis) is opt-out-able via a Compose profile (§2.2.3).

**Single shipped compose file, not a dev/prod pair.** Surveyed PyPI-distributed Docker-wrapping tools (Supabase, Airbyte `abctl`, Airflow, Dagster, Prefect, MLflow) all converge on **one shipped compose artefact**. Nobody ships parallel `docker-compose.prod.yml` + `docker-compose.dev.yml` side by side because it forces users to answer "which one do I run?" before doing anything. The split here is:

- **Shipped (package data):** `src/pragmata/annotation/docker-compose.yml` - production-first: pinned tags, env-driven credentials (no hardcoded defaults), sensible resource defaults, localhost-only port bindings. This is the file `pragmata annotation up` resolves at runtime.
- **Contributor dev (cloned repo):** `deploy/annotation/docker-compose.dev.override.yml` (proposed - does not exist yet) - layered on top of the shipped file via Makefile targets using `docker compose -f ... -f ...`. Typical contents: well-known default creds (`argilla`/`1234`), stdout logging, looser health-check timing, exposed debug ports.

End users (`pragmata annotation up`) only ever touch the shipped (package-data) file via CLI flags. The dev override is exclusively for contributors working in a cloned repo.

This matches Airflow's [documented pattern](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) ("compose file is a quick-start, not a production config" - with dev niceties layered via overrides) and keeps us from maintaining two sources of truth.

Migration steps from today's single `deploy/annotation/docker-compose.dev.yml`:
  1. Extract prod-safe defaults into `src/pragmata/annotation/docker-compose.yml` as package data (the new SSOT for runtime)
  2. Strip dev-only overrides into `deploy/annotation/docker-compose.dev.override.yml`
  3. Update Makefile targets to stack both via `docker compose -f ... -f ...`

## 2.2 Compose file distribution

Two axes to decide: (a) where the compose file the daemon reads actually lives, (b) how many "bundles" the shipped file supports.

<!-- TODO: review change -->
### 2.2.1 Where the compose file lives (daemon-reads-from)

Three options were considered:

| Option | What it is | Default-path UX | Power-user UX | Upgrade drift | SOTA precedent |
|---|---|---|---|---|---|
| **Y. Locked (compose stays inside package) - *recommended*** | Resolve via `importlib.resources` at runtime; user never sees the YAML. Overrides only via CLI flags / `config.yaml` / env vars we expose. | Zero-config | Customisation surface = `--external-postgres`, `--external-elastic`, port flags, etc. (§2.2.3) - sufficient for v0.1 | None - we own the file | [Supabase CLI](https://github.com/supabase/cli) (went further: constructs the project programmatically in Go, no compose file at all) |
| **X. User-editable (default copy, drift-flagged)** | First `up` copies packaged YAML → user config dir. User may edit. On subsequent `up`, drift is flagged. | Zero-config: first-run user never touches YAML | Standard Docker mental model: edit the YAML | Real - flagged, user resolves manually | [dbt `profiles.yml`](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml), [VS Code `settings.json`](https://code.visualstudio.com/docs/getstarted/settings) |
| **Z. Eject** | Start with Y; `pragmata annotation eject` copies compose out and pragmata then uses the ejected copy, warning the user they own it from there | Zero-config | Explicit escape hatch, clean managed-vs-owned contract | None for non-ejected users; ejected users own drift | [create-react-app `eject`](https://create-react-app.dev/docs/available-scripts#npm-run-eject), [Expo eject-to-bare-workflow](https://docs.expo.dev/archive/customizing/) |

**Recommendation: Y for v0.1.** Users should not have to understand or edit Docker Compose YAML on the supported paths. The customisation surface that matters - external Postgres/Elasticsearch URLs, port bindings, image tags - is exposed through CLI flags / env / config (§2.2.3). If that surface is sufficient (and for v0.1 it is), keeping the compose file package-owned avoids drift, simplifies upgrades, and prevents feature-creep where every Compose field becomes a CLI flag.

- **X** was previously recommended but creates an ownership/drift problem on day one and makes the happy path Docker-centric. Reserve user-owned compose for an explicit advanced escape hatch (i.e. option Z), not the default.
- **Z** is a clean future escape hatch. Document the pattern, but do **not** build the `eject` verb until concrete demand materialises (>2 users blocked on something the §2.2.3 flag surface can't express).

>Related rejected patterns: generate compose from a template at install time (two sources of truth, drifts on upgrade); remote URL fetch (breaks offline installs, trust boundary). Supabase [issue #2435](https://github.com/supabase/cli/issues/2435) documents the specific pain of user-editable compose + CLI tight version coupling - option Y avoids the problem entirely.

<!-- TODO: review change -->
### 2.2.2 Distribution mechanism (how pkg'd YAML travels)

Ships as package data inside the installed package at `src/pragmata/annotation/docker-compose.yml`, resolved at runtime via `importlib.resources.files("pragmata.annotation") / "docker-compose.yml"` + `as_file()`. Same mechanism already in use for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html).

Image tags are pinned in the shipped compose and treated as package-owned. `pip install -U pragmata` ships a new compose with new tags - users automatically pick it up on next `up` (no drift, no warning needed under option Y).

### 2.2.3 Profiles / bundles (the flag surface for external backing services)

Explicitly supports **Docker Compose profiles + external backing service URLs**. We use Compose's built-in `profiles` feature. Profile names carry forward from the current dev compose unchanged: `all-bundled`, `external-pg`, `external-es`.


```yaml
services:
  argilla:
    image: argilla/argilla-server:<pinned>
    depends_on: [postgres, elasticsearch, redis]
    profiles: [all-bundled, external-pg, external-es]

  worker:
    image: argilla/argilla-server:<pinned>
    command: argilla worker
    profiles: [all-bundled, external-pg, external-es]

  postgres:
    profiles: [all-bundled, external-es]   # skipped when profile is external-pg
    image: postgres:<pinned>
    # ...

  elasticsearch:
    profiles: [all-bundled, external-pg]   # skipped when profile is external-es
    image: docker.elastic.co/elasticsearch/elasticsearch:<pinned>
    # ...

  redis:
    profiles: [all-bundled, external-pg, external-es]
    image: redis:<pinned>
    # ...
```

**Proposed v0.1.0 CLI flag surface (minimal):**
```
pragmata annotation up                                    # all-bundled profile (zero-config default)
pragmata annotation up --external-postgres <url>          # external-pg profile, wire Argilla to external PG
pragmata annotation up --external-elastic <url>           # external-es profile, wire Argilla to external ES
```

Internally: `--external-postgres` = select `external-pg` profile + inject `ARGILLA_DATABASE_URL`; `--external-elastic` = select `external-es` profile + inject `ARGILLA_ELASTICSEARCH`. Settings resolution (§1.1.4) applies as normal - flag > env > config > default.

Precedent for profiles: [Airflow's official Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) ships profiles (`flower`, etc.). [Dagster Helm values](https://github.com/dagster-io/dagster/tree/master/helm/dagster) mirrors the same pattern at a different abstraction level.

<!-- TODO: review change -->
## 2.3 First-run UX

**`pragmata annotation up` is the first-run command. No separate `init`. `annotation setup` stays as Argilla provisioning (§1.3), invoked after the stack is up.**

- Pre-flight in order: extra installed → Docker daemon reachable → required ports free
- Resolve the packaged compose via `importlib.resources` (no copy to disk - option Y, §2.2.1)
- Pulls images on first invocation (slow; log clearly - make this prominent in docs)
- Health-polls Argilla's health endpoint with a timeout
- On success: prints URL, default API key, and next command

^^^ most of the core parts of this are already implemented

Precedent: `supabase start`, `abctl local install`, `prefect server start`. Idempotent single command, safe to re-run.

<!-- TODO: review change -->
## 2.4 Upgrade

**`pip install -U pragmata` is the sole upgrade primitive. The compose file is package-owned (option Y, §2.2.1), so upgrades pick up the new file automatically with no drift to manage.**

- Named Docker volumes with a deterministic prefix (`pragmata_annotation_*`) persist data across container recreation
- New compose ships with new image tags - `pragmata annotation up` after upgrade picks up the new file via `importlib.resources`. No drift detection, no warning, no `reset-compose` verb needed (we own the file)
- For destructive Argilla schema migrations between majors, document the backup step; pragmata cannot protect users from upstream-breaking changes

>Research note: Airbyte's `abctl local install` is idempotent and designed to fix Compose upgrade brittleness ([Airbyte discussion #40599](https://github.com/airbytehq/airbyte/discussions/40599)). We don't need that machinery here - the locked compose model sidesteps the brittleness it was built to address.

## 2.5 Uninstall

**`pragmata annotation down` stops the stack; `pragmata annotation down --volumes` additionally wipes data. No global `pragmata uninstall`.**

- `pip uninstall pragmata` removes the package
- `~/.config/pragmata/` removal is documented but user-owned. No cleanup verb.

Precedent: `abctl local uninstall [--persisted]` - explicit data-wipe flag is the industry norm.

<!-- TODO: review change -->
## 2.6 Prod bootstrap / unattended install

**Decision for v0.1: no shipped script. Document the two-line install (`pipx install 'pragmata[annotation]' && pragmata annotation up`) in the README.** If demand materialises later (>2 deployers asking for unattended install support), add option C below - skip a static shell script.

### 2.6.1 Scope

pragmata ships three tools. Only `annotation` has any bootstrap beyond `pip install`:
- `querygen`: `pipx install 'pragmata[querygen]' && export OPENAI_API_KEY=...` - two-line README
- `eval`: same shape as `querygen`
- `annotation`: real sequence (install → wait for Docker → pull images → start stack → poll health → print creds)

The question is whether `annotation` needs a separate install artefact. For v0.1, no.

### 2.6.2 Options surveyed

| Option | What it is | SOTA precedent | Our cost | Verdict |
|---|---|---|---|---|
| **A. No script (docs only) - *chosen for v0.1*** | Docs snippet: `pipx install 'pragmata[annotation]' && pragmata annotation up` | Every Tier-1 PyPI-distributed Docker-wrapping tool surveyed ([Supabase](https://supabase.com/docs/guides/local-development/cli/getting-started), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/get-started/install), [Dagster](https://docs.dagster.io/getting-started/install), [MLflow](https://mlflow.org/docs/latest/tracking.html)) | None | **Adopt** |
| **B. `scripts/bootstrap-annotation.sh` in repo** | Static shell file checked into git, linked from docs | [Docker convenience script](https://github.com/docker/docker-install), [rustup](https://rustup.rs/), [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) | Maintenance drift - every CLI flag change invalidates it. Docker's own convenience installer is [famously not recommended for production](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) for exactly this reason | **Skip** |
| **C. `pragmata annotation print-install-script`** | CLI prints a shell transcript pinned to the installed CLI version; user redirects to file and executes | Novel - no surveyed tool does this | Trivial: a string template the CLI owns. Always version-matched | **Defer** - reach for if/when demand appears, jump straight here from A (don't pass through B) |

If later we do need explicit unattended recipes for cloud-init / systemd / Kubernetes: document in `docs/deploy/` per-environment, don't package. Helm charts exist for [Dagster](https://artifacthub.io/packages/helm/dagster/dagster) and [Prefect](https://artifacthub.io/packages/helm/prefecthq/prefect-server) as separate artefacts - same pattern if we ever need one.

## 2.7 Cross-platform runtime

**Rely on the generic `docker compose` CLI on PATH. Remain agnostic to the user's Docker runtime.**

- Pre-flight check: `docker version` succeeds (daemon reachable). If not, fail clear: *"Docker daemon not reachable. Start your Docker runtime and try again."*
- No `--runtime` flag, no auto-detection, no runtime-specific branching. Which Docker implementation the user has is not our concern.

Precedent: Supabase and `abctl` both run over the generic `docker` CLI with no engine-specific logic.

## 2.8 Error taxonomy (extension of §1.4)

Beyond the generic cases in §1.4, `annotation up` must also handle:
- port conflict (print occupying process if detectable)
- image-pull failure (network / registry)
- Argilla health-poll timeout (print the container log tail)
- compose-file missing from package (indicates broken install - `pip install --force-reinstall 'pragmata[annotation]'`)

---

## Current codebase baseline

| Area | Implemented today | Proposed in this doc |
|---|---|---|
<!-- TODO: review change -->
| **Config resolution** | `ResolveSettings.resolve(overrides, env, config, defaults)`, `load_config_file()`, `resolve_api_key()` (env-only), `API_KEY_ENV_VARS`, `MissingSecretError` - all in [`core/settings/`](../../src/pragmata/core/settings/) | Add auto-discovery (project-level `./pragmata.yaml` / `pyproject.toml [tool.pragmata]` → user-level `~/.config/pragmata/config.yaml`), `PRAGMATA_<TOOL>_<KEY>` env prefix. *No pragmata-owned credentials file (deferred); Argilla delegates to `~/.cache/argilla/credentials` after env (§1.1.5).* |
| **Config file path** | Explicit `--config` flag only; no XDG, no `platformdirs` | `platformdirs.user_config_dir("pragmata")` + `PRAGMATA_CONFIG_DIR` override |
| **Docker stack lifecycle (`up`/`down`)** | Not implemented - only Makefile targets (`docker-up`, `docker-down`, etc.) that read [`deploy/annotation/docker-compose.dev.yml`](../../deploy/annotation/docker-compose.dev.yml) | Add `pragmata annotation up` / `down` CLI commands |
| **`annotation setup` / `teardown`** | Exist - manage **Argilla workspaces, users, datasets** (not Docker). See [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py) | **Semantics unchanged**; `up`/`down` is an *additional* pair of verbs for stack lifecycle |
| **Compose file distribution** | Dev-only file in `deploy/`; **not shipped** in the installed wheel | Ship `src/pragmata/annotation/docker-compose.yml` as **package data, locked** (option Y, §2.2.1) - resolved at runtime via `importlib.resources`, never copied to disk. Dev override stays in `deploy/` for contributors only |
| **Package data** | `importlib.resources` already used for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html) | Same mechanism for the compose file |
| **Interactive prompts** | None - `setup` is headless-flag-driven only | **No change.** No `questionary`, no wizard for v0.1 (§1.3.1) |


<!-- TODO: review change -->
## Open questions

Most of the previously-open questions are resolved by the v0.1 framing in this doc. Recap:

| ID | Question | Resolution |
|---|---|---|
| §Q-prod-script | Ship a per-tool bootstrap script for `annotation`? | **No for v0.1** (§2.6). Document the two-line install. Add option C if demand materialises. |
| §Q-wizard-lib | Questionary or minimal prompts? | **Neither for v0.1** (§1.3.1). No interactive wizard - all commands are headless. Revisit only if a concrete interactive flow appears. |
| §Q-auth-split | Separate `auth` / `login` command, or part of `setup`? | **Neither for v0.1** (§1.1.5). No pragmata-managed auth/login. Argilla delegates to its own credential store after env; LLM providers stay env-only. |
| §Q-argilla-creds | Delegate to Argilla's credential store or maintain our own? | **Delegate** (§1.1.5). One source of truth per service. |
| §Q-setup-verb | Rename `annotation setup` to free up `setup` for a config wizard? | **No** (§1.3). Existing `annotation setup` semantics retained (Argilla provisioning); no config wizard needed. |
| §Q-wizard-placement | Where and how do wizards appear? | **They don't, for v0.1** (§1.3.1). Reopen only with a concrete interactive flow. |
| §Q-configured-check | Include a "configured" pre-flight check? | **No** (§1.4). Under zero-config, "installed but unconfigured" is not a failure mode. |

Remaining open items:

- **Two project-config formats?** §1.1.3 supports both `./pragmata.yaml` and `pyproject.toml [tool.pragmata]` (first-match-wins, ruff pattern). Worth confirming we want the maintenance surface of both vs. picking one.

## References

- [ADR-0007 - Packaging & invocation surface](../decisions/0007-packaging-invocation-surface.md)
- [ADR-0003 - Infra: self-hosted only](../decisions/0003-infra-self-hosted-only.md)
- Precedent CLIs: [`gh auth login`](https://cli.github.com/manual/gh_auth_login), [Supabase CLI](https://supabase.com/docs/reference/cli/introduction), [`dbt init`](https://docs.getdbt.com/reference/commands/init), [AWS CLI config](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html), [gcloud properties](https://docs.cloud.google.com/sdk/docs/properties), [platformdirs](https://platformdirs.readthedocs.io/)
- Precedent for Docker orchestration + compose distribution: [Supabase CLI](https://github.com/supabase/cli), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/), [Dagster](https://docs.dagster.io/), [Airflow Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
