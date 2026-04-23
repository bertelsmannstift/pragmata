# Install, Bootstrap & Config UX

Status: Draft (very much in discussion!)
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
4. **Zero config OOTB.** Setup is optional; defaults work out of the box. Each tool must be usable immediately after install via opinionated defaults - no `setup` required for the happy path. `setup` exists for experienced users who need to override defaults (custom Argilla URL, alternate LLM provider, non-default workspace, etc.). First-run behaviour synthesises a working config from defaults + env vars; the wizard is a convenience, not a gate. Follows `ruff` / `black` (zero-config by design) and `supabase start` (sensible local defaults, no wizard required)
5. **Flags suppress prompts.** Interactive wizard by default; any required flag present short-circuits the corresponding prompt. Same command, both use cases. Follows [`gh auth login`](https://cli.github.com/manual/gh_auth_login).
6. **Tool-scoped complexity.** Per-tool setup commands (`pragmata annotation setup`, `pragmata querygen setup`, `pragmata eval setup`). No global `pragmata setup`. Setup is diff per tool (annotation needs Docker + Argilla creds; querygen needs LLM provider creds; eval tbc).
7. **Dev tooling ≠ production tooling.** `Makefile` is dev-only, prod install path needs proper shell scripts, documented separately.

## Current codebase baseline

| Area | Implemented today | Proposed in this doc |
|---|---|---|
| **Config resolution** | `ResolveSettings.resolve(overrides, env, config, defaults)`, `load_config_file()`, `resolve_api_key()` (env-only), `API_KEY_ENV_VARS`, `MissingSecretError` - all in [`core/settings/`](../../src/pragmata/core/settings/) | Add auto-discovery (project-level `./pragmata.yaml` / `pyproject.toml [tool.pragmata]` → user-level `~/.config/pragmata/config.yaml`), credentials-file fallback, `PRAGMATA_<TOOL>_<KEY>` env prefix |
| **Config file path** | Explicit `--config` flag only; no XDG, no `platformdirs` | `platformdirs.user_config_dir("pragmata")` + `PRAGMATA_CONFIG_DIR` override |
| **Docker stack lifecycle (`up`/`down`)** | Not implemented - only Makefile targets (`docker-up`, `docker-down`, etc.) that read [`deploy/annotation/docker-compose.dev.yml`](../../deploy/annotation/docker-compose.dev.yml) | Add `pragmata annotation up` / `down` CLI commands |
| **`annotation setup` / `teardown`** | Exist - but manage **Argilla workspaces, users, datasets** (not Docker). See [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py) | Semantics unchanged; `up`/`down` is an *additional* pair of verbs  (alternative: keep it infra-tied, e.g. infra up/down, or something similar) |
| **Compose file distribution** | Dev-only file in `deploy/`; **not shipped** in the installed wheel | Ship `src/pragmata/annotation/docker-compose.yml` as package data (single file, prod-first, dev override via separate file - see §2.2)* |
| **Package data** | `importlib.resources` already used for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html). | Same mechanism for the compose file |
| **`questionary` / wizards** | No wizard today - `setup` is headless-flag-driven only | Add `questionary` under `[annotation]` (or other tools as needed)|

*proposing to move from deploy/ to annotation/ to keep infra bundled with tool.
## Install model

### Optional extras, one per tool

```
pip install pragmata                     # bare install - CLI skeleton only
pip install pragmata[annotation]         # + argilla (already exists); + platformdirs, questionary (proposed)
pip install pragmata[querygen]           # + langchain, sentence-transformers (already exists)
pip install pragmata[eval]               # + eval deps (proposed - eval tool not yet implemented)
pip install pragmata[all]                # convenience: everything (proposed - does not exist yet)
```

### Lazy imports at the package boundary

- optional-extra packages (`argilla`, `langchain`, etc.) must not be imported at module scope on the root import path. 
- each tool's public API entry point wraps its optional dep in a `try/except ImportError` and re-raises with the exact `pip install` command. Pattern borrowed from [`transformers` error handling](https://github.com/huggingface/transformers/issues/24147):

> `ImportError: pragmata.annotation requires the 'annotation' extra. Install with: pip install 'pragmata[annotation]'`
> this is the fail loudly and point to fix principle

Failure modes must be distinguished:

| Failure | Cause | Error message hint |
|---|---|---|
| **Extra not installed** | `pip install pragmata` (bare) -> user calls `pragmata annotation setup` | "Install with: `pip install pragmata[annotation]`" |
| **Docker missing (annotation only)** | Has extra + config -> no Docker daemon | "Docker is required for `pragmata annotation`. Install Docker" |
| *TODO confirm:* **"Installed but unconfigured"** would not occur happen (due to zero-config principle)| Has extra -> hasn't run setup | "Run `pragmata annotation setup` first" |


> NB: new deps proposed in this doc
>
> | Package | Where it goes | Why | Notes |
> |---|---|---|---|
> | [`platformdirs`](https://platformdirs.readthedocs.io/) | core deps | OS-appropriate user config dir (`~/.config/pragmata/` etc.) | ~50KB, zero deps, PyPA-endorsed (vendored in `pip`) |
> | [`questionary`](https://questionary.readthedocs.io/) | `[annotation]` extra (and/or `[querygen]`, `[eval]` if they gain wizards -> core dep?) | Interactive setup wizard prompts (`pragmata annotation setup` etc.) | See Open question §Q-wizard-lib in this doc: commit to Questionary or use a lighter prompt lib? Prompt-toolkit conflicts largely resolved in 2.x |
>
> **`platformdirs`** - resolves config/cache/data dirs correctly across Linux/macOS/Windows from a single call, so we don't hardcode `~/.config/` (Linux-only). Full rationale and precedent in §1.1.1.
>
> **`questionary`** - renders interactive `setup` wizard prompts (text, select, confirm, password) on top of `prompt_toolkit`. Chosen for `gh`/`supabase`-grade wizard ergonomics (arrow-key selects, masked secrets, validators, skip-on-flag) w/o us reimplementing TTY handling. Alternatives considered: [`rich.prompt`](https://rich.readthedocs.io/en/stable/prompt.html) (no arrow-key select, no masked input UX - too bare for a multi-field wizard); [`prompt_toolkit`](https://python-prompt-toolkit.readthedocs.io/) directly (what Questionary wraps - more power, much more boilerplate); [`InquirerPy`](https://inquirerpy.readthedocs.io/) (feature-equivalent fork of PyInquirer, also `prompt_toolkit`-based - viable swap-in if Questionary stalls); [`click.prompt`](https://click.palletsprojects.com/en/stable/prompts/) (Typer's built-in - line-by-line only, no select widget).

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

*Proposed additions (non-breaking - same layer order, just adds fallbacks where layers are currently empty):*

1. **Auto-discover `config_path`** when the caller doesn't pass one -> walk up from cwd for a project-local `./pragmata.yaml` or `pyproject.toml [tool.pragmata]`; fall back to `platformdirs.user_config_dir("pragmata") / "config.yaml"`. Explicit `config_path` still wins, so every current caller behaves identically. See §1.1.3.
2. **Add a credentials-file fallback for `resolve_api_key()`** below env vars -> if the env var is unset, read from `~/.config/pragmata/credentials` before raising `MissingSecretError`. Env still wins.

> TODO review ^^^

Resulting effective chain (additions in *italics*):
`overrides > env > `*`credentials file (secrets only, new fallback)`*` > `*`project config (./pragmata.yaml or pyproject.toml [tool.pragmata])`*` > user config (explicit config_path, else `*`auto-discovered ~/.config/pragmata/config.yaml`*`) > defaults`

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

### 1.1.2 Files

Two files, roles separated (here borrowing from AWS pattern - [`~/.aws/config`](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html) + `~/.aws/credentials`):

```
~/.config/pragmata/
├── config.yaml         # Non-secret settings, per-tool sections. Safe to share.
└── credentials         # Secrets only (API keys). chmod 600. Never shared.
```

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

**Secrets still excluded.** Same rule as `config.yaml` - project-level files are commit-safe; secrets only live in env vars or `~/.config/pragmata/credentials` (§1.1.5).

Precedent: [ruff](https://docs.astral.sh/ruff/configuration/) (supports both `ruff.toml` and `pyproject.toml [tool.ruff]`, first match wins), [dbt](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml) (project `dbt_project.yml` + user `~/.dbt/profiles.yml`), [black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#configuration-via-a-file) (walks up for `pyproject.toml`).

### 1.1.4 Precedence chain

Universal pattern (aws/terraform/kubectl/dbt/pip), extended with a project layer (§1.1.3):

```
CLI flag  >  env var  >  credentials file (secrets)  >  project config (./pragmata.yaml or pyproject.toml)  >  user config (~/.config/pragmata/config.yaml)  >  built-in defaults
```

**Proposed** env var prefix for non-secret tool settings: `PRAGMATA_<TOOL>_<KEY>`, e.g. `PRAGMATA_ANNOTATION_ARGILLA_URL`.
- follows [gcloud's `CLOUDSDK_SECTION_PROPERTY`](https://docs.cloud.google.com/sdk/docs/properties).
- rejected: pydantic-settings' `PRAGMATA__ANNOTATION__URL` double-underscore - less shell-friendly, harder to type.

> *Current state*: no systematic `PRAGMATA_*` prefix exists yet. Today the codebase reads a few hardcoded env vars directly (notably `ARGILLA_API_URL` in [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py)). The proposal is to add prefix-based resolution *alongside* canonical provider env vars (secrets), not to replace the latter.

**Secrets are the exception - they use canonical provider env vars, not the `PRAGMATA_*` prefix.** See §1.1.5; already implemented in `API_KEY_ENV_VARS` / `resolve_api_key()`.

Small set of shared top-level vars (outside per-tool scoping, all proposed):
- `PRAGMATA_CONFIG_DIR` - override config location
- `PRAGMATA_WORKSPACE_DIR` - override workspace root
- TODO: define here

### 1.1.5 Secrets

Aligns with the existing `resolve_api_key()` contract (`core/settings/settings_base.py`). Canonical provider env vars are the source of truth - not a `PRAGMATA_*`-prefixed name.

*Existing (implemented):*

1. CLI flag / kwarg (`--api-key ...` lands as `api_key` kwarg on the API layer)
2. Canonical provider env var from `API_KEY_ENV_VARS`:
   - `ARGILLA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`
3. Missing env var → `MissingSecretError` raised by `resolve_api_key()`

Secrets are **never read from `config.yaml`**: enforced by shape (e.g. `ArgillaSettings` holds `api_url` only; no `api_key` field exists on settings models).

*Proposed addition (non-breaking):*

Slot a `~/.config/pragmata/credentials` file (`chmod 600`) *between* env and `MissingSecretError`. Extends `resolve_api_key()` to read the file if the env var is unset, before raising. Env still wins. Current env-only callers would behave same.

Effective chain after addition (same order; addition in *italics*):
`kwarg > canonical env var > `*`~/.config/pragmata/credentials`*` > MissingSecretError`

See Open question §Q-argilla-creds for the Argilla-specific sub-question (delegate to Argilla's own credential store vs maintain our own).

### 1.1.6 Idempotent re-setup

`aws configure` pattern: detect existing values, show as defaults in brackets, empty input keeps current.

```
? Argilla URL [current: http://localhost:6900]:
? API key [current: **** (set)]:
```

No `--force` flag. No `--reset` for v0.1.0. Re-running setup is always safe.

**Deferred:** named profiles (`--profile staging`) - this is AWS pattern, but overkill / we use `make` targets for dev.

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

## 1.3 CLI surface

- Each tool has its own config wizard verb
- `annotation` also has stack lifecycle verbs (`up`/`down`) because it orchestrates Docker - other tools don't

**Terminology conflict to resolve.** The existing `pragmata annotation setup` command already exists and provisions **Argilla workspaces/users/datasets** against a running server (see [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py)) - it is *not* a config wizard. The proposed config-wizard semantics in this doc therefore cannot reuse `setup` without breaking. Two resolutions, see §Q-setup-verb:

- *Option A (recommended):* rename existing `annotation setup` → `annotation provision` (or `init-workspace`), free up `setup` for the config wizard.
- *Option B:* call the config wizard `configure` instead. Existing `setup` keeps its workspace-provisioning meaning.

Proposed full surface (assuming Option A):

```
pragmata annotation setup       # OPTIONAL config wizard -> overrides defaults, writes config (NEW)
pragmata annotation up          # starts local Docker stack (works without setup - defaults) (NEW)
pragmata annotation down        # stops local Docker stack (NEW)
pragmata annotation provision   # CURRENT `setup` renamed: Argilla workspace/user/dataset provisioning
pragmata annotation import      # runtime - needs stack up (EXISTS)
pragmata annotation export      # runtime - needs stack up (EXISTS)
pragmata annotation iaa         # runtime - needs stack up (EXISTS)

pragmata querygen setup         # OPTIONAL config wizard (NEW)
pragmata querygen generate      # runtime - works OOTB if provider env var present
                                # (currently `gen-queries` - see Open question §Q-generate-naming)

pragmata eval setup             # OPTIONAL config wizard (NEW - eval tool itself doesn't exist yet)
pragmata eval run               # runtime (NEW)
```

> *Current implementation state* (see the baseline table at the top for the complete gap map)
>
> - `pragmata --version`, `--verbose`
> - `pragmata querygen gen-queries`
> - `pragmata annotation setup/teardown/import/export/iaa`  - existing `setup`/`teardown` target Argilla workspace provisioning, not Docker or config-wizard UX

Naming rationale (see research notes):

- `setup` - one-shot config wizard that resolves config and writes it. Idempotent. Rerunnable. Chosen over `init` because `init` connotes "scaffold a project in cwd" in supabase/dbt/wrangler convention, which is not what we do. Also, it should work using defaults OOTB -> not a first init, but a later configuration by experienced users.
- `up`/`down` - Docker stack lifecycle. `docker compose up` inheritance is clear. Chosen over `start`/`stop` (supabase) because `up` preserves the Compose mental model
- `provision` / `teardown` - proposed names for workspace/user management (currently `setup`/`teardown`). Analogue: `aws iam create-user`, `kubectl create namespace` - provisioning operations against an already-running service. See §Q-setup-verb.
- NB: no `pragmata init` / no global setup. Per-tool setups scale better than a mega-wizard that branches (the `dbt init` branching pattern only works when setup is structurally identical across plugins - not the case here).

### 1.3.1 Wizard trigger: flags suppress prompts

Single command does both interactive and headless. `gh auth login` pattern:

```
pragmata annotation setup                                  # full wizard
pragmata annotation setup --url http://... --api-key ...   # headless, no prompts
pragmata annotation setup --url http://...                 # partial - prompt for missing fields only
```

No `--non-interactive` mode flag: flags themselves are the escape hatch.

Caveat from research: `gh`'s implementation has a known gap - partial flags + non-TTY stdin hangs -> we need:
- **TTY detection**: if `not sys.stdin.isatty()` and required values missing -> fail fast with headless-flags error, don't hand to Questionary (TODO come back to this)
- **Optional `--no-prompt`** as belt-and-braces for CI - refuses to prompt regardless of TTY state.

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

$ pragmata annotation import foo.json
Error: Argilla stack is not running.
  Run: pragmata annotation up

NB: given OOTB works with opinionated defaults principle, we would NOT have following. (TODO: confirm this):

$ pragmata annotation import foo.json      
Error: annotation is not configured.
  Run: pragmata annotation setup
```

Failure mode is checked in order: **extra-installed -> Docker-running -> stack-up.** Each check is a strict prerequisite for the next - stack-up can't succeed without Docker-running. We fail at the first missing prerequisite with the corresponding fix.

> TODO: if we do decide we need initial configure, slot `configured` between `extra-installed` and `Docker-running` -> extra-installed -> configured -> Docker-running -> stack-up. Pure-Python check (does the config resolve?), so it's cheaper to fail fast there before spinning up a Docker subprocess.

Extended error taxonomy for `annotation up` specifically: see §2.5.

---

# §2 Infra UX

Covers `pragmata annotation` only: stack composition, compose file distribution, first-run, upgrade, uninstall, prod bootstrap, cross-platform runtime, error taxonomy.

Infra UX scope:
> Docker Compose profiles, external backing service URLs. Default is bundled and automatic, customised is configured via env/config and executed via bash script helpers.

The `Makefile` targets (`docker-up`, `docker-down`, `test-stack`) are **dev-only** - they bind to `deploy/annotation/docker-compose.dev.yml` and assume a cloned repo + `make` (= non-starter for Windows, PyPI installs, and unattended/prod environments). The CLI command `pragmata annotation up` is the single end-user entry point and must work identically across all three.

SOTA for PyPI-distributed CLIs that wrap Docker stacks (Supabase CLI, Airbyte `abctl`, Dagster, Prefect, MLflow, Argilla itself) converges on a few points: install is side-effect free, first-run bootstrap is an idempotent single command, upgrade is the #1 pain point, and dev ≠ prod.

## 2.1 Stack composition

Bundled stack (mirrors the existing [`deploy/annotation/docker-compose.dev.yml`](../../deploy/annotation/docker-compose.dev.yml)):

All bundled by default (zero-config principle). Each backing service (postgres/elasticsearch/redis) is opt-out-able via a Compose profile - see §2.2 for the user surface.

**Single shipped compose file, not a dev/prod pair.** Surveyed PyPI-distributed Docker-wrapping tools (Supabase, Airbyte `abctl`, Airflow, Dagster, Prefect, MLflow) all converge on **one shipped compose artefact** - either a single file or a programmatic equivalent. Nobody ships parallel `docker-compose.prod.yml` + `docker-compose.dev.yml` files side by side because it forces users to answer "which one do I run?" before they've done anything. Instead:

- The **shipped** compose (`src/pragmata/annotation/docker-compose.yml`, see §2.2) is **production-first**: pinned tags, env-driven credentials (no hardcoded defaults), sensible resource defaults, localhost-only port bindings.
- **Dev-only conveniences** live in a companion [`deploy/annotation/docker-compose.dev.override.yml`](../../deploy/annotation/docker-compose.dev.override.yml) (proposed - does not exist yet) that Makefile targets apply on top via `docker compose -f ... -f ...`. Typical contents: well-known default creds (`argilla`/`1234`), stdout logging, looser health-check timing, exposed debugging ports.
- End users (`pragmata annotation up`) only ever touch the shipped file. The override is for contributors working in a cloned repo.

This matches Airflow's [documented pattern](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) ("compose file is a quick-start, not a production config" - with dev niceties layered via overrides) and keeps us from maintaining two sources of truth.

Migration note: today's `docker-compose.dev.yml` serves both roles (it's the only file). The migration is (a) extract prod-safe defaults into `src/pragmata/annotation/docker-compose.yml` as package data, (b) strip the dev-only overrides into `docker-compose.dev.override.yml`, (c) update Makefile targets to stack both.

## 2.2 Compose file distribution

Two axes to decide: (a) where the compose file the daemon reads actually lives, (b) how many "bundles" the shipped file supports.

### 2.2.1 Where the compose file lives (daemon-reads-from)

Three options:

> NB: for context:
> - Y is SOTA, but (overly?) rigid. Z is a nice workaround (also SOTA), but adds extra CLI verb and requires extending mental model. X is comfortable middlde ground for us.
> - If X, then the concern becomes how to manage drift on compose updates (three -way merges if not careful - see Supabase [issue #2435](https://github.com/supabase/cli/issues/2435)), however as we don't expect to be updating compose much (/we have a relatively simple setup) this isn't a major stumbling block.

| Option | What it is | Default-path UX | Power-user UX | Upgrade drift | SOTA precedent |
|---|---|---|---|---|---|
| **X. User-editable (default copy, drift-flagged) - *recommended*** | First `up` copies packaged YAML → `$PRAGMATA_CONFIG_DIR/annotation/docker-compose.yml`. User may edit. *On subsequent `up`, if user's file differs from packaged, print warning listing changed keys + path to new packaged file, proceed with user's file unchanged (no auto-reset/merge tool - YAGNI)* | Zero-config: first-run user never touches YAML | Standard Docker mental model: edit the YAML, `up` uses it | Minimal: flagged, user resolves manually. Rare since we don't expect frequent compose updates. | [dbt `profiles.yml`](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml), [VS Code `settings.json`](https://code.visualstudio.com/docs/getstarted/settings), [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) - ship opinionated defaults as a starting point, user owns their copy |
| **Y. Locked (compose stays inside package)** | Resolve via `importlib.resources` at runtime; user never sees the YAML. Overrides only via CLI flags / `config.yaml` entries we expose. | Zero-config | Rigid - every customisation becomes a feature request or an undocumented workaround | None - we own the file | [Supabase CLI](https://github.com/supabase/cli) (went further: constructs the project programmatically in Go, no compose file at all) |
| **Z. Eject** | Start with Y; `pragmata annotation eject` copies compose out and pragmata then uses the ejected copy, warning the user they own it from there | Zero-config | Explicit escape hatch, clean managed-vs-owned contract | None for non-ejected users; ejected users own drift | [create-react-app `eject`](https://create-react-app.dev/docs/available-scripts#npm-run-eject), [Expo eject-to-bare-workflow](https://docs.expo.dev/archive/customizing/), [Next.js manual config extraction](https://nextjs.org/docs) |

**Recommendation (X).** User-editable default, preserves standard Docker contract (compose YAML is editable), keeps zero-config happy-path intact (packaged file IS the default), and researchers who want to change a port open the file like they would for any other Docker stack. Drift mitigation is deliberately minimal - we do not expect to ship compose updates often; over-engineering for rare upgrades is waste.

- **Y** breaks  Docker contract researchers expect and creates feature-creep pressure to expose every Compose field as a flag. 
- **Z** adds a verb users don't understand without docs.

>Related rejected patterns: generate compose from a template at `setup` time (two sources of truth, drifts on upgrade); remote URL fetch (breaks offline installs, trust boundary). Supabase [issue #2435](https://github.com/supabase/cli/issues/2435) documents the specific pain of user-editable compose + CLI tight version coupling - our mitigation is loose version coupling (diff-flag only, no CLI-enforced pins).

### 2.2.2 Distribution mechanism (how pkg'd YAML travels)

Ships as package data inside the installed package (`pragmata/annotation/docker-compose.yml`), resolved at runtime via `importlib.resources.files(...) / "docker-compose.yml"` + `as_file()`. Copied on first `up` per §2.2.1 (option X).

Image tags are pinned in the shipped compose and treated as package-owned. `pip install -U pragmata` ships a new compose with new tags. Users on the default copy pick up the new file; users who edited see the drift warning (§2.3).

### 2.2.3 Profiles / bundles (the flag surface for external backing services)

Explicitly supports **Docker Compose profiles + external backing service URLs**. We use Compose's built-in `profiles` feature. Profile names carry forward from the current dev compose unchanged: `all-bundled`, `external-pg`, `external-es`.

Shipped compose structure (names match [`deploy/annotation/docker-compose.dev.yml`](../../deploy/annotation/docker-compose.dev.yml)):

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

See Open question §Q-profile-flags for minimal-vs-full passthrough.

Precedent for profiles: [Airflow's official Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) ships profiles (`flower`, etc.). [Dagster Helm values](https://github.com/dagster-io/dagster/tree/master/helm/dagster) mirrors the same pattern at a different abstraction level.

## 2.3 First-run UX

**`pragmata annotation up` is the first-run command. No separate `init`; `setup` stays optional for overrides.**

- Pre-flight in order: extra installed → Docker daemon reachable → required ports free (§1.4 taxonomy)
- On first `up`: copy packaged compose → `$PRAGMATA_CONFIG_DIR/annotation/docker-compose.yml` (§2.2.1)
- Pulls images on first invocation (slow; log clearly - make this prominent in docs)
- Health-polls Argilla's health endpoint with a timeout
- On success: prints URL, default API key, and next command

Precedent: `supabase start`, `abctl local install`, `prefect server start`. Idempotent single command, safe to re-run.

## 2.4 Upgrade

**`pip install -U pragmata` is the sole upgrade primitive. Stay lean - users own their compose; we flag drift, nothing more.**

- Named Docker volumes with a deterministic prefix (`pragmata_annotation_*`) persist data across container recreation
- On `up` after an upgrade, if the user's `$PRAGMATA_CONFIG_DIR/annotation/docker-compose.yml` diverges from the packaged one:
  - Print a warning naming the divergent keys + the path to the new packaged file (via `importlib.resources` - printable one-liner)
  - Proceed with the user's file unchanged
- No auto-reset, no `reset-compose` verb, no merge tool. Surveyed tools (dbt, Prefect, Dagster, Supabase) don't ship a "reset my config" verb either - users copy from docs or rerun `init`.
- For destructive Argilla schema migrations between majors, document the backup step; pragmata cannot protect users from upstream-breaking changes.

Precedent: Airbyte's `abctl local install` is idempotent and was specifically designed to fix Compose upgrade brittleness ([Airbyte discussion #40599](https://github.com/airbytehq/airbyte/discussions/40599)). We skip the heavier machinery - we're not shipping frequent compose changes, so engineering for rare upgrades is waste.

## 2.5 Uninstall

**`pragmata annotation down` stops the stack; `pragmata annotation down --volumes` additionally wipes data. No global `pragmata uninstall`.**

- `pip uninstall pragmata` removes the package
- `~/.config/pragmata/` removal is documented but user-owned. No cleanup verb.

Precedent: `abctl local uninstall [--persisted]` - explicit data-wipe flag is the industry norm.

## 2.6 Prod bootstrap / unattended install

### 2.6.1 Scope

pragmata ships three tools. Only `annotation` has bootstrap beyond `pip install`:
- `querygen`: `pipx install 'pragmata[querygen]' && export OPENAI_API_KEY=...` - not script-worthy, two-line README
- `eval`: same shape as `querygen`
- `annotation`: real sequence (install → wait for Docker → pull images → start stack → poll health → print creds) - script-worthy *if* we ship one

So "should pragmata ship a bootstrap script" is really "should `annotation` ship one." The other tools don't need it.

### 2.6.2 Two positions to reconcile

These two sources diverge on whether to ship a bash script. Laying out both.

| Source | What they suggest | Implied implementation | Rationale |
|---|---|---|---|
| **Position A** (prior framing) | "for prod it'll be replaced likely by bash script" + "customised is configured via env/config and executed via bash script helpers" | Some form of shell script replaces Makefile in prod; shell helpers wire env/config into Compose | Makefile is dev-only and needs a prod equivalent; keeps env/config-driven UX while leveraging bash for wiring |
| **Position B** (SOTA survey) | No script for v0.1.0; CLI-only is sufficient | `pipx install` + `pragmata annotation up` documented in README | Every Tier-1 PyPI-distributed Docker-wrapping tool surveyed ([Supabase](https://supabase.com/docs/guides/local-development/cli/getting-started), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/get-started/install), [Dagster](https://docs.dagster.io/getting-started/install), [MLflow](https://mlflow.org/docs/latest/tracking.html)) is CLI-only for prod bootstrap |

**Reading the divergence.** The two positions may be a framing mismatch rather than a real disagreement. Position A's "bash script" phrasing is less specific than it sounds - "*replaced likely by bash script*" reads as an aside; "*bash script helpers*" describes the wiring layer between user config and Compose, not a user-facing install script. If "bash script" means "whatever replaces the Makefile for unattended/prod installs" - that's satisfied by any of options A/B/C below. If it specifically means a shipped `scripts/bootstrap.sh` - that's option B. Worth confirming; the design doesn't hinge on the answer, but option A and option B have different maintenance implications.

### 2.6.3 Options for the user-facing install surface

| Option | What it is | SOTA precedent | Our cost | Who benefits |
|---|---|---|---|---|
| **A. No script (docs only) - *research agent recommended v0.1.0*** | Docs snippet: `pipx install 'pragmata[annotation]' && pragmata annotation up`. Maybe `curl`-able from docs site directly (Docusaurus renders shell). | Every Tier-1 tool surveyed | None | No separate artefact, but IT admins copy-paste two lines from docs |
| **B. `scripts/bootstrap-annotation.sh` in repo** | Static shell file checked into git, linked from docs | [Docker convenience script](https://github.com/docker/docker-install), [rustup](https://rustup.rs/), [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) | Maintenance drift - every CLI flag change invalidates it. Docker's own convenience installer is [famously not recommended for production](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) for exactly this reason. | `curl \| bash` users who want one artefact; the most literal interpretation of Position A's "bash script" instinct |
| **C. `pragmata annotation print-install-script`** | CLI prints a shell transcript pinned to the installed CLI version; user redirects to file and executes | Novel - no surveyed tool does this | Trivial: a string template the CLI owns. Always version-matched. | Admins who want an auditable file before running anything, plus automatic version-pinning |

**Recommendations diverge:**
- *Position B (SOTA survey):* A for v0.1.0; if demand materialises (>2 deployers asking the same "how do I do this unattended" question), jump directly to C. Skip B.
- *Position A (prior framing):* B implied by the "replaced by bash script" phrasing.

**My read:** A is the leanest start. B's maintenance drift is a real cost we'd rather not take on. C is the clever future escape hatch if we ever need one. The divergence above is probably a framing mismatch - "some unattended path" vs strictly a "user-facing install script" - needs confirmation. See Open question §Q-prod-script.

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

## Open questions

Consolidated from inline markers throughout. Cross-referenced from the relevant sections.

### §Q-defaults-per-tool: what are the zero-config defaults per tool?

Principle 4 (setup optional) commits us to zero-config happy paths for each tool. That requires deciding *what* those defaults are:
- **annotation:** `argilla_url=http://localhost:6900`, default workspace, auto-generated API key written to Argilla on `up`. Straightforward - stack is self-contained.
- **querygen:** needs an LLM provider. No defensible "zero-config" default - we can't pick a provider for the user, and the SDK needs a key. Options:
  - (a) Require at least one provider env var (`OPENAI_API_KEY`, `MISTRAL_API_KEY`, etc.); auto-detect provider from which one is set. Fail fast with a clear message if none set: *"querygen needs a provider. Set one of: OPENAI_API_KEY, ... . Or run `pragmata querygen setup`."*
  - (b) Require explicit `setup` for querygen, breaking principle 4 for this one tool.
  - *Recommendation:* (a). Breaks no principle and is how `langchain`/`litellm` work.
- **eval:** likely depends on querygen's provider + possibly a judge model. Same pattern as querygen.

### §Q-argilla-creds: delegate to Argilla's own credential store or maintain our own?

Context: Argilla's own client already writes to [`~/.cache/argilla/credentials`](https://docs.argilla.io/) on `argilla login`.

- *Delegate model:* if user has already run `argilla login`, `pragmata annotation setup` detects that and skips the API key prompt. Keeps one source of truth and composes cleanly with the existing `ARGILLA_API_KEY` env var resolution. Downside: couples us to Argilla's file format + location; breaks if they rename/relocate it; harder to reason about precedence when two stores exist.
- *Maintain our own model:* `~/.config/pragmata/credentials` holds Argilla + LLM provider keys in one place. Consistent resolution pattern across every secret pragmata needs. Downside: duplicates Argilla's store if the user has also run `argilla login`; pragmata has to write/chmod/own the file; users need to update it in two places if they rotate keys.
- *Recommendation:* delegate for Argilla specifically (it's the only provider with its own persistent credential store); maintain our own for LLM providers (no equivalent exists). Resolution order for Argilla becomes: `kwarg > ARGILLA_API_KEY env > ~/.cache/argilla/credentials > ~/.config/pragmata/credentials > MissingSecretError`.

### §Q-profile-flags: minimal external-service flags vs full `--profile` passthrough?

- *Minimal (recommended):* `--external-postgres` / `--external-elastic` only. Covers the real enterprise-deployer case (corp DB), closed surface, no surprises.
- *Full passthrough:* `--profile NAME` passes through to `docker compose`. Flexible, but users can activate profiles we don't expect. Footgun for v0.1.0.

### §Q-prod-script: ship a per-tool bootstrap script for `annotation`?

See §2.6 for full options matrix. Short version:
- *Position B (SOTA survey):* A (no script for v0.1.0); jump to C if demand appears
- *Position A (prior framing):* B implied by "replaced by bash script"
- *Likely framing mismatch* - "bash script" may mean "any non-Make unattended path" rather than specifically a shipped `.sh` file. Confirm before committing.

### §Q-wizard-lib: Questionary or minimal prompts?

Research strongly prefers [Questionary](https://questionary.readthedocs.io/); 2.1 addresses the older prompt_toolkit clash. Alternative: a lighter library for the handful of prompts we actually need.

### §Q-auth-split: `pragmata annotation auth` / `login` separate command, or part of `setup`?

Argilla API keys rotate independently of compose config. `gh` and `supabase` split auth from config; `aws` combines. Research leans split; current draft combines.

### §Q-generate-naming: keep `gen-queries` or rename to `generate`?

Current implementation: `pragmata querygen gen-queries`. Proposal in this doc: `pragmata querygen generate`. Trivial rename; non-breaking if done before v0.1.0.

### §Q-setup-verb: what to call the Argilla workspace-provisioning command?

The existing `pragmata annotation setup` provisions Argilla workspaces/users/datasets against a running server. The proposed config-wizard UX in this doc wants to claim `setup` for itself (matching `gh auth login`, `supabase`, `dbt` convention). They cannot coexist.

- *Option A (recommended):* rename the existing command → `pragmata annotation provision` (or `init-workspace`). Frees `setup` for the config wizard. One-line breaking change before v0.1.0.
- *Option B:* call the config wizard `configure` instead. Existing `setup` stays. Downside: `configure` is less conventional; `aws configure` is the main precedent.

Trivial to execute either way; the question is naming precedence.

### §Q-wizard-placement: where and how do wizards appear?

Still open - noted in guiding principles but not resolved.

### §Q-configured-check: include a "configured" pre-flight check?

If we keep zero-config OOTB, we never need an "installed but unconfigured" error. If we add one, slot it between `extra-installed` and `Docker-running` in §1.4.

## References

- [ADR-0007 - Packaging & invocation surface](../decisions/0007-packaging-invocation-surface.md)
- [ADR-0003 - Infra: self-hosted only](../decisions/0003-infra-self-hosted-only.md)
- Precedent CLIs: [`gh auth login`](https://cli.github.com/manual/gh_auth_login), [Supabase CLI](https://supabase.com/docs/reference/cli/introduction), [`dbt init`](https://docs.getdbt.com/reference/commands/init), [AWS CLI config](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html), [gcloud properties](https://docs.cloud.google.com/sdk/docs/properties), [platformdirs](https://platformdirs.readthedocs.io/)
- Precedent for Docker orchestration + compose distribution: [Supabase CLI](https://github.com/supabase/cli), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/), [Dagster](https://docs.dagster.io/), [Airflow Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
