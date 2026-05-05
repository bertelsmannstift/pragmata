# Config & Settings UX

Status: Draft
Related:
- ADR-0007 (packaging & invocation surface)
- ADR-0012 (install & bootstrap UX) [draft]
- [`annotation-bootstrap.md`](annotation-bootstrap.md) - annotation-only stack lifecycle, compose distribution, prod bootstrap

## Purpose

How the Python/CLI entrypoints accept settings, resolve config, and present errors. Applies uniformly to all three packaged tools (`annotation`,§ `querygen`, `eval`).

Annotation-specific stack lifecycle (Docker, compose, `up`/`down`, prod bootstrap) is in [`annotation-bootstrap.md`](annotation-bootstrap.md).

## Guiding principles

1. **Modular install.** Users can install and use any single tool without the others. `pip install pragmata[annotation]` must not require querygen or eval deps.
2. **Install is side-effect free.** No prompts, no Docker, no network calls at `pip install` time. Configuration happens on first explicit invocation (if at all - see principle 4). Follows PyPI policy and `gh auth login` / `supabase` convention.
3. **Fail clearly, point to fix.** Never raw traceback on a first-use error. Always: "X is not Y. Run `pragmata ... ` to fix." Follows `gh`, `supabase`, `vercel`, `dbt`, `fly`, `railway`.
4. **Zero config OOTB.** Defaults work out of the box. Each tool must be usable immediately after install via opinionated defaults. Settings resolution (defaults + env + config + explicit args) is the same on every run - there is no separate "first-run config synthesis" path. Follows `ruff` / `black` (zero-config by design) and `supabase start` (sensible local defaults).
5. **Run commands are deterministic; interactivity is opt-in.** Normal execution commands (`import`, `export`, `gen-queries`, `up`, `down`, etc.) are non-interactive by default - they read settings from kwargs/flags + env + config and either succeed or fail clearly. Interactivity is reserved for explicit setup/init flows that bootstrap infrastructure or provision external state (e.g. `annotation setup` against a running Argilla server). No general "config wizard" overlay on normal commands.
6. **No per-tool config wizards.** Only `annotation` has a `setup` verb (Argilla provisioning), inherited from current behaviour. `querygen` and `eval` configure entirely through args/env/config; no per-tool config commands.

## Install model

### Optional extras

```
pip install pragmata                     # bare install - CLI skeleton only
pip install pragmata[annotation]         
pip install pragmata[querygen]           
pip install pragmata[eval]               
```

Extras capture heavy, optional, provider-specific, or runtime-sensitive dependencies (e.g. `argilla` SDK, `langchain`, provider clients). Tool ownership alone does not justify an extra: lightweight shared dependencies that support a first-class capability across tools can stay regular core deps; each extra is audited dependency-by-dependency, not by blanket "if querygen-only → extra."

### Lazy imports at the package boundary

- optional-extra packages (`argilla`, `langchain`, etc.) must not be imported at module scope on the root import path
- guard the import narrowly at the actual import site (typically inside `core/` modules where the dep is used), not by blanket-wrapping entrypoints. Blanket wrappers mask unrelated import errors inside the optional dep itself
- the `ImportError` handler names the missing **package** and points to the **extra** that provides it. Pattern borrowed from [`transformers` error handling](https://github.com/huggingface/transformers/issues/24147):

> `ImportError: 'argilla' is required for pragmata.annotation. Install with: pip install 'pragmata[annotation]'`. This is the "fail loudly and point to fix" principle.

Failure modes must be distinguished:

| Failure | Cause | Error message hint |
|---|---|---|
| **Extra not installed** | `pip install pragmata` (bare) -> user calls `pragmata annotation ...` | "Install with: `pip install pragmata[annotation]`" |
| **Docker missing (annotation only)** | Has extra -> no Docker daemon | "Docker is required for `pragmata annotation`. Install Docker" |

Under the zero-config principle, "installed but unconfigured" is **not** a normal failure mode - defaults are sufficient and resolution is consistent on every run.

> **New dependdencies proposed in this doc**
>
> | Package | Where it goes | Why | Notes |
> |---|---|---|---|
> | [`platformdirs`](https://platformdirs.readthedocs.io/) | core deps | OS-appropriate user config dir (`~/.config/pragmata/` etc.) | ~50KB, zero deps, PyPA-endorsed (vendored in `pip`) |
>
> **`platformdirs`** - resolves config/cache/data dirs correctly across Linux/macOS/Windows from a single call, so we don't hardcode `~/.config/` (Linux-only). Full rationale and precedent in §1.1.
>
> Not planned: **`questionary`** (or any interactive prompt library). There is no general config wizard (principle 5) and no per-tool interactive setup flow for querygen/eval. `pragmata annotation setup` (Argilla provisioning) is headless-flag-driven and stays that way.

---

## 1. Config system

Build on the existing `ResolveSettings.resolve` chain (`core/settings/`). Layer order, merge mechanics (`deep_merge`, `prune_unset`, `UNSET`), and `resolve_api_key` contract stay unchanged.

*Existing chain kept as-is:*
`overrides > env > config > defaults`
- `overrides`: call-site kwargs (CLI flags land here as kwargs)
- `env`: environment-derived layer
- `config`: YAML loaded via `load_config_file(config_path)`; caller passes `config_path` explicitly
- `defaults`: pydantic model defaults
- Secrets resolved separately via `resolve_api_key()` (env-only, never read from `config`)

*Proposed addition (non-breaking - same layer order, adds auto-discovery where the layer is currently empty):*

**Auto-discover `config_path`** when the caller doesn't pass one. Resolution becomes:

1. If the caller passes an explicit `config_path` → use that, skip auto-discovery entirely
2. Otherwise → walk up from cwd for a project-local `./pragmata.yaml` or `pyproject.toml [tool.pragmata]` (first match wins)
3. Otherwise → fall back to `platformdirs.user_config_dir("pragmata") / "config.yaml"`

**Explicit beats implicit:** an explicit `config_path` overrides *all* auto-discovered config (project + user). Project-only auto-discovery only fires when no explicit path is passed. See §1.3, §1.4.

Resulting effective chain:

```
  > overrides
  > env
  > explicit config_path (if provided)
  > auto-discovered project config (./pragmata.yaml or pyproject.toml [tool.pragmata])
  > auto-discovered user config (~/.config/pragmata/config.yaml)
  > defaults
```

### 1.1 Location

Uses [`platformdirs`](https://platformdirs.readthedocs.io/) to resolve OS-appropriate user directories:

`platformdirs.user_config_dir("pragmata")`:

- Linux: `~/.config/pragmata/`
- macOS: `~/Library/Application Support/pragmata/`
- Windows: `%APPDATA%\pragmata\`

Then `PRAGMATA_CONFIG_DIR` env var overrides. Follows `poetry`, `wandb`, `huggingface_hub`.

>Rejected: `~/.pragmata/` (legacy single-dot convention). Greenfield 2026 Python tools use XDG.

**Reasoning:**

*Why user-level config dir as the base layer (not cwd):*
- pragmata is library-called-from-user-scripts + CLI - users run it from arbitrary directories, so making cwd the *only* config location breaks reproducibility
- Credentials (Argilla API keys, provider keys) should persist across projects - device-wide config avoids re-auth per repo
- Project-level overrides sit *above* the user-level config in the precedence chain (§1.3), so repos can still pin their own settings when needed - we get both persistent device-wide defaults and repo-scoped overrides.

*Why `platformdirs` (not hardcoded `~/.config/pragmata/`):*
- Hardcoding `~/.config/` is Linux-correct but wrong on macOS (`~/Library/Application Support/`) and Windows (`%APPDATA%\`). `platformdirs` resolves all three from one call.
- PyPA-endorsed and vendored in `pip` - don't reinvent the wheel here.
- Respects `XDG_CONFIG_HOME` on Linux for free

*Why XDG over legacy dotfile:*
- `~/.pragmata/` is the pre-XDG convention (`dbt`, `aws`, `kubectl`) (= fine but dated)
- greenfield 2026 Python tools (`poetry`, `wandb`, `huggingface_hub`, `ruff`) all use XDG, aligning pragmata with the ecosystem users already have in `~/.config/`
- XDG separates config from cache from data - future-proofs a cache dir (`~/.cache/pragmata/`) without polluting the same tree

*Why `PRAGMATA_CONFIG_DIR` override:*
- standard escape hatch for CI, containers, multi-user machines, testing - every mature CLI exposes one
- Cheap, zero downside, unblocks unpredictable use cases

### 1.2 Files

Single non-secret config file under the resolved user config dir:

```
~/.config/pragmata/
└── config.yaml         # Non-secret settings, per-tool sections. Safe to share.
```

Secrets are not stored in any pragmata-owned file (§1.5). LLM provider keys go through env vars; Argilla delegates to its own credential store.

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

### 1.3 Project-level config override

Users can pin per-repo settings in either:

- `./pragmata.yaml` (same shape as the user-level `config.yaml` - per-tool top-level sections), or
- `pyproject.toml` under `[tool.pragmata]` (same shape, nested under the TOML table)

Resolution walks up from cwd until it finds one of these (or hits the filesystem root). **The walk stops at the first match** - only one file is ever read per run. A repo that has both `pragmata.yaml` and `pyproject.toml [tool.pragmata]` will always use whichever the walk hits first; the other is ignored. Project-level values override user-level `~/.config/pragmata/config.yaml`; everything above in the precedence chain (CLI flags, env vars) still wins over project-level.

**Both formats supported.** The two formats are alternative conventions a repo author picks between, not parallel sources that are merged. `pragmata.yaml` mirrors user-level file shape so snippets copy 1:1 between device and repo. `pyproject.toml` is the PEP 518 standard home for Python tool config - repos that already centralise config there (ruff, mypy, pytest) can keep pragmata in the same file rather than adding another dotfile. Same rule as ruff: supports both `ruff.toml` and `pyproject.toml [tool.ruff]`, walk stops at first hit, no merging.

**Secrets still excluded.** Same rule as `config.yaml` - project-level files are commit-safe; secrets only flow via kwargs/flags or env vars (§1.5).

Precedent: [ruff](https://docs.astral.sh/ruff/configuration/) (supports both `ruff.toml` and `pyproject.toml [tool.ruff]`, first match wins), [dbt](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml) (project `dbt_project.yml` + user `~/.dbt/profiles.yml`), [black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#configuration-via-a-file) (walks up for `pyproject.toml`).

### 1.4 Precedence chain

Universal pattern (aws/terraform/kubectl/dbt/pip), extended with a project layer (§1.3). **Explicit beats implicit at every step**, including the explicit-config-path slot:

```
CLI flag / kwarg
  >  env var
  >  explicit config_path (if passed)
  >  auto-discovered project config (./pragmata.yaml or pyproject.toml [tool.pragmata])
  >  auto-discovered user config (~/.config/pragmata/config.yaml)
  >  built-in defaults
```

Secrets follow a separate, narrower chain (env-only for LLM providers; Argilla delegates to `~/.cache/argilla/credentials` after env). No pragmata-owned credential store for v0.1. See §1.5.

**Proposed** env var prefix for non-secret tool settings: `PRAGMATA_<TOOL>_<KEY>`, e.g. `PRAGMATA_ANNOTATION_ARGILLA_URL`.
- follows [gcloud's `CLOUDSDK_SECTION_PROPERTY`](https://docs.cloud.google.com/sdk/docs/properties).
- rejected: pydantic-settings' `PRAGMATA__ANNOTATION__URL` double-underscore - less shell-friendly, harder to type.

> *Current state*: no systematic `PRAGMATA_*` prefix exists yet. Today the codebase reads a few hardcoded env vars directly (notably `ARGILLA_API_URL` in [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py)). The proposal is to add prefix-based resolution *alongside* canonical provider env vars (secrets), not to replace the latter.

**Secrets are the exception - they use canonical provider env vars, not the `PRAGMATA_*` prefix.** See §1.5; already implemented in `API_KEY_ENV_VARS` / `resolve_api_key()`.

Small set of shared top-level vars (outside per-tool scoping):
- `PRAGMATA_CONFIG_DIR` - escape hatch overriding the `platformdirs` config location (CI, containers, multi-user machines, testing)

> Rejected: `PRAGMATA_WORKSPACE_DIR`. Workspace/base dir is already an explicit kwarg/flag with cwd default; a global env override would create hidden state without clear benefit, and is structurally inconsistent with the per-tool `PRAGMATA_<TOOL>_<KEY>` scheme.

### 1.5 Secrets

Aligns with the existing `resolve_api_key()` contract (`core/settings/settings_base.py`). Canonical provider env vars are the source of truth - not a `PRAGMATA_*`-prefixed name. 


**No pragmata-owned credential store (i.e. `~/.config/pragmata/credentials`) for v0.1**: a persistent local secret store is a much larger product/security surface (perms, redaction, rotation, cross-platform handling) than non-secret config discovery, and is deferred until concrete demand exists.

**Resolution chain for LLM providers** (OpenAI, Anthropic, Mistral, Cohere, DeepSeek, Google):

```
kwarg  >  canonical env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)  >  MissingSecretError
```

**Resolution chain for Argilla** (delegates to Argilla's own credential store after env, no pragmata-owned fallback):

```
kwarg  >  ARGILLA_API_KEY env  >  ~/.cache/argilla/credentials (Argilla's own store)  >  MissingSecretError
```

Argilla's client already writes to `~/.cache/argilla/credentials` on `argilla login`. We delegate to it rather than maintaining a parallel store, which would create rotation and precedence confusion for the same service.

Secrets are never read from `config.yaml` or any project/user pragmata config file: enforced by shape (e.g. `ArgillaSettings` holds `api_url` only; no `api_key` field exists on settings models).

### 1.6 Config templates (deferred)

No interactive config editor for v0.1. If users want to materialise a config file, the lean option (deferred until demand) is a non-interactive command that emits a commented template:

```
pragmata <tool> configure --write-template > pragmata.yaml
```

This is a string template the CLI owns - documented defaults plus comments naming each env var equivalent. No prompts, idempotent merging, or `aws configure`-style "current value" detection. Users edit the file in their editor; the precedence chain (§1.4) does the rest.

**Deferred entirely for v0.1:** both the template-writer command itself, and any interactive config flow.

## 2. Entrypoint signatures

The Python API entrypoints and the CLI commands share one contract: kwargs/flags + env + config, resolved identically.

> For `annotation_setup`, `annotation_import`, `annotation_export`: Argilla URL / API key / workspace / dataset settings / other client options, injected via args, env, and config (secrets only via args and env).

From issue #39, already implemented:

- Public API signatures accept tool-specific settings as kwargs with `UNSET` defaults
- `config_path` kwarg for config file override
- Internal resolution: `ToolSettings.resolve(config=..., overrides={...})` (§1)
- Secrets via args or env only; never in config files (§1.5)

**CLI <-> API coupling.** CLI commands are thin wrappers over the API - flags land as kwargs with `UNSET` for un-passed; [ADR-0007](../decisions/0007-packaging-invocation-surface.md).

**External-service wiring.** Users should be able to pass anything relevant for the client, including external service URLs (e.g. an external Postgres connection string) - the annotation infra layer surfaces these as flags too (see [`annotation-bootstrap.md`](annotation-bootstrap.md) §2.3 for `--external-postgres`, `--external-elastic`). The API entrypoint accepts them as kwargs; CLI surfaces them as flags; env-var and config-file fallbacks work the same way as for every other setting.

## 3. CLI surface

- `annotation setup` keeps its existing semantics: provisioning Argilla workspaces/users against a running server. Headless-flag-driven, not a config wizard.
- `annotation` adds stack lifecycle verbs (`up`/`down`) because it orchestrates Docker - other tools don't. See [`annotation-bootstrap.md`](annotation-bootstrap.md).
- `querygen` and `eval` have no `setup` verb. Credentials and provider selection flow through args/env/config like every other setting; no per-tool setup proliferation.

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
- `up`/`down` - Docker stack lifecycle. `docker compose up` inheritance is clear. Chosen over `start`/`stop` (supabase) because `up` preserves the Compose mental model. See [`annotation-bootstrap.md`](annotation-bootstrap.md).
- No `pragmata init`, no global setup, no per-tool config wizards. Settings flow: kwargs/flags + env + config (§1.4).

### 3.1 Headless by default

All commands - including `annotation setup` (Argilla provisioning) - are non-interactive: required values come from flags, env vars, or config. Missing required values fail fast with a clear error that names the missing setting and the flag/env var that supplies it.

```
pragmata annotation setup --url http://... --api-key ...   # headless, no prompts
pragmata annotation setup --url http://...                 # missing api-key → fail fast with clear error
```

No prompts, TTY-dependent branching, or `--no-prompt` mode flag (none needed - nothing prompts). Same behaviour in interactive shells, CI, and Python API callers.

## 4. First-use error UX

All runtime commands (`import`, `export`, `gen-queries`, `run`, etc.) validate preconditions before doing work and fail clearly if any are missing. No raw tracebacks.

The shared failure modes (apply to every tool):

```
$ pragmata annotation import foo.json      # bare pip install pragmata
Error: pragmata.annotation requires the 'annotation' extra.
  Install with: pip install 'pragmata[annotation]'
```

Under zero-config (principle 4), "annotation is not configured" is **not** a failure mode - defaults always resolve to a working config.

Annotation has additional infra-specific failure modes (Docker daemon, stack-up, port conflicts, image-pull failure, etc.) - see [`annotation-bootstrap.md`](annotation-bootstrap.md) §5.

## 5. Codebase baseline

| Area | Implemented today | Proposed in this doc |
|---|---|---|
| **Config resolution** | `ResolveSettings.resolve(overrides, env, config, defaults)`, `load_config_file()`, `resolve_api_key()` (env-only), `API_KEY_ENV_VARS`, `MissingSecretError` - all in [`core/settings/`](../../src/pragmata/core/settings/) | Add auto-discovery (project-level `./pragmata.yaml` / `pyproject.toml [tool.pragmata]` → user-level `~/.config/pragmata/config.yaml`), `PRAGMATA_<TOOL>_<KEY>` env prefix. *No pragmata-owned credentials file (deferred); Argilla delegates to `~/.cache/argilla/credentials` after env (§1.5).* |
| **Config file path** | Explicit `--config` flag only; no XDG, no `platformdirs` | `platformdirs.user_config_dir("pragmata")` + `PRAGMATA_CONFIG_DIR` override |
| **`annotation setup` / `teardown`** | Exist - manage **Argilla workspaces, users, datasets** (not Docker). See [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py) | **Semantics unchanged** |
| **Interactive prompts** | None - `setup` is headless-flag-driven only | **No change.** No `questionary`, no config wizard (§3.1) |

## References

- [ADR-0007 - Packaging & invocation surface](../decisions/0007-packaging-invocation-surface.md)
- [`annotation-bootstrap.md`](annotation-bootstrap.md) - annotation-only stack lifecycle
- Precedent CLIs: [`gh auth login`](https://cli.github.com/manual/gh_auth_login), [Supabase CLI](https://supabase.com/docs/reference/cli/introduction), [`dbt init`](https://docs.getdbt.com/reference/commands/init), [AWS CLI config](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html), [gcloud properties](https://docs.cloud.google.com/sdk/docs/properties), [platformdirs](https://platformdirs.readthedocs.io/)
