# Annotation Stack Lifecycle

Status: Draft
Related:
- ADR-0007 (packaging & invocation surface)
- ADR-0003 (infra: self-hosted only)
- [`config-and-settings.md`](config-and-settings.md) - shared settings/config resolution that this doc builds on

## Purpose

Stack composition, compose distribution, first-run/upgrade/uninstall lifecycle, prod bootstrap, cross-platform runtime, and infra-specific error UX for `pragmata annotation`. Other tools (`querygen`, `eval`) don't ship infra; this doc is annotation-only.

Shared concerns (config resolution, settings precedence, secrets, generic CLI error UX) live in [`config-and-settings.md`](config-and-settings.md).

## Guiding principle (annotation-specific)

**Dev tooling ≠ production tooling.** The `Makefile` is dev-only; the prod install path goes through the CLI (`pragmata annotation up`). They share the same shipped compose file but apply different overrides.

> The `Makefile` targets (`docker-up`, `docker-down`, `test-stack`) bind to `deploy/annotation/docker-compose.dev.yml` and assume a cloned repo + `make` (= non-starter for Windows, PyPI installs, and unattended/prod environments).
>
> The CLI command `pragmata annotation up` is the single end-user entry point and must work identically across all three environments.

SOTA for PyPI-distributed CLIs that wrap Docker stacks (Supabase CLI, Airbyte `abctl`, Dagster, Prefect, MLflow, Argilla itself) converges on a few points:
- install is side-effect free
- first-run bootstrap is an idempotent single command
- upgrade is the #1 pain point (we mostly skip this)
- dev ≠ prod

## 1. Stack composition

The runtime compose file ships as **package data under `src/pragmata/annotation/docker-compose.yml`** and is resolved at runtime via `importlib.resources`. Users never see or edit the YAML - all supported customisation flows through CLI flags / env / config (§2.3). This is the "locked compose" model (option Y in §2.1); see that section for the full rationale and rejected alternatives.

All services bundled by default (zero-config principle, see [`config-and-settings.md`](config-and-settings.md) principle 4). Each backing service (postgres/elasticsearch/redis) is opt-out-able via a Compose profile (§2.3).

**Single shipped compose file, not a dev/prod pair.** Surveyed PyPI-distributed Docker-wrapping tools (Supabase, Airbyte `abctl`, Airflow, Dagster, Prefect, MLflow) all converge on **one shipped compose artefact**; nobody ships parallel `docker-compose.prod.yml` + `docker-compose.dev.yml` (as forces users to answer "which one do I run?" before doing anything). The split here is:

- **Shipped (package data):** `src/pragmata/annotation/docker-compose.yml` - production-first: pinned tags, env-driven credentials (no hardcoded defaults), sensible resource defaults, localhost-only port bindings. This is the file `pragmata annotation up` resolves at runtime.
- **Contributor dev (cloned repo):** `deploy/annotation/docker-compose.dev.override.yml` (proposed - does not exist yet) - layered on top of the shipped file via Makefile targets using `docker compose -f ... -f ...`. Typical contents: well-known default creds (`argilla`/`1234`), stdout logging, looser health-check timing, exposed debug ports.

End users (`pragmata annotation up`) only ever touch the shipped (package-data) file via CLI flags. The dev override is exclusively for contributors working in a cloned repo.

>Migration steps from today's single `deploy/annotation/docker-compose.dev.yml`:
>  1. Extract prod-safe defaults into `src/pragmata/annotation/docker-compose.yml` as package data (the new SSOT for runtime)
>  2. Strip dev-only overrides into `deploy/annotation/docker-compose.dev.override.yml`
>  3. Update Makefile targets to stack both via `docker compose -f ... -f ...`

## 2. Compose file distribution

Two axes to decide: (a) where the compose file the daemon reads actually lives, (b) how many "bundles" the shipped file supports.

### 2.1 Where the compose file lives (daemon-reads-from)

Three options were considered:

| Option | What it is | Default-path UX | Power-user UX | Upgrade drift | SOTA precedent |
|---|---|---|---|---|---|
| **Y. Locked (compose stays inside package) - *recommended*** | Resolve via `importlib.resources` at runtime; user never sees the YAML. Overrides only via CLI flags / `config.yaml` / env vars we expose. | Zero-config | Customisation surface = `--external-postgres`, `--external-elastic`, port flags, etc. (§2.3) - sufficient for v0.1 | None - we own the file | [Supabase CLI](https://github.com/supabase/cli) (went further: constructs the project programmatically in Go, no compose file at all) |
| **X. User-editable (default copy, drift-flagged)** | First `up` copies packaged YAML → user config dir. User may edit. On subsequent `up`, drift is flagged. | Zero-config: first-run user never touches YAML | Standard Docker mental model: edit the YAML | Real - flagged, user resolves manually | [dbt `profiles.yml`](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml), [VS Code `settings.json`](https://code.visualstudio.com/docs/getstarted/settings) |
| **Z. Eject** | Start with Y; `pragmata annotation eject` copies compose out and pragmata then uses the ejected copy, warning the user they own it from there | Zero-config | Explicit escape hatch, clean managed-vs-owned contract | None for non-ejected users; ejected users own drift | [create-react-app `eject`](https://create-react-app.dev/docs/available-scripts#npm-run-eject), [Expo eject-to-bare-workflow](https://docs.expo.dev/archive/customizing/) |

**Recommendation: Y for v0.1.** Users should not have to understand or edit Docker Compose YAML on the supported paths. The customisation surface that matters - external Postgres/Elasticsearch URLs, port bindings, image tags - is exposed through CLI flags / env / config (§2.3). If that surface is sufficient (and for v0.1 it is), keeping the compose file package-owned avoids drift, simplifies upgrades, and prevents feature-creep where every Compose field becomes a CLI flag.

- **X** creates an ownership/drift problem on day one and makes the happy path Docker-centric. Reserve user-owned compose for an explicit advanced escape hatch (i.e. option Z), not the default.
- **Z** is a clean future escape hatch. Document the pattern, but do **not** build the `eject` verb until concrete demand materialises.

>Related rejected patterns: generate compose from a template at install time (two sources of truth, drifts on upgrade); remote URL fetch (breaks offline installs, trust boundary). Supabase [issue #2435](https://github.com/supabase/cli/issues/2435) documents the specific pain of user-editable compose + CLI tight version coupling - option Y avoids the problem entirely.

### 2.2 Distribution mechanism (how pkg'd YAML travels)

Ships as package data inside the installed package at `src/pragmata/annotation/docker-compose.yml`, resolved at runtime via `importlib.resources.files("pragmata.annotation") / "docker-compose.yml"` + `as_file()`. Same mechanism already in use for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html).

Image tags are pinned in the shipped compose and treated as package-owned. `pip install -U pragmata` ships a new compose with new tags - users automatically pick it up on next `up` (no drift, no warning needed under option Y).

### 2.3 Profiles / bundles (the flag surface for external backing services)

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

**Proposed v0.1 CLI flag surface (minimal):**

```
pragmata annotation up                                    # all-bundled profile (zero-config default)
pragmata annotation up --external-postgres <url>          # external-pg profile, wire Argilla to external PG
pragmata annotation up --external-elastic <url>           # external-es profile, wire Argilla to external ES
```

Internally: `--external-postgres` = select `external-pg` profile + inject `ARGILLA_DATABASE_URL`; `--external-elastic` = select `external-es` profile + inject `ARGILLA_ELASTICSEARCH`. Settings resolution applies as normal - flag > env > config > default (see [`config-and-settings.md`](config-and-settings.md) §1.4).

Precedent for profiles: [Airflow's official Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) ships profiles (`flower`, etc.). [Dagster Helm values](https://github.com/dagster-io/dagster/tree/master/helm/dagster) mirrors the same pattern at a different abstraction level.

## 3. Lifecycle

### 3.1 First-run UX

**`pragmata annotation up` is the first-run command. No separate `init`. `annotation setup` stays as Argilla provisioning (see [`config-and-settings.md`](config-and-settings.md) §3), invoked after the stack is up.**

`up` does:
- Pre-flight in order: extra installed? → Docker daemon reachable? → required ports free?
- Resolve the packaged compose via `importlib.resources` (no copy to disk - option Y, §2.1)
- Pulls images on first invocation (slow; log clearly - make this prominent in docs)
- Health-polls Argilla's health endpoint with a timeout
- On success: prints URL, default API key, print creds (once) and next command

Most of the core parts of this UX are already implemented.

Precedent: `supabase start`, `abctl local install`, `prefect server start`. Idempotent single command, safe to re-run.

### 3.2 Upgrade

**`pip install -U pragmata` is the sole upgrade primitive. The compose file is package-owned (option Y, §2.1), so upgrades pick up the new file automatically with no drift to manage.**

- Named Docker volumes with a deterministic prefix (`pragmata_annotation_*`) persist data across container recreation
- New compose ships with new image tags - `pragmata annotation up` after upgrade picks up the new file via `importlib.resources`. No drift detection / warning / `reset-compose` verb needed (we own the file)
- For destructive Argilla schema migrations between majors, document the backup step; pragmata does not protect users from upstream-breaking changes

>NB: Airbyte's `abctl local install` is idempotent and designed to fix Compose upgrade brittleness ([Airbyte discussion #40599](https://github.com/airbytehq/airbyte/discussions/40599)). We don't need that machinery here - our locked compose model sidesteps the brittleness here.

### 3.3 Uninstall

**`pragmata annotation down` stops the stack; `pragmata annotation down --volumes` additionally wipes data. No global `pragmata uninstall`.**

- `pip uninstall pragmata` removes the package
- `~/.config/pragmata/` removal is documented but user-owned. No cleanup verb.

## 4. Prod bootstrap / unattended install

**Decision for v0.1: no shipped script. Document the two-line install (`pip install 'pragmata[annotation]' && pragmata annotation up`) in the README.** 

### 4.1 Scope

pragmata ships three tools. Only `annotation` has any bootstrap beyond `pip install`:

- `querygen`: `pip install 'pragmata[querygen]' && export OPENAI_API_KEY=...` - two-line README
- `eval`: same as `querygen`
- `annotation`: real sequence (install → wait for Docker → pull images → start stack → poll health → print creds stdout)

The question is whether `annotation` needs a separate install artefact. For v0.1, no.

### 4.2 Options surveyed

| Option | What it is | SOTA precedent | Our cost | Verdict |
|---|---|---|---|---|
| **A. No script (docs only) - *chosen for v0.1*** | Docs snippet: `pip install 'pragmata[annotation]' && pragmata annotation up` | Every Tier-1 PyPI-distributed Docker-wrapping tool surveyed ([Supabase](https://supabase.com/docs/guides/local-development/cli/getting-started), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/get-started/install), [Dagster](https://docs.dagster.io/getting-started/install), [MLflow](https://mlflow.org/docs/latest/tracking.html)) | None | **Adopt** |
| **B. `scripts/bootstrap-annotation.sh` in repo** | Static shell file checked into git, linked from docs | [Docker convenience script](https://github.com/docker/docker-install), [rustup](https://rustup.rs/), [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) | Maintenance drift - every CLI flag change invalidates it. Docker's own convenience installer is [not recommended for production](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) for exactly this reason | **Skip** |

> **Proposed `annotation up` sequence (to implement):**
> 1. Pre-flight checks: `pragmata[annotation]` extra present → Docker daemon reachable → required ports free. Fail fast with actionable message on any failure.
> 2. Resolve compose file via `importlib.resources` (in-memory path, no copy to disk).
> 3. `docker compose pull` - only on first run or after `pip install -U` (detect by comparing pinned image digests vs locally cached). Log clearly; this is the slow step.
> 4. `docker compose up -d` with resolved profile (§2.3).
> 5. Poll `GET /api/v1/health` until 200 or timeout (default 120 s). Stream a progress indicator.
> 6. On first-run only: generate a random `argilla` admin password, inject via env, print URL + credentials to stdout. Subsequent `up` calls skip this step - stack is already configured.
> 7. Print next-step hint: `pragmata annotation setup --users <path>`.


## 5. Cross-platform runtime

**Rely on the generic `docker compose` CLI on PATH. Remain agnostic to the user's Docker runtime.**

- Pre-flight check: `docker version` succeeds (daemon reachable). If not, fail clear: *"Docker daemon not reachable. Start your Docker runtime and try again."*
- No `--runtime` flag, no auto-detection, no runtime-specific branching. Which Docker implementation the user has is not our concern.

Precedent: Supabase and `abctl` both run over the generic `docker` CLI with no engine-specific logic.

## 6. Error taxonomy (extension of shared UX)

Generic first-use errors (extra not installed) are covered in [`config-and-settings.md`](config-and-settings.md) §4. `annotation up` adds infra-specific failure modes:

```
$ pragmata annotation up                    # no Docker daemon
Error: Docker daemon not reachable.
  Start your Docker runtime and try again.
  See: https://docs.docker.com/get-docker/

$ pragmata annotation import foo.json       # stack not running
Error: Argilla stack is not running.
  Run: pragmata annotation up
```

Failure mode is checked in order: *extra-installed → Docker-running → stack-up.* Each check is a strict prerequisite for the next. We fail at the first missing prerequisite with the corresponding fix.

Beyond these, `annotation up` must also handle:
- port conflict (print occupying process if detectable)
- image-pull failure (network / registry)
- Argilla health-poll timeout (print the container log tail)
- compose-file missing from package (indicates broken install - `pip install --force-reinstall 'pragmata[annotation]'`)

## 7. Codebase baseline

| Area | Implemented today | Proposed in this doc |
|---|---|---|
| **Docker stack lifecycle (`up`/`down`)** | Not implemented - only Makefile targets (`docker-up`, `docker-down`, etc.) that read [`deploy/annotation/docker-compose.dev.yml`](../../deploy/annotation/docker-compose.dev.yml) | Add `pragmata annotation up` / `down` CLI commands |
| **Compose file distribution** | Dev-only file in `deploy/`; **not shipped** in the installed wheel | Ship `src/pragmata/annotation/docker-compose.yml` as package data, locked (option Y, §2.1) - resolved at runtime via `importlib.resources`, never copied to disk. Dev override stays in `deploy/` for contributors only |
| **Package data** | `importlib.resources` already used for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html) | Same mechanism for the compose file |

## References

- [ADR-0007 - Packaging & invocation surface](../decisions/0007-packaging-invocation-surface.md)
- [ADR-0003 - Infra: self-hosted only](../decisions/0003-infra-self-hosted-only.md)
- [`config-and-settings.md`](config-and-settings.md) - shared settings/config resolution
- Precedent for Docker orchestration + compose distribution: [Supabase CLI](https://github.com/supabase/cli), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/), [Dagster](https://docs.dagster.io/), [Airflow Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
