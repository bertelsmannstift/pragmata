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

**Dev tooling ≠ production tooling.** The `Makefile` is dev-only; the prod intall/deploy path goes through the CLI (`pragmata annotation up`). Both routes run the same shipped compose file underneath but layer different overrides on top:

> The `Makefile` targets (`docker-up`, `docker-down`, `test-stack`) bind to `deploy/annotation/docker-compose.dev.yml` and assume a cloned repo + `make` (= non-starter for Windows, PyPI installs, and unattended/prod environments).
>
> The CLI command `pragmata annotation up` is the single end-user entry point and must work identically across all three environments.

SOTA across CLIs that wrap a container stack - whether the CLI is PyPI-distributed (Dagster, Prefect, Argilla, Airflow) or a standalone binary (Supabase CLI in Go; Airbyte `abctl` in Go, which wraps Kubernetes via kind rather than Compose) - converges on a few points:

- **Install is side-effect free.** `pip install 'pragmata[annotation]'` installs the package and bundles the compose file; it doesn't start containers, pull images, write user config, or modify Docker daemon state. Install and "do anything" are separate steps. (Installers that pull images or start daemons at install time break CI, container builds, and exploratory venv workflows.)
- **First-run bootstrap is an idempotent single command.** One verb (`pragmata annotation up`) gets you a running stack from a clean machine. Re-running it on an already-running stack is a no-op + health check, not a "did I run init yet?" decision the user has to track.
- **Upgrade is the #1 pain point - and we mostly sidestep it** (see [§3.2](#32-upgrade)). Surveyed tools fail upgrade three ways: user-edited compose drifts from shipped; image tags fall out of sync with CLI version; persistent volumes carry old schema into new containers. Our locked-compose model (option Y) eliminates the first two - the file is package-owned, so `pip install -U` upgrades CLI and compose atomically with no drift. Argilla's own schema migrations between majors remain an upstream concern we don't shield users from.
- **Dev ≠ prod, by artefact.** The contributor workflow (cloned repo, `make`, well-known dev credentials in `.env.dev.example`, exposed debug ports, looser health-check timing) and the end-user workflow (PyPI install, `pragmata annotation up`, env-injected credentials, localhost-only ports, production-tight defaults) run *different file pairs*, not the same file with "remember to change these in prod" comments. The shipped compose is the prod baseline; the dev override layers contributor-only conveniences on top.

**Test boundary.** Integration tests (`make test-stack`, `tests/integration/`) run against `(shipped + dev-override)` - matching what contributors hit locally, with dev credentials. A separate packaging smoke test exercises the shipped file *alone* from an installed wheel (no override layered on top), asserting only that `importlib.resources` resolves the file and that the YAML parses. This is the only test that catches wheel / sdist divergence (e.g. the compose file silently dropped from `[tool.hatch.build.targets.wheel]` after a refactor). Two test paths, deliberately: the dev-override path covers behavioural correctness against real credentials; the shipped-only smoke test covers the packaging contract. SOTA doesn't converge here (Airflow and Argilla test against their dev compose because there's no separate shipped artefact; Supabase / Airbyte test against the locked artefact because that's the only one) - pragmata's two-artefact model means we need both, but the packaging test is intentionally minimal (~20 lines) to keep the cost down.

## 1. Stack composition

The runtime compose file ships as **package data under `src/pragmata/annotation/docker-compose.yml`** and is resolved at runtime via `importlib.resources`. Users never see or edit the YAML - all supported customisation flows through CLI flags / env / config (§2.3). This is the "locked compose" model (option Y in §2.1); see that section for the full rationale and rejected alternatives.

All services bundled by default (zero-config principle, see [`config-and-settings.md`](config-and-settings.md) principle 4). Each backing service (postgres/elasticsearch/redis) is opt-out-able via a Compose profile (§2.3).

**Deterministic project name.** The shipped compose declares a top-level `name: pragmata_annotation` (Compose v2.4+). This locks the project name regardless of the cwd `docker compose` is invoked from, so volumes, networks, and containers carry stable prefixes: volumes `pragmata_annotation_argilladata`, `_postgresdata`, `_elasticdata`, `_redisdata`; default network `pragmata_annotation_default`; containers `pragmata_annotation-argilla-1` etc. Without this, project name defaults to the cwd directory name and the prefixes shift per caller - breaking volume re-attach across upgrades and the named-container detection used by port-conflict diagnostics (§6). Equivalent fallback for older Compose: invoke as `docker compose -p pragmata_annotation`; the in-file `name:` keyword is preferred so the contract travels with the artefact.

**One shipped runtime artefact + a contributor override, not two parallel runtime files.** Surveyed container-stack-wrapping CLIs (Airflow, Dagster, Prefect on PyPI; Supabase as a Go binary; `abctl` on kind, not Compose) all ship a single runtime file users invoke directly; parallel `docker-compose.prod.yml` + `docker-compose.dev.yml` would force "which one do I run?" on every invocation. We adopt the same shape:

- **Shipped (package data):** `src/pragmata/annotation/docker-compose.yml` - production-first: pinned tags, env-driven credentials (no hardcoded defaults), sensible resource defaults, localhost-only port bindings (multi-annotator deployments add a reverse proxy on top - see §4.3). This is the file `pragmata annotation up` resolves at runtime.
- **Contributor dev (cloned repo):** `deploy/annotation/docker-compose.dev.override.yml` (proposed - does not exist yet) - layered on top of the shipped file via Makefile targets using `docker compose -f ... -f ...`. Typical contents: well-known default creds (`argilla` / `argilla123`, per [`.env.dev.example`](../../deploy/annotation/.env.dev.example)), stdout logging, looser health-check timing, exposed debug ports.

End users (`pragmata annotation up`) only ever touch the shipped (package-data) file via CLI flags. The dev override is exclusively for contributors working in a cloned repo.

**Prod credential injection.** The shipped compose is a thin wrapper that renames Argilla's three bare credential env vars to `ARGILLA_`-prefixed equivalents: `ARGILLA_USERNAME → USERNAME`, `ARGILLA_PASSWORD → PASSWORD`, `ARGILLA_API_KEY → API_KEY`. Argilla's `start_argilla_server.sh` reads the bare names; we prefix them on the host side because bare `USERNAME` and `PASSWORD` collide with names already in use by other tooling - notably the OS-provided `%USERNAME%` on Windows, the `USERNAME` shell var some `.env` loaders read implicitly, and `USERNAME` / `PASSWORD` keys in GitHub Actions matrix env blocks. Prefixing eliminates the collision class on the host without touching upstream. Backing-service vars (`ARGILLA_DATABASE_URL`, `ARGILLA_ELASTICSEARCH`, `ARGILLA_REDIS_URL`, `POSTGRES_PASSWORD`) are read through directly with no rename. Operators set these in their shell, CI environment, or a `.env` file they own before running `pragmata annotation up`. If `ARGILLA_USERNAME` and `ARGILLA_PASSWORD` are absent (or set to empty strings - see below), the upstream script's `if [ -n "$USERNAME" ] && [ -n "$PASSWORD" ]` guard skips default-user creation and the server starts normally. Dev defaults are in [`deploy/annotation/.env.dev.example`](../../deploy/annotation/.env.dev.example); contributors copy this to `.env` and the Makefile picks it up. End users never touch that file.

> **Empty-string vs unset.** Docker Compose's `KEY: ${VAR}` substitution emits `KEY=` (empty string) into the container when `$VAR` is unset on the host, not an absent var. Upstream's `[ -n "$USERNAME" ]` test treats empty and unset the same, so behaviour is correct either way - but the safety relies on upstream's specific check, not on Compose passing through "absent". If upstream later switches to `[ -v USERNAME ]` (defined / not defined) this guarantee breaks, and the shipped compose would need to move to list-form `environment: [USERNAME, PASSWORD, API_KEY]` (Compose drops list entries whose source env var is unset on the host) to keep "absent stays absent".

This rename is the only env-var rewriting the shipped compose does. It's intentionally minimal - anything else is pass-through.

**Resource caps and restart policy (shipped baseline).** The current dev compose sets `restart: unless-stopped` on every service and constrains only Elasticsearch heap (`ES_JAVA_OPTS: -Xms512m -Xmx512m`). The shipped compose tightens both:

- **Per-service memory caps.** Use Compose's `deploy.resources.limits.memory` (honoured by `docker compose` in non-swarm mode since v1.27): Argilla `1g`, worker `512m`, Postgres `512m`, Elasticsearch `1.5g` (with `ES_JAVA_OPTS=-Xms768m -Xmx768m` to stay ~50% under the cgroup limit), Redis `256m`. Total ceiling ~3.75 GB, comfortably under the 4 GB Docker Desktop default on macOS and the smallest realistic Linux VM. No CPU pinning - container scheduling on a single-host CLI deployment isn't worth the operational surface.
- **`restart: on-failure` instead of `unless-stopped` on the shipped baseline.** `unless-stopped` makes the stack auto-start on every host reboot until the operator runs an explicit `pragmata annotation down` - acceptable for the contributor dev workflow but a footgun for laptop users who installed via `pipx` and don't expect a 3.75 GB stack to come up on every login. `on-failure` keeps containers self-healing within a session but does not survive reboot. Operators who want the always-on behaviour (dedicated annotation VM) set `PRAGMATA_RESTART_POLICY=unless-stopped` before `up`, or layer their own override.
- **Override path for both.** The contributor dev override (`docker-compose.dev.override.yml`) restores `restart: unless-stopped` and lifts the memory caps - contributors hitting a constrained Argilla while writing tests is the exact failure mode the override exists to prevent.

>**Migration from today's single `deploy/annotation/docker-compose.dev.yml`:**
>
>| File | Action | Role after migration |
>|---|---|---|
>| `deploy/annotation/docker-compose.dev.yml` | **Delete** (replaced by the two files below) | n/a |
>| `src/pragmata/annotation/docker-compose.yml` | **Create** as package data | Prod-first shipped file; `pragmata annotation up` reads this at runtime via `importlib.resources` |
>| `deploy/annotation/docker-compose.dev.override.yml` | **Create** | Dev-only overrides; layered on top of the shipped file by Makefile targets via `docker compose -f ... -f ...` |
>| `Makefile` | **Update** | Change `COMPOSE_FILE := deploy/annotation/docker-compose.dev.yml` to stack both via `-f src/pragmata/annotation/docker-compose.yml -f deploy/annotation/docker-compose.dev.override.yml`. Affects targets `docker-up`, `docker-up-external-*`, `docker-down`, `docker-stop`, `docker-logs`, `docker-status`, `test-stack`. |
>| `tests/conftest.py`, `tests/integration/test_dev_stack_integration.py` | **Audit** | Both currently hit `http://localhost:6900/api/v1/me`. Decide whether integration tests run against `(shipped + dev-override)` (matches what contributors hit) or shipped-only (matches what end users hit). Recommendation: stay on `(shipped + dev-override)` for the existing tests, add a separate packaging smoke test that exercises the shipped file alone from an installed wheel (§2.2). |
>
>Steps:
>  1. Extract prod-safe defaults from `docker-compose.dev.yml` into `src/pragmata/annotation/docker-compose.yml` (the new SSOT for runtime)
>  2. Strip dev-only overrides into `deploy/annotation/docker-compose.dev.override.yml`
>  3. Update Makefile targets to stack both via `docker compose -f ... -f ...`
>  4. Update the test path per the audit row above; make sure `make test-stack` and the existing integration tests still pass against the new layout *before* deleting the old file
>  5. Delete `deploy/annotation/docker-compose.dev.yml`

## 2. Compose file distribution

Two axes to decide: (a) where the compose file the daemon reads actually lives, (b) how many "bundles" the shipped file supports.

### 2.1 Where the compose file lives (daemon-reads-from)

Three options were considered:

| Option | What it is | Default-path UX | Power-user UX | Upgrade drift | SOTA precedent |
|---|---|---|---|---|---|
| **Y. Locked (compose stays inside package, recommended)** | Resolve via `importlib.resources` at runtime; user never sees the YAML. Overrides only via CLI flags / `config.yaml` / env vars we expose. | Zero-config | Customisation surface = `--external-postgres`, `--external-elastic`, port flags, etc. (§2.3) - sufficient for v0.1 | None - we own the file | [Supabase CLI](https://github.com/supabase/cli) (went further: constructs the project programmatically in Go, no compose file at all) |
| **X. User-editable (default copy, drift-flagged)** | First `up` copies packaged YAML → user config dir. User may edit. On subsequent `up`, drift is flagged. | Zero-config: first-run user never touches YAML | Standard Docker mental model: edit the YAML | Real - flagged, user resolves manually | [dbt `profiles.yml`](https://docs.getdbt.com/docs/core/connect-data-platform/profiles.yml), [VS Code `settings.json`](https://code.visualstudio.com/docs/getstarted/settings) |
| **Z. Eject** | Start with Y; `pragmata annotation eject` copies compose out and pragmata then uses the ejected copy, warning the user they own it from there | Zero-config | Explicit escape hatch, clean managed-vs-owned contract | None for non-ejected users; ejected users own drift | [create-react-app `eject`](https://create-react-app.dev/docs/available-scripts#npm-run-eject), [Expo eject-to-bare-workflow](https://docs.expo.dev/archive/customizing/) |

**Recommendation: Y for v0.1.** Users should not have to understand or edit Docker Compose YAML on the supported paths. The customisation surface that matters - external Postgres/Elasticsearch URLs, port bindings, image tags - is exposed through CLI flags / env / config (§2.3). If that surface is sufficient (and for v0.1 it is), keeping the compose file package-owned avoids drift, simplifies upgrades, and prevents feature-creep where every Compose field becomes a CLI flag.

- **X** creates an ownership/drift problem on day one; it makes the happy path Docker-centric. Reserve user-owned compose for an explicit advanced escape hatch (i.e. option Z), not the default.
- **Z** is a clean future escape hatch; document the pattern, but do **not** build the `eject` verb until concrete demand materialises.

>Related rejected patterns: generate compose from a template at install time (two sources of truth, drifts on upgrade); remote URL fetch (breaks offline installs, trust boundary). Supabase [issue #2435](https://github.com/supabase/cli/issues/2435) documents the specific pain of user-editable compose + CLI tight version coupling: option Y avoids the problem entirely.

### 2.2 Distribution mechanism (how pkg'd YAML travels)

Ships as package data inside the installed package at `src/pragmata/annotation/docker-compose.yml`, resolved at runtime via `importlib.resources.files("pragmata.annotation") / "docker-compose.yml"` + `as_file()`. Same mechanism already in use for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html).

Image tags are pinned in the shipped compose and treated as package-owned. `pip install -U pragmata` ships a new compose with new tags - users automatically pick it up on next `up` (no drift, no warning needed under option Y).

**Packaging contract (proposed - not yet configured):** `docker-compose.yml` must be confirmed as package data in both wheel and sdist. Today `pyproject.toml` declares `build-backend = "hatchling.build"` but has no explicit `[tool.hatch.build.targets.wheel]` block; hatchling's `src`-layout default picks up `src/pragmata/` automatically, and *does* by default include non-`.py` files under package directories - but that default is the failure mode we want to guard against (it depends on file-pattern defaults that have shifted between hatchling versions and that can be silently overridden by a future `include`/`exclude` line). The explicit contract pins both *which package roots are shipped* and *which non-Python artefacts must accompany them*:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/pragmata"]

[tool.hatch.build.targets.wheel.force-include]
"src/pragmata/annotation/docker-compose.yml" = "pragmata/annotation/docker-compose.yml"
"src/pragmata/core/annotation/collapsible_field.html" = "pragmata/core/annotation/collapsible_field.html"

[tool.hatch.build.targets.sdist]
include = [
  "src/pragmata/**/*.py",
  "src/pragmata/**/*.yml",
  "src/pragmata/**/*.html",
  "pyproject.toml",
  "README.md",
  "LICENSE.md",
]
```

`force-include` is the explicit guard: it lists the non-Python artefacts the wheel *must* contain by path, so a future `exclude` rule or layout refactor cannot silently drop them without a corresponding edit to this block. Verify by inspecting the built wheel before shipping:

```
unzip -l dist/*.whl | grep docker-compose
```

A packaging smoke test exercises the shipped file *alone* from an installed wheel (not in-tree, no dev override layered on top). Assertion scope is deliberately minimal - file resolves + YAML parses, nothing more:

```python
import yaml
from importlib.resources import files

def test_shipped_compose_is_packaged():
    path = files("pragmata.annotation").joinpath("docker-compose.yml")
    assert path.is_file()
    yaml.safe_load(path.read_text())  # raises if malformed
```

Behavioural correctness (does Argilla actually come up against the shipped compose) is *not* this test's job - that belongs to the dev-override integration tests, which run against the same shipped file with credentials layered on. The packaging test exists for one purpose: catch wheel / sdist divergence that no in-tree test can see. This is the only check that detects "the compose file silently dropped from the wheel after a `force-include` / `exclude` refactor" before users hit it.

### 2.3 Profiles / bundles (the flag surface for external backing services)

Explicitly supports **Docker Compose profiles + external backing service URLs**. We use Compose's built-in `profiles` feature. Profile names carry forward from the current dev compose with one addition for v0.1: `all-bundled`, `external-pg`, `external-es`, `external-redis` (new - operators with managed Redis: Elasticache, Upstash, etc.).

**Two-tier model.** `argilla` and `worker` are baseline services - they have no `profiles:` key and so run unconditionally on every `up`. Backing services (`postgres`, `elasticsearch`, `redis`) are opt-out via profiles: the profile name names what's *external*, so the matching service is *omitted* from that profile's list. Reading the YAML below: if a service has no `profiles:` key it's always-on; if it has one, it's a bundled backing service whose absence from a profile means "the operator is supplying that one externally".

```yaml
services:
  # Always-on services: no `profiles:` key, so they start under every profile selection
  # (Compose default: services without `profiles:` always run). This matches the current
  # dev compose's behaviour and means `argilla` / `worker` are unconditionally part of the stack.
  argilla:
    image: argilla/argilla-server:<pinned>
    depends_on: [postgres, elasticsearch, redis]

  worker:
    image: argilla/argilla-server:<pinned>
    command: argilla worker

  # Backing services: opt-out per profile. The profile name names what's *external*,
  # so the matching service is *omitted* from that profile's list.
  postgres:
    profiles: [all-bundled, external-es, external-redis]   # skipped when profile is external-pg
    image: postgres:<pinned>
    # ...

  elasticsearch:
    profiles: [all-bundled, external-pg, external-redis]   # skipped when profile is external-es
    image: docker.elastic.co/elasticsearch/elasticsearch:<pinned>
    # ...

  redis:
    profiles: [all-bundled, external-pg, external-es]      # skipped when profile is external-redis
    image: redis:<pinned>
    # ...
```

**Proposed v0.1 CLI flag surface (minimal):**

```
pragmata annotation up                                    # all-bundled profile (zero-config default)
pragmata annotation up --external-postgres <url>          # external-pg profile, wire Argilla to external PG
pragmata annotation up --external-elastic <url>           # external-es profile, wire Argilla to external ES
pragmata annotation up --external-redis <url>             # external-redis profile, wire Argilla to external Redis
```

Flags compose: passing `--external-postgres … --external-redis …` selects the union of those two externals and bundles only Elasticsearch. (Implementation note: we we drive the inclusion/exclusion by setting Compose `COMPOSE_PROFILES` to multiple values - this is simple and matches Compose's own model.)

Internally: `--external-postgres` = select `external-pg` profile + inject `ARGILLA_DATABASE_URL`; `--external-elastic` = select `external-es` profile + inject `ARGILLA_ELASTICSEARCH`; `--external-redis` = select `external-redis` profile + inject `ARGILLA_REDIS_URL`. Settings resolution applies as normal - flag > env > config > default (see [`config-and-settings.md`](config-and-settings.md) §1.4).

Precedent for profiles: [Airflow's official Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) ships profiles (`flower`, etc.). [Dagster Helm values](https://github.com/dagster-io/dagster/tree/master/helm/dagster) mirrors the same pattern at a different abstraction level.

## 3. Lifecycle

### 3.1 First-run UX

**`pragmata annotation up` is the first-run command. No separate `init`. `annotation setup` stays as Argilla provisioning (see [`config-and-settings.md`](config-and-settings.md) §3), invoked after the stack is up.**

`up` does:
- Pre-flight in order: extra installed? → Docker daemon reachable? → required ports free?
- Resolve the packaged compose via `importlib.resources` (no copy to disk - option Y, §2.1)
- Pulls images on first invocation (slow; stream per-image progress to terminal; rerun-safe - see §4.2 step 3). The README must call this out as a one-off ~5 GB download (Argilla + Postgres + Elasticsearch + Redis) so users on metered/slow networks know what to expect on the first `up`.
- Health-polls Argilla's `/api/v1/status` endpoint with a timeout. The endpoint is unauthenticated and returns `{version, search_engine, memory}`; it awaits `search_engine.info()` inline, so a degraded or unreachable search backend surfaces here as a non-200 rather than requiring a separate check.
- On success: prints Argilla URL and next-step hint

Most of the core parts of this UX are already implemented.

Precedent for "idempotent single command, safe to re-run": `abctl local install`, `prefect server start`. (Supabase requires `supabase init` to scaffold a project directory before `supabase start` - we don't replicate that because pragmata has no per-project state to scaffold for `annotation up`; project config lives in `pyproject.toml` / `pragmata.yaml` via the shared settings resolver, not in a per-tool init step.)

### 3.2 Upgrade

**`pip install -U pragmata` is the sole upgrade primitive. The compose file is package-owned (option Y, §2.1), so upgrades pick up the new file automatically with no drift to manage.**

- Named Docker volumes with a deterministic prefix (`pragmata_annotation_*`) persist data across container recreation
- New compose ships with new image tags - `pragmata annotation up` after upgrade picks up the new file via `importlib.resources`. No drift detection / warning / `reset-compose` verb needed (we own the file)
- For destructive Argilla schema migrations between majors, document the backup step; pragmata does not protect users from upstream-breaking changes

>NB: Airbyte's `abctl local install` is idempotent and designed to fix Compose upgrade brittleness ([Airbyte discussion #40599](https://github.com/airbytehq/airbyte/discussions/40599)). We don't need that machinery - our locked compose model sidesteps the brittleness.

### 3.3 Uninstall

**`pragmata annotation down` stops the stack; `pragmata annotation down --volumes` additionally wipes data. No global `pragmata uninstall`.**

- `pip uninstall pragmata` removes the package
- `~/.config/pragmata/` removal is documented but user-owned. No cleanup verb.

### 3.4 Data persistence boundaries

Named Docker volumes (`pragmata_annotation_argilladata`, `_postgresdata`, `_elasticdata`, `_redisdata`) live under the Docker data root - `/var/lib/docker/volumes/` on Linux, the Docker Desktop VM's disk on macOS/Windows. Persistence guarantees follow the host's disk semantics, not pragmata's:

| Event | Data persists? | Notes |
|---|---|---|
| Container recreation (`down` → `up`) | Yes | Volume re-attaches by name. |
| Host reboot | Yes | Docker daemon restarts; volumes remount. |
| `pragmata annotation down --volumes` | **No** | Explicit data wipe; matches `docker compose down -v`. |
| Azure VM `stop-deallocate` + `start` | Yes, by default | OS disk (where `/var/lib/docker` lives) survives deallocation. **Caveat:** if the operator has moved Docker's data root to the ephemeral temp disk (`/mnt/resource` on Azure) for performance, deallocation wipes it. The shipped compose uses named volumes, not bind mounts, so we inherit whatever the host's Docker daemon is configured to do. |
| VM deletion / OS disk replacement | No | The OS disk goes with the VM. Data is lost unless backed up out-of-band. |
| `docker volume prune` | No (if pruned) | Manual user action. |

**What pragmata does not provide in v0.1:**
- Automated backup of named volumes. Operators wanting durability across VM deletion should snapshot the data root (or the Postgres / Elasticsearch contents directly) on whatever cadence their infra demands. Document this in operator-facing notes; don't build a backup verb in v0.1.
- Bind-mount overrides for the volume mountpoints. Some operators will want `/var/lib/docker/volumes/...` redirected to a managed disk or NFS mount. Available today via Docker daemon config (`data-root` in `/etc/docker/daemon.json`) without any pragmata code; mention in passing in operator-facing docs.

**Failure-mode warning for the README:** explicitly call out that ephemeral-disk Docker configurations *will* lose data on deallocation. Don't assume operators read Docker daemon docs.

## 4. Prod bootstrap / unattended install

**Decision for v0.1: no shipped script. Document the two-line install (`pip install 'pragmata[annotation]' && pragmata annotation up`) in the README.**

The two-line install works identically across pip, `pipx install 'pragmata[annotation]'`, and `uv tool install 'pragmata[annotation]'` - in all three cases the `pragmata` entry point ends up on PATH and `annotation up` resolves the packaged compose the same way. No installer-specific branching in the docs is needed; just show `pip` as the default and mention pipx/uv work the same.

**Scope of "prod" for the two-line install: single-host, single-operator (or co-located annotators on the same machine).** Argilla binds to `127.0.0.1:6900` by the shipped compose's defaults, so the stack is only reachable from the host running `annotation up`. For multi-annotator deployments where annotators sit on different machines, the two-line install is *not* sufficient on its own and must be combined with a reverse proxy - see §4.3.

### 4.1 Scope

pragmata ships three tools. Only `annotation` has any bootstrap beyond `pip install`:

- `querygen`: `pip install 'pragmata[querygen]' && export OPENAI_API_KEY=...` - two-line README
- `eval`: same as `querygen`
- `annotation`: real sequence (install → wait for Docker → pull images → start stack → poll health → print URL + next-step hint)

The question is whether `annotation` needs a separate install artefact. For v0.1, no.

### 4.2 Options surveyed

| Option | What it is | SOTA precedent | Our cost | Verdict |
|---|---|---|---|---|
| **A. No script (docs only) - *chosen for v0.1*** | Docs snippet: `pip install 'pragmata[annotation]' && pragmata annotation up` | Every Tier-1 Docker-stack-wrapping tool surveyed ([Supabase](https://supabase.com/docs/guides/local-development/cli/getting-started), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/get-started/install), [Dagster](https://docs.dagster.io/getting-started/install), [Airflow Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)) | None | **Adopt** |
| **B. `scripts/bootstrap-annotation.sh` in repo** | Static shell file checked into git, linked from docs | [Docker convenience script](https://github.com/docker/docker-install), [rustup](https://rustup.rs/), [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) | Maintenance drift - every CLI flag change invalidates it. Docker's own convenience installer is [not recommended for production](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) for exactly this reason | **Skip** |

> **Proposed `annotation up` sequence (to implement):**
> 1. Pre-flight checks: `pragmata[annotation]` extra present → Docker daemon reachable → required ports free. Fail fast with actionable message on any failure.
> 2. Resolve compose file via `importlib.resources` (in-memory path, no copy to disk).
> 3. `docker compose up --pull missing` (Compose's default, made explicit). First invocation pulls all images; subsequent invocations skip the registry round-trip entirely because the images are already present locally. `pip install -U pragmata` ships a compose file with new pinned tags - those new tags are missing locally and so get pulled on the next `up`, which gives us the "upgrade pulls, steady-state doesn't" behaviour for free without pragmata-side digest tracking. Stream per-image progress directly to the user's terminal during a first-run pull (don't buffer or summarise - the user needs to see that something is happening during a multi-gigabyte download). On network failure mid-pull: rerun the whole `up` step. Partial layers are kept by Docker and resumed, so reruns are cheap; we don't track partial state ourselves.
>
>    **Why not `--pull always`** (which is what the Makefile uses)? `--pull always` re-contacts the registry on every `up` even for pinned tags that haven't changed. The Makefile's contributor-facing use case wants that (developers sometimes hand-edit the compose or bump tags locally and want a guaranteed re-pull on `make docker-up`); the end-user CLI doesn't (registry round-trips on every invocation are pure latency cost since we own the tag bump cadence via `pip install -U`). Asymmetry is intentional: contributors and end users have different needs here. **Why not pragmata-side digest tracking?** Reinvents `--pull missing`, and introduces a sync-failure mode where our sentinel can disagree with the actual Docker daemon state (after `docker image prune`, `DOCKER_HOST` switch, etc.). Compose already tracks "do I have this image locally"; we let it.
> 4. `docker compose up -d` with resolved profile (§2.3).
> 5. Poll `GET /api/v1/status` until 200 or timeout (default 120 s). Stream a progress indicator. (`/api/v1/status` is unauthenticated and returns `{version, search_engine, memory}`; because it awaits `search_engine.info()` inline, a 200 implies the search backend is up too. Preferable to the auth-gated `/api/v1/me` the current dev compose uses, which depends on a matching API key on the client side.)
> 6. Print Argilla URL (`http://localhost:6900` by default) and next-step hint. Credentials are not printed. For dev defaults see [`deploy/annotation/.env.dev.example`](../../deploy/annotation/.env.dev.example); for production, set `ARGILLA_USERNAME`, `ARGILLA_PASSWORD`, `ARGILLA_API_KEY` in the environment before running `up` (the shipped compose reads them directly, no `PRAGMATA_*` wrapper).
> 7. Print next-step hint: `pragmata annotation setup --users <path>`.

### 4.3 Production deployment beyond single-host

The shipped compose binds Argilla to `127.0.0.1:6900` (loopback only) as a safety default - a service bound to `0.0.0.0` on a cloud VM is immediately reachable from the public internet, and Argilla's built-in auth (username / password / API key over plain HTTP) is not designed for that exposure. Two deployment shapes follow from that default:

**Single-host (in scope for the two-line install).** Operator runs `pragmata annotation up` on a machine; annotators sit on the same machine and hit `http://localhost:6900` in their browsers. Works out of the box. Common for solo research workflows, demos, single-annotator pilot studies.

**Multi-annotator across machines (requires operator setup beyond the two-line install).** Annotators on their own laptops need to reach Argilla on a server. The supported pattern is a reverse proxy on the host that terminates TLS and forwards to `127.0.0.1:6900`:

- Reverse proxy of choice: [Caddy](https://caddyserver.com/docs/quick-starts/reverse-proxy) is the lowest-friction option (automatic Let's Encrypt, ~3-line config); nginx and Traefik are equally valid. **Alternatively, in environments where inbound public ports aren't available** (university networks, locked-down VPCs, no public IP), [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) achieves the same end via an outbound-only connection from the host - no inbound firewall rules, no DNS, no cert lifecycle on the operator side. Operator owns the proxy / tunnel lifecycle either way; pragmata does not manage it.
- Argilla's own credentials (`ARGILLA_USERNAME` / `ARGILLA_PASSWORD` / `ARGILLA_API_KEY`) must be set to non-default values before `annotation up` - the dev defaults (`argilla` / `argilla123`) are well-known and unsafe on a network-reachable instance.
- The host's firewall / network security group must allow inbound `:443` (proxy) and block direct `:6900` access from outside the host.

We deliberately do *not* in v0.1:
- Ship a CLI flag for the bind address (`--bind 0.0.0.0`). A flag invites operators to expose plain-HTTP Argilla on a public IP, which is the failure mode the loopback default exists to prevent. Trigger to revisit: 3+ unique requests for direct binding without a reverse proxy. The escape hatch today is editing a contributor override or running raw `docker compose` against the shipped file with a `ports:` override.
- Ship a Caddy / Traefik sidecar profile. Real implementation work (TLS lifecycle, hostname config, optional basic-auth layer) for a path the operator can stand up in 5 lines themselves. Trigger to revisit: an operator who wants pragmata to manage their TLS / cert renewal.

The README's prod-install section must distinguish these two cases explicitly, not present the two-line install as a complete prod story.


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
- **port conflict** - print the occupying process if detectable (`lsof -i :6900` on Unix, `Get-NetTCPConnection` on Windows). The single most likely real-world failure: a prior `annotation up` left a container bound to 6900 that the user forgot about. Detect this case specifically (named container with `pragmata_annotation_*` prefix on the port) and suggest `pragmata annotation down` rather than a generic "port in use".
- **`docker compose` plugin missing** - distinct from "Docker daemon not reachable". On macOS/Windows with Docker Desktop, both are usually present; on Linux with the Engine package, the Compose plugin is a separate install. Detect by running `docker compose version`: exit-code 0 means present; non-zero with "docker: 'compose' is not a docker command" means the plugin is missing. Fix: link to <https://docs.docker.com/compose/install/linux/>.
- **insufficient disk space for image pull** - check available space against a known floor (~6 GB headroom for the full stack) before invoking `docker compose up --pull missing`. On failure: report the shortfall, point at the Docker root dir (`docker info --format '{{.DockerRootDir}}'`).
- **image-pull failure** (network / registry) - distinguish DNS / TLS / 401-from-registry / generic network errors where possible from `docker compose up` stderr; surface the verbatim Docker error rather than re-wrapping.
- **Argilla health-poll timeout** - print the container log tail (`docker compose logs --tail=50 argilla`).
- **compose-file missing from package** (indicates broken install) - `pip install --force-reinstall 'pragmata[annotation]'`.

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
