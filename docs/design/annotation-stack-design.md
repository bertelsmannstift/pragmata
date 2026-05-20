# Annotation Stack Design

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
- **Upgrade is the #1 pain point.** Surveyed tools fail it three ways: user-edited compose drifts from shipped; image tags drift from CLI version; persistent volumes carry old schema into new containers. Locked compose (option Y) eliminates the first two; the third is upstream's concern. See [§3.2](#32-upgrade).
- **Dev ≠ prod, by artefact.** The contributor workflow (cloned repo, `make`, well-known dev credentials in `.env.dev.example`, exposed debug ports, looser health-check timing) and the end-user workflow (PyPI install, `pragmata annotation up`, env-injected credentials, localhost-only ports, production-tight defaults) run *different file pairs*, not the same file with "remember to change these in prod" comments. The shipped compose is the prod baseline; the dev override layers contributor-only conveniences on top.

**Test boundary.** Two test paths, deliberately: integration tests (`make test-stack`, `tests/integration/`) run against `(shipped + dev-override)` to cover behavioural correctness; a separate packaging smoke test runs the shipped file alone from an installed wheel to cover the packaging contract. SOTA doesn't converge here (Airflow / Argilla test only their dev compose because there's no separate shipped artefact; Supabase / Airbyte test only the locked artefact because that's the only one) - pragmata's two-artefact model needs both. Mechanics and assertion scope in §2.2.

## 1. Stack composition

The runtime compose file ships as **package data under `src/pragmata/annotation/docker-compose.yml`** and is resolved at runtime via `importlib.resources`. Users never see or edit the YAML - all supported customisation flows through CLI flags / env / config (§2.3). This is the "locked compose" model (option Y in §2.1); see that section for the full rationale and rejected alternatives.

All services bundled by default (zero-config principle, see [`config-and-settings.md`](config-and-settings.md) principle 4). Each backing service (postgres/elasticsearch/redis) is opt-out-able via a Compose profile (§2.3).

**Deterministic project name.** The shipped compose declares a top-level `name: pragmata_annotation` (Compose v2.4+). This locks the project name regardless of the cwd `docker compose` is invoked from, so volumes, networks, and containers carry stable prefixes: volumes `pragmata_annotation_argilladata`, `_postgresdata`, `_elasticdata`, `_redisdata`; default network `pragmata_annotation_default`; containers `pragmata_annotation-argilla-1` etc. Without this, project name defaults to the cwd directory name and the prefixes shift per caller - breaking volume re-attach across upgrades and the named-container detection used by port-conflict diagnostics (§6). Equivalent fallback for older Compose: invoke as `docker compose -p pragmata_annotation`; the in-file `name:` keyword is preferred so the contract travels with the artefact.

**One shipped runtime artefact + a contributor override, not two parallel runtime files.** Surveyed container-stack-wrapping CLIs (Airflow, Dagster, Prefect on PyPI; Supabase as a Go binary; `abctl` on kind, not Compose) all ship a single runtime file users invoke directly; parallel `docker-compose.prod.yml` + `docker-compose.dev.yml` would force "which one do I run?" on every invocation. We adopt the same shape:

- **Shipped (package data):** `src/pragmata/annotation/docker-compose.yml` - production-first: pinned tags, env-driven credentials (no hardcoded defaults), sensible resource defaults, localhost-only port bindings (multi-annotator deployments add a reverse proxy on top - see §4.3). This is the file `pragmata annotation up` resolves at runtime.
- **Contributor dev (cloned repo):** `deploy/annotation/docker-compose.dev.override.yml` (proposed - does not exist yet) - layered on top of the shipped file via Makefile targets using `docker compose -f ... -f ...`. Typical contents: well-known default creds (`argilla` / `argilla123`, per [`.env.dev.example`](../../deploy/annotation/.env.dev.example)), stdout logging, looser health-check timing, exposed debug ports.

End users (`pragmata annotation up`) only ever touch the shipped (package-data) file via CLI flags. The dev override is exclusively for contributors working in a cloned repo.

**Prod credential injection.** The shipped compose is a thin wrapper that renames Argilla's three bare credential env vars to `ARGILLA_`-prefixed equivalents: `ARGILLA_USERNAME → USERNAME`, `ARGILLA_PASSWORD → PASSWORD`, `ARGILLA_API_KEY → API_KEY`. Argilla's `start_argilla_server.sh` reads the bare names; we prefix them on the host side because bare `USERNAME` and `PASSWORD` collide with names already in use by other tooling - notably the OS-provided `%USERNAME%` on Windows, the `USERNAME` shell var some `.env` loaders read implicitly, and `USERNAME` / `PASSWORD` keys in GitHub Actions matrix env blocks. Prefixing eliminates the collision class on the host without touching upstream. This rename is the only env-var rewriting the shipped compose does. It's intentionally minimal - anything else is pass-through: backing-service vars (`ARGILLA_DATABASE_URL`, `ARGILLA_ELASTICSEARCH`, `ARGILLA_REDIS_URL`, `POSTGRES_PASSWORD`) are read through directly with no rename. Operators set these in their shell, CI environment, or a `.env` file they own before running `pragmata annotation up`. If `ARGILLA_USERNAME` and `ARGILLA_PASSWORD` are absent (or set to empty strings - see below), the upstream script's `if [ -n "$USERNAME" ] && [ -n "$PASSWORD" ]` guard skips default-user creation and the server starts normally. Dev defaults are in [`deploy/annotation/.env.dev.example`](../../deploy/annotation/.env.dev.example); contributors copy this to `.env` and the Makefile picks it up. End users never touch that file.

> **`ARGILLA_API_KEY` is dual-purpose.** The same env var that sets the *server's bootstrap* API key here is the *client's* API key that the Argilla SDK reads when `pragmata annotation setup` / `import` / `export` talk to the running server (see [`config-and-settings.md`](config-and-settings.md) §1.5). The two values must match; in the happy path the operator exports `ARGILLA_API_KEY` once and both planes pick it up. Rotation is not just a value change - the server keeps the bootstrap key it was first started with, so the only supported rotation path in v0.1 is `pragmata annotation down --volumes` followed by `up` with the new value, then re-`setup`. We will document this in the operator notes; and won't ship a key-rotation verb initially.

**Resource caps and restart policy (shipped baseline).** The current dev compose sets `restart: unless-stopped` on every service and constrains only Elasticsearch heap (`ES_JAVA_OPTS: -Xms512m -Xmx512m`). The shipped compose tightens both. Mechanisms and policies, not values - the concrete numbers are an implementation concern to be set against measured `docker stats` for a realistic-size annotation campaign, not chosen here:

- Per-service memory caps via Compose `deploy.resources.limits.memory`
- `restart:` policy via Compose variable substitution. Shipped compose writes `restart: ${PRAGMATA_ANNOTATION_RESTART_POLICY:-on-failure}`; Compose substitutes from the host env at file-load time, no pragmata-side code reads the var. The `PRAGMATA_ANNOTATION_` prefix follows the per-tool env-var convention in [`config-and-settings.md`](config-and-settings.md) §1.4 (`PRAGMATA_<TOOL>_<KEY>`). Default `on-failure` keeps containers self-healing within a session but does not survive reboot - sidesteps the laptop risk where `unless-stopped` brings a multi-GB stack back up on every login for users who installed via `pipx` for one-off use. Operators who want always-on behaviour (dedicated annotation VM) export `PRAGMATA_ANNOTATION_RESTART_POLICY=unless-stopped` before `up`. 
- Override path for both: contributor dev override (`docker-compose.dev.override.yml`) restores `restart: unless-stopped` and removes memory caps - contributors hitting a constrained Argilla while writing tests is the exact failure mode the override exists to prevent.

>Concrete cap values are deliberately not in this doc. They depend on the campaign dataset (query × chunk count for Task 1, answer × context count for Task 2 etc.) and on which Argilla version we pin. Set them when creating the shipped compose by running `make test-stack` against a representative dataset, reading steady-state RSS from `docker stats`.

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

Resolved at runtime via `importlib.resources.files("pragmata.annotation") / "docker-compose.yml"` + `as_file()`. Same mechanism already in use for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html). Image tags are pinned in the shipped compose and treated as package-owned; upgrade flow in §3.2.

**Packaging contract (proposed - not yet configured):** `docker-compose.yml` must be confirmed as package data in both wheel and sdist. Today `pyproject.toml` declares `build-backend = "hatchling.build"` but has no explicit `[tool.hatch.build.targets.wheel]` block. Two things pin down:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/pragmata"]

[tool.hatch.build.targets.wheel.force-include]
"src/pragmata/annotation/docker-compose.yml" = "pragmata/annotation/docker-compose.yml"
```

`packages` pins the wheel root explicitly so the shipped layout doesn't depend on hatchling's src-layout autodetection. `force-include` is reserved for the compose file alone, because it's the locked shipped infrastructure artefact whose presence is part of the user-facing stack contract. Other non-Python resources that live under `src/pragmata` are also picked up by the package-tree default (the sdist also relies on hatchling's default: everything not VCS-ignored, plus `pyproject.toml` / README / LICENSE always), but these are not the explicit concern of ours, not to be tested for. The packaging smoke test below would be the actual regression guard. Once configured, the built wheel can be verified manually with:

```
unzip -l dist/*.whl | grep docker-compose
```

A packaging smoke test (not yet built) would exercise the installed wheel end-to-end (not in-tree, no dev override). Intended scope: build + install the package, resolve `pragmata/annotation/docker-compose.yml` via `importlib.resources`, parse as YAML. Deliberately narrow to compose only, since compose is the part of the contract that can silently drift without breaking in-tree tests; other internal resources (e.g. `collapsible_field.html`) rely on hatchling's package-tree default and are out of scope for the smoke test.

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

`up` runs the sequence in §4.2: pre-flight → resolve packaged compose → `--pull missing` → start → health-poll `/api/v1/status` → print URL + next-step hint. The README must flag the first-run image pull as a one-off ~5 GB download (Argilla + Postgres + Elasticsearch + Redis) so users on metered/slow networks know what to expect.

Most of the core parts of this UX are already implemented.

Precedent for "idempotent single command, safe to re-run": `abctl local install`, `prefect server start`. (Supabase requires `supabase init` to scaffold a project directory before `supabase start` - we don't replicate that because pragmata has no per-project state to scaffold for `annotation up`; project config lives in `pyproject.toml` / `pragmata.yaml` via the shared settings resolver, not in a per-tool init step.)

### 3.2 Upgrade

**`pip install -U pragmata` is the sole upgrade primitive.**

- Named Docker volumes (`pragmata_annotation_*`) persist data across container recreation; next `up` resolves the upgraded compose via `importlib.resources` and picks up the new pinned tags. No drift detection / warning / `reset-compose` verb needed (we own the file).
- For destructive Argilla schema migrations between majors, document the backup step; pragmata does not protect users from upstream-breaking changes.

>NB: Airbyte's `abctl local install` is idempotent and designed to fix Compose upgrade brittleness ([Airbyte discussion #40599](https://github.com/airbytehq/airbyte/discussions/40599)). We don't need that machinery - locked compose sidesteps it.

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

### 3.5 Python API surface for `up` / `down`

The CLI is a thin wrapper over a Python API, matching [`config-and-settings.md`](config-and-settings.md) §2's contract ("Public API signatures accept tool-specific settings as kwargs with `UNSET` defaults … CLI commands are thin wrappers over the API"). The same pattern that backs [`api/annotation_setup.py`](../../src/pragmata/api/annotation_setup.py) extends to `up` / `down`:

```python
# src/pragmata/api/annotation_stack.py (proposed)
from pragmata.core.settings.settings_base import UNSET, Unset

def up(
    *,
    external_postgres: str | Unset = UNSET,
    external_elastic: str | Unset = UNSET,
    external_redis: str | Unset = UNSET,
    profile: str | Unset = UNSET,             # explicit override; normally derived from the external_* kwargs
    pull: str | Unset = UNSET,                # "missing" (default) | "always" | "never"
    health_timeout_seconds: int | Unset = UNSET,  # default 120
    config_path: str | Path | Unset = UNSET,
) -> StackUpResult:
    """Start the annotation stack. Idempotent; safe to re-invoke on a running stack."""
    ...

def down(
    *,
    volumes: bool = False,                    # if True, also removes named volumes (destructive)
    config_path: str | Path | Unset = UNSET,
) -> StackDownResult:
    """Stop the annotation stack. With volumes=True, also wipes persistent data."""
    ...
```

**Implementation shape (not a contract, but worth pinning):** the API function resolves settings via the standard `AnnotationStackSettings.resolve(...)` chain (§1.4 of the config doc), builds the `docker compose` argv (`docker compose --file <importlib-resolved path> --project-name pragmata_annotation up -d ...`), invokes it via `subprocess.run`, captures and surfaces upstream Docker errors with the failure-mode taxonomy in §6, then health-polls and returns a `StackUpResult` (Argilla URL, image-pull summary, time-to-healthy). No Docker SDK dependency - subprocess is what the existing dev Makefile already uses and what every surveyed Python wrapper (Prefect, Dagster, Airbyte for the dev path) does too. The Docker SDK would buy nothing here and adds a heavy non-essential dep.

**CLI mirror:**

```python
# src/pragmata/cli/commands/annotation.py (proposed extension)
@annotation_app.command("up")
def up_command(
    external_postgres: str | None = typer.Option(None, "--external-postgres"),
    external_elastic: str | None = typer.Option(None, "--external-elastic"),
    external_redis: str | None = typer.Option(None, "--external-redis"),
    config: str | None = _config_opt,
) -> None:
    from pragmata import annotation
    result = annotation.up(
        external_postgres=UNSET if external_postgres is None else external_postgres,
        external_elastic=UNSET if external_elastic is None else external_elastic,
        external_redis=UNSET if external_redis is None else external_redis,
        config_path=UNSET if config is None else config,
    )
    typer.echo(f"Argilla ready at {result.url}")
```

This means programmatic use is supported on day one:

```python
from pragmata import annotation
annotation.up(external_postgres="postgresql://user:pass@db.internal/argilla")
annotation.setup(users=[...])
# ... import / export / iaa ...
annotation.down()
```

The settings resolution chain (`flag > env > config_path > project config > user config > defaults`) applies *unmodified* to `up` / `down` kwargs that go through the pragmata-side resolver. The one exception is settings consumed by Compose substitution at file-load time (currently only `PRAGMATA_ANNOTATION_RESTART_POLICY`) - those are env-only and bypass the resolver, see §1.

## 4. Prod bootstrap / unattended install

**Decision for v0.1: no shipped script. Document the two-line install (`pip install 'pragmata[annotation]' && pragmata annotation up`) in the README.**

The two-line install works identically across pip, `pipx install 'pragmata[annotation]'`, and `uv tool install 'pragmata[annotation]'` - in all three cases the `pragmata` entry point ends up on PATH and `annotation up` resolves the packaged compose the same way. No installer-specific branching in the docs is needed; just show `pip` as the default and mention pipx/uv work the same.

**Scope of "prod" for the two-line install: single-host, single-operator (or co-located annotators on the same machine).** Multi-annotator across machines needs a reverse proxy on top - see §4.3.

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

The shipped compose binds Argilla to `127.0.0.1:6900` (loopback only) as a safety default - a service bound to `0.0.0.0` on a cloud VM is immediately reachable from the public internet, and Argilla's plain-HTTP auth is not designed for that exposure. Two deployment shapes follow:

**Single-host (in scope for the two-line install).** Operator runs `pragmata annotation up` on a machine; annotators sit on the same machine and hit `http://localhost:6900` in their browsers. Works out of the box. Common for solo research workflows, demos, single-annotator pilot studies.

**Multi-annotator across machines (requires operator setup beyond the two-line install).** Annotators on their own laptops need to reach Argilla on a server. The supported pattern is a reverse proxy on the host that terminates TLS and forwards to `127.0.0.1:6900`:

- Reverse proxy of choice: [Caddy](https://caddyserver.com/docs/quick-starts/reverse-proxy) is the lowest-friction option (automatic Let's Encrypt, ~3-line config); nginx and Traefik are equally valid. **Alternatively, in environments where inbound public ports aren't available** (university networks, locked-down VPCs, no public IP), [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) achieves the same end via an outbound-only connection from the host - no inbound firewall rules, no DNS, no cert lifecycle on the operator side. Operator owns the proxy / tunnel lifecycle either way; pragmata does not manage it.
- Argilla's own credentials (`ARGILLA_USERNAME` / `ARGILLA_PASSWORD` / `ARGILLA_API_KEY`) must be set to non-default values before `annotation up` - the dev defaults (`argilla` / `argilla123`) are well-known and unsafe on a network-reachable instance.
- The host's firewall / network security group must allow inbound `:443` (proxy) and block direct `:6900` access from outside the host.

We deliberately do *not* in v0.1:
- Ship a CLI flag for the bind address (`--bind 0.0.0.0`). A flag invites operators to expose plain-HTTP Argilla on a public IP, which is the failure mode the loopback default exists to prevent. 
- Ship a Caddy / Traefik sidecar profile. Real implementation work (TLS lifecycle, hostname config, optional basic-auth layer) for a path the operator can stand up in 5 lines themselves. 
- Ship a `--port` flag or `PRAGMATA_ANNOTATION_ARGILLA_PORT` env var that changes the bind port. The shipped compose binds to `:6900` unconditionally in v0.1. (See "Client URL ↔ server port" below.)

**Client URL ↔ server port (compat note for [`config-and-settings.md`](config-and-settings.md) §1.4).** PR 162 lists `PRAGMATA_ANNOTATION_ARGILLA_URL` as an example client-side setting - this is what the SDK and `pragmata annotation setup` / `import` / `export` use to *reach* a running Argilla. The shipped compose's `:6900` bind is independent: `pragmata annotation up` does not read `PRAGMATA_ANNOTATION_ARGILLA_URL` and will not rebind to a different port if that var is set. In v0.1 it is the operator's responsibility to keep these in sync; the practical contract is:

- Default everywhere is `http://localhost:6900` - leave both unset and everything works.
- If you set `PRAGMATA_ANNOTATION_ARGILLA_URL` to a non-default URL (e.g. talking to a remote Argilla, or running behind a reverse proxy), `up` is not the right command - you're connecting to an existing server, skip `up` entirely and go straight to `setup`.
- A future `--port` flag (and matching `PRAGMATA_ANNOTATION_ARGILLA_PORT`) would close the loop by letting both planes share one knob; deferred until concrete demand exists.

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
| **Docker stack lifecycle (`up`/`down`)** | Not implemented - only Makefile targets (`docker-up`, `docker-down`, etc.) that read [`deploy/annotation/docker-compose.dev.yml`](../../deploy/annotation/docker-compose.dev.yml) | Add `pragmata annotation up` / `down` CLI commands (§3, §3.5) |
| **Compose file distribution** | Dev-only file in `deploy/`; **not shipped** in the installed wheel | Ship as package data + contributor dev override (§1, §2.1, §2.2) |
| **Package data** | `importlib.resources` already used for [`core/annotation/collapsible_field.html`](../../src/pragmata/core/annotation/collapsible_field.html) | Same mechanism for the compose file (§2.2) |

## References

- [ADR-0007 - Packaging & invocation surface](../decisions/0007-packaging-invocation-surface.md)
- [ADR-0003 - Infra: self-hosted only](../decisions/0003-infra-self-hosted-only.md)
- [`config-and-settings.md`](config-and-settings.md) - shared settings/config resolution
- Precedent for Docker orchestration + compose distribution: [Supabase CLI](https://github.com/supabase/cli), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/3.0/), [Dagster](https://docs.dagster.io/), [Airflow Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
