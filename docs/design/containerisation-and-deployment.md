# Containerisation & Production Deployment

Status: Draft
Related:
- ADR-0003 (infra: self-hosted only)
- ADR-0007 (packaging & invocation surface)
- ADR-0012 (install & bootstrap UX) [draft]
- [`config-and-settings.md`](config-and-settings.md) - shared settings/config
- [`annotation-bootstrap.md`](annotation-bootstrap.md) - annotation stack lifecycle, compose distribution

## Purpose

How pragmata is distributed and deployed in production. Covers the artefact strategy (PyPI vs container image), how the three tools differ in deployment shape, persistence/backup, TLS, and the operational story for a self-hosted Argilla campaign that runs for weeks.

Annotation stack composition (services, profiles, compose-file location) is in [`annotation-bootstrap.md`](annotation-bootstrap.md). Settings resolution is in [`config-and-settings.md`](config-and-settings.md). This doc is purely about **how the artefact ships and how operators run it**.

## Guiding principle

**The primary artefact is the PyPI package. A container image is an optional convenience, not the deployment unit.**

Pragmata's three tools split into two deployment shapes:

- `querygen`, `eval` - short-lived batch jobs. Run on a developer laptop or a CI runner. No infra to manage.
- `annotation` - orchestrates a long-running Argilla stack (Argilla server + Postgres + Elasticsearch + Redis) that must stay up for the duration of an annotation campaign (weeks).

These two shapes do not benefit from the same packaging. Trying to unify them under a single "ship one Docker image and run everything from it" model creates more friction than it removes - notably, it forces the question of how a containerised CLI orchestrates sibling containers, whose only common answer (mounting `/var/run/docker.sock`) is rejected as a production anti-pattern by every Tier-1 OSS project surveyed (§3.1).

## 1. Artefact strategy

### 1.1 What we ship

| Artefact | Status | Purpose |
|---|---|---|
| **PyPI package** (`pragmata`, `pragmata[annotation\|querygen\|eval]`) | **Primary, v0.1** | Single source of truth. Library + CLI for all three tools. Ships the locked annotation compose file as package data ([`annotation-bootstrap.md`](annotation-bootstrap.md) §2). |
| **Container image** (`ghcr.io/.../pragmata:<version>`) | **Deferred to v0.2+** | Convenience wrapper for `querygen`/`eval` jobs in CI/cloud-run contexts. Not load-bearing - users with Python can `pip install` instead. Not the path for `annotation` (§3). |
| **Helm chart** | Out of scope | Reach for only if/when a deployer asks for k8s-native annotation deployment. |

For v0.1 the PyPI package is sufficient. Adding a container image is cheap once we have a release pipeline (one extra `docker build && push` step in CI) but it is not on the critical path.

### 1.2 Why not container-as-primary

Surveyed Tier-1 PyPI-distributed CLIs that wrap infrastructure: **Supabase CLI, Airbyte `abctl`, Prefect, Dagster, MLflow, dbt Core**. Pattern is consistent:

- Primary artefact is a binary or a pip-installable package.
- Container images, where they exist, are secondary - for users who want isolation or reproducibility.
- None of them route the *orchestrator* through a container in production - the binary/CLI runs on the host and talks to Docker (or k8s) directly.

Reasons this matters for pragmata:

- **`querygen`/`eval`** are short-lived. A developer running `pragmata querygen gen-queries` from a notebook or shell does not want `docker pull` overhead per invocation.
- **`annotation`** must spawn its own sibling stack. Containerising the orchestrator forces the DooD question (§3.1) for no real gain - the host is already a Linux box with Docker on it (that's a hard prerequisite anyway).
- **Two artefacts to maintain** = two release surfaces, two upgrade stories, two version-skew matrices. v0.1 cannot afford that.

### 1.3 Versioning & registry

When/if we publish a container image (v0.2+):

- **Registry**: GHCR (`ghcr.io/bertelsmannstift/pragmata`). Free for public repos, no separate credentials, builds straight from the GitHub release workflow. Docker Hub and Quay are viable alternatives but offer no concrete advantage given the codebase already lives on GitHub.
- **Tags**: pin to the released package version (`ghcr.io/.../pragmata:0.2.0`). Also publish `:latest` for convenience but always document pinned tags in install snippets.
- **Image-vs-package version**: image and PyPI package release in lockstep - the image just `pip install`s the same wheel. No independent version axis.

Sidecar images (Argilla, Postgres, Elasticsearch, Redis) are pinned in the shipped compose file (digests, not floating tags - see [`annotation-bootstrap.md`](annotation-bootstrap.md) §2.2). Those upgrade with the pragmata release that bumps them.

## 2. Tool-by-tool deployment shape

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT SHAPES                               │
├─────────────────┬───────────────────┬────────────────────────────────────┤
│ Tool            │ Lifecycle         │ Operator runs                      │
├─────────────────┼───────────────────┼────────────────────────────────────┤
│ querygen        │ short-lived job   │ pip install + invoke               │
│ eval            │ short-lived job   │ pip install + invoke               │
│ annotation      │ long-running infra│ pip install + `annotation up`      │
│                 │ (stack stays up   │ which spawns Argilla + PG + ES +   │
│                 │  during campaign) │ Redis via host docker compose      │
└─────────────────┴───────────────────┴────────────────────────────────────┘
```

### 2.1 `querygen` / `eval` - batch jobs

**Production shape**: dev laptop, CI runner, or a cloud batch job (Cloud Run, AWS Batch, Azure Container Instances).

```bash
# Local / CI
pip install 'pragmata[querygen]'
export OPENAI_API_KEY=...
pragmata querygen gen-queries --config querygen.yml --base-dir ./out

# Containerised (deferred to v0.2)
docker run --rm \
  -e OPENAI_API_KEY=... \
  -v "$PWD/config:/config" \
  -v "$PWD/out:/workspace" \
  ghcr.io/bertelsmannstift/pragmata:0.2.0 \
  pragmata querygen gen-queries --config /config/querygen.yml --base-dir /workspace
```

The container form is purely a convenience - same wheel, just pre-installed. No operational difference.

### 2.2 `annotation` - long-running stack

**Production shape**: a Linux VM (or k8s node) that the deploying organisation owns. Pragmata installed via pip; the Argilla stack runs as a sibling Compose project on the same host.

```bash
# On the deployment VM
pipx install 'pragmata[annotation]'
pragmata annotation up                  # starts the stack on the host
pragmata annotation setup --url http://localhost:6900 --api-key ...
                                        # provisions Argilla workspaces / users / datasets
# stack runs for weeks; annotators access via reverse proxy (§4)
pragmata annotation export ...          # ad-hoc export jobs against the running stack
pragmata annotation down                # at end of campaign
```

`pragmata annotation up` resolves the package-data compose file via `importlib.resources` and shells out to `docker compose up -d` on the host (§3.2). The pragmata process itself is **not** containerised - it runs on the host as a normal Python process and exits after kicking the stack off.

This matches the Supabase CLI model: the orchestrator runs on the host, talks to the host Docker daemon, and gets out of the way. The CLI's job is to materialise a known-good compose configuration and invoke the daemon - not to host its own runtime container.

## 3. The "containerised orchestrator" question

This is the question SG flagged. Resolution: **for v0.1 we do not run pragmata-itself in a container in the annotation deployment path.** The annotation operator runs `pragmata annotation up` directly on the host.

If at some point a deployer insists on running the pragmata CLI inside a container (e.g. immutable infra, no host Python), the supported answer is option 3.2 below: skip `annotation up` entirely and let them run `docker compose -f <shipped-file> up` directly. We do not support DooD.

### 3.1 Why not Docker-outside-of-Docker (DooD)

DooD = mount `/var/run/docker.sock` into a container so it can spawn sibling containers on the host daemon.

- **Security**: Docker socket access is root-equivalent on the host. OWASP, Docker's own docs, and every security review treat socket-mounting in production as a privilege-escalation vector. Read-only mount does not help. Rootless Docker mitigates but does not eliminate.
- **Operator confusion**: makes the trust boundary between "the pragmata container" and "the host's whole container fleet" invisible. If the pragmata container is compromised, the attacker owns every container on the host.
- **No precedent in our peer set**: Supabase, Airbyte `abctl`, Dagster, Prefect, MLflow, dbt - none ship a containerised CLI that DooDs a sibling stack as the recommended path. Earthly explicitly rejects DooD as a design principle. Where DooD shows up at all (Dagster runners, some CI patterns) it's flagged dev-only.
- **Zero gain over running the CLI on the host**: pragmata is a Python process that takes <1s to invoke and then exits. Containerising it does not buy isolation worth the security tradeoff.

### 3.2 Escape hatch: skip `annotation up`, run compose directly

If a deployer cannot or does not want to install pragmata on the host (e.g. immutable infra, hardened image policy), they can run the shipped compose file directly:

```bash
# Extract the compose file from the wheel without installing pragmata system-wide
python -m pip download --no-deps pragmata
unzip -j pragmata-*.whl 'pragmata/annotation/docker-compose.yml' -d ./deploy
# Or: clone the repo and use deploy/annotation/docker-compose.yml

docker compose -f ./deploy/docker-compose.yml up -d
# Provisioning still needs the CLI - run it from anywhere with network access to Argilla:
pipx run 'pragmata[annotation]' annotation setup --url http://argilla.host:6900 ...
```

This is an explicit escape hatch, not the happy path. We document it; we do not optimise for it.

### 3.3 What we ruled out

| Option | Why not |
|---|---|
| DooD socket mount | §3.1 |
| Docker-in-Docker (DinD, privileged container) | Worse than DooD - same security profile plus a nested daemon to manage |
| Sibling-pod-spawn from a kind/k3s cluster (Airbyte `abctl` style) | Adds a kubernetes runtime as a hard dependency for users who just want annotation. Massive over-engineering. |
| Two separate images ("job runner" + "stack") | Two release surfaces, no benefit - the "stack" image is just Argilla's existing image |

## 4. Production operations

These are the operator's responsibility, not pragmata's, but the design must not actively obstruct them.

### 4.1 Persistence & backup

Argilla writes to Postgres (annotation submissions, users, workspaces) and Elasticsearch (record indexes). Both must survive container recreation across upgrades and restarts.

- Compose file declares **named volumes** with deterministic prefixes (`pragmata_annotation_postgres_data`, `pragmata_annotation_elastic_data`, `pragmata_annotation_redis_data`). Already the case in the current dev compose.
- `pragmata annotation down` does **not** wipe volumes; `pragmata annotation down --volumes` does (already specified in [`annotation-bootstrap.md`](annotation-bootstrap.md) §3.3).
- Backup is operator-owned. Document the standard recipe:
  - `docker compose exec postgres pg_dump -U postgres argilla > backup.sql`
  - Elasticsearch native snapshot API (or volume-level snapshot if the storage supports it)
  - Schedule via host cron / systemd timer; pragmata does not ship a backup verb in v0.1

### 4.2 TLS, reverse proxy, auth

Argilla speaks HTTP on port 6900. The shipped compose **binds to localhost only** by default (so a fresh `annotation up` does not silently expose Argilla to the public internet). Production exposure is the operator's call.

Recommended pattern (documented, not enforced):

```
                   ┌─────────────────────┐
   annotators ───► │ nginx (TLS, 443)    │
                   │ optionally: oauth2  │ ──► localhost:6900 (argilla)
                   │ proxy / SSO         │
                   └─────────────────────┘
                        host VM
```

- TLS termination at nginx with Let's Encrypt or org cert
- Argilla's built-in user/role system handles annotator auth; no SSO requirement for v0.1
- If SSO is required: front with `oauth2-proxy` or equivalent

This is a docs concern, not a code concern. The compose file just needs to default to localhost binding so we don't ship a footgun.

### 4.3 Deployment topology

| Topology | When | Notes |
|---|---|---|
| **Single VM + docker compose** | default - any campaign | Linux VM, pragmata installed via pipx, Argilla stack via `annotation up`. Matches Argilla's own recommended deployment. |
| **Kubernetes** | only if the deployer already runs k8s | No Helm chart from pragmata in v0.1. Operator translates the compose file or uses Argilla's own k8s deployment. Pragmata CLI runs as a Job for setup/import/export. |
| **Managed Argilla** | out of scope | ADR-0003 - self-hosted only |

For Bertelsmann Stiftung's first campaigns the single-VM compose deployment is the realistic target. K8s is a future-someone-else's-problem path.

## 5. Open questions

| Question | Status |
|---|---|
| **Container image in v0.1?** Currently deferred to v0.2+ (§1.1). Worth confirming we're OK shipping PyPI-only for v0.1 - even with the dev/CI ergonomic loss for `querygen`/`eval` users who'd prefer `docker run`. | Open |
| **Localhost-only binding default for the shipped compose?** §4.2 proposes binding Argilla to `127.0.0.1:6900` by default to avoid accidental public exposure. The current dev compose binds to `0.0.0.0:6900` (`"${ARGILLA_PORT:-6900}:6900"`). Confirm we want to flip this for the shipped (prod-first) compose. | Open |
| **Backup verb?** §4.1 leaves backup to the operator. Worth checking whether a thin `pragmata annotation backup` / `restore` (wrapping `pg_dump` + ES snapshot) is in scope for v0.1 or v0.2+. | Open |

## References

- [ADR-0003 - Infra: self-hosted only](../decisions/0003-infra-self-hosted-only.md)
- [ADR-0007 - Packaging & invocation surface](../decisions/0007-packaging-invocation-surface.md)
- [`config-and-settings.md`](config-and-settings.md) - settings resolution
- [`annotation-bootstrap.md`](annotation-bootstrap.md) - stack composition, compose distribution, lifecycle
- Precedent: [Supabase CLI](https://supabase.com/docs/guides/local-development), [Airbyte abctl](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart), [Prefect](https://docs.prefect.io/), [Dagster](https://docs.dagster.io/), [MLflow](https://mlflow.org/), [dbt Core](https://docs.getdbt.com/), [Argilla self-hosting](https://docs.argilla.io/latest/getting_started/how-to-deploy-argilla-with-docker/)
- DooD security: [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
