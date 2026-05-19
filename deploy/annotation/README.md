# `deploy/annotation/` — contributor stack scaffolding

> **End users**: this directory is for contributors working on pragmata
> in a cloned repo. To run pragmata as an end user, see the top-level
> README — the supported path is `pip install 'pragmata[annotation]'`
> then `pragmata annotation up` (CLI lifecycle, forthcoming per
> [`docs/design/annotation-stack-lifecycle.md`](../../docs/design/annotation-stack-lifecycle.md)).
> Operator-facing input schemas (annotator roster, record format) ship
> as package data under
> [`src/pragmata/annotation/examples/`](../../src/pragmata/annotation/examples/README.md);
> pip-installed users resolve them via `importlib.resources`.

This directory holds the contributor-only artefacts needed to run the
Argilla annotation stack in dev: the Docker Compose definition, the
template env file, and the per-developer `.env` (gitignored).

## Files

| Path | Tracked? | Purpose |
|---|---|---|
| `docker-compose.dev.yml` | yes | Argilla stack (Argilla + Postgres + Elasticsearch + Redis) used by the `make docker-*` targets. Will split into a shipped `src/pragmata/annotation/docker-compose.yml` (package data) + a dev-only `docker-compose.dev.override.yml` per the lifecycle design doc — that work is tracked separately. |
| `.env.dev.example` | yes | Template with the well-known dev credentials and the env-var contract for the four backing-service profiles. |
| `.env` | no (gitignored) | Real env for your local dev stack. Created by `make ensure-env` (run automatically by `make docker-up`) from `.env.dev.example`. |

## Starting and stopping the dev stack

```bash
make docker-up                 # all services bundled (default profile)
make docker-up-external-pg     # bring your own Postgres
make docker-up-external-es     # bring your own Elasticsearch
make docker-up-external-redis  # bring your own Redis

make docker-down               # stop stack, preserve volumes (safe default)
make docker-down-clean         # stop stack, remove volumes (destructive)
make docker-status             # show service health
make docker-logs               # tail container logs

make test-stack                # smoke-test the stack in an isolated Compose project
```

`make docker-up` brings Argilla up at `http://localhost:6900`. The
well-known dev credentials are in [`.env.dev.example`](.env.dev.example).
End-user (prod) deployments **do not** use the same credentials — set
`ARGILLA_USERNAME`, `ARGILLA_PASSWORD`, `ARGILLA_API_KEY` in the
environment before running the CLI's eventual `pragmata annotation up`.

## Smoke-testing your changes

Once the stack is up, the simplest end-to-end check uses the example
fixtures shipped as package data at
[`src/pragmata/annotation/examples/`](../../src/pragmata/annotation/examples/README.md):

```bash
pragmata annotation setup --users src/pragmata/annotation/examples/users.example.json
pragmata annotation import src/pragmata/annotation/examples/topic.example.jsonl --dataset-id smoke
pragmata annotation export --dataset-id smoke
pragmata annotation teardown --dataset-id smoke
```

Zero-config: `setup` provisions the built-in three-workspace topology
(`retrieval`, `grounding`, `generation`). See
[`src/pragmata/annotation/examples/README.md`](../../src/pragmata/annotation/examples/README.md)
for the full operator-facing schemas and how to deviate from defaults.

## Adding a new UI locale

Annotation UI strings (Argilla dataset titles/questions/guidelines + the
custom discard widget) live in
[`src/pragmata/core/annotation/locales/`](../../src/pragmata/core/annotation/locales/).
Add a new locale by extending the catalog there; unit tests enforce key
completeness against the English source of truth.

Stored label *values* (`yes` / `no`, `DiscardReason.*.value`) and field
`name=` identifiers stay stable across locales, so exports merge cleanly
across multi-language deployments.

## Settings, secrets, and config layering

These concerns are shared across all pragmata tools; see
[`docs/design/config-and-settings.md`](../../docs/design/config-and-settings.md)
for the resolution chain (CLI flag > env > explicit `--config` >
auto-discovered project config > auto-discovered user config > defaults),
the `PRAGMATA_<TOOL>_<KEY>` env-var convention, and the env-only secrets
policy. Auto-discovery (project `./pragmata.yaml` /
`pyproject.toml [tool.pragmata]` and `platformdirs` user config) is not
yet implemented — until it lands, explicit `--config <path>` is the only
file-based settings layer.
