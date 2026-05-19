# Annotation example schemas

Reference schemas for the inputs that `pragmata annotation` commands
consume. These files ship as package data inside the installed wheel
(via `importlib.resources`) so end users on `pip install
'pragmata[annotation]'` can resolve them programmatically — no clone of
the repo required.

| File | Used by | Optional? |
|---|---|---|
| `users.example.json` | `pragmata annotation setup --users <path>` | Yes — omit `--users` to skip user provisioning |
| `topic.example.jsonl` | `pragmata annotation import <path>` | No (records are the input data) |

## Accessing the examples

Resolve via `importlib.resources` and read or copy:

```python
from importlib.resources import files

users_path = files("pragmata.annotation.examples") / "users.example.json"
print(users_path.read_text())
```

Or, if you've cloned the repo, the files live at
`src/pragmata/annotation/examples/` next to this README.

A `pragmata annotation init <dir>` verb that scaffolds these examples
into a working directory is planned (lifecycle Block B / C).

## `users.example.json` — annotator/owner roster

A JSON array of `UserSpec` entries. Each entry needs `username`, `role`
(`annotator` or `owner`), and the list of `workspaces` to grant access
to. The example uses the default workspace names (`retrieval`,
`grounding`, `generation`) so it works against the zero-config
`pragmata annotation setup` topology.

`password` is **optional**. Omit it and pragmata auto-generates a
password per user, which is echoed once to the terminal by the CLI so
you can record it. The example deliberately omits passwords — don't
commit them to git.

Full schema: `pragmata.core.settings.annotation_settings.UserSpec`.

## `topic.example.jsonl` — input records

One JSON object per line, each a `QueryResponsePair` with the fields
shown in the example (`query`, `answer`, `context_set`, `chunks`,
optional `language`). The bundled example mixes English and German
records to demonstrate the `language` field. The dataset_id used at
import time is the run-scoping suffix; the records themselves are
language- and topic-agnostic.

Full schema: `pragmata.core.schemas.annotation_import.QueryResponsePair`.

## Customising the workspace topology (optional)

If you want topic-prefixed workspaces (e.g. `bildung-retrieval` instead
of the default `retrieval`) or non-default overlap thresholds, write a
YAML config and pass `--config <path>` to `setup` and `import`:

```yaml
workspace_dataset_map:
  bildung-retrieval:
    retrieval: {}
  bildung-grounding:
    grounding: {}
  bildung-generation:
    generation: {}

calibration_fraction: 0.1
```

The full set of tunable fields lives in
`pragmata.core.settings.annotation_settings.AnnotationSettings`.
Connection settings (`argilla.api_url`, `api_key`) belong on the env /
CLI layer, not in this YAML — see `docs/design/config-and-settings.md`
in the repo for the precedence chain and secrets policy.

If you pass a `--config` with custom workspace names, the
`users.example.json` workspace assignments need updating to match.
