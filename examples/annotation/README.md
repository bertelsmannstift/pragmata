# `examples/annotation/`

Reference schemas for the two files an operator typically provides to
`pragmata annotation` commands. Both are optional — `pragmata annotation
setup` runs zero-config against the built-in three-workspace default
topology (`retrieval`, `grounding`, `generation`), and a separate config
YAML is only needed if you want to deviate from those defaults.

| File | Used by | Optional? |
|---|---|---|
| [`users.example.json`](users.example.json) | `pragmata annotation setup --users <path>` | Yes — omit to skip user provisioning |
| [`topic.example.jsonl`](topic.example.jsonl) | `pragmata annotation import <path>` | No (records are the input data) |

These files ship with the repo as reference schemas; copy them to your own
working directory and edit, then point CLI flags at the copies. They are
not shipped in the installed wheel.

## `users.example.json` — annotator/owner roster

A JSON array of [`UserSpec`](../../src/pragmata/core/settings/annotation_settings.py)
entries. Each entry needs `username`, `role` (`annotator` or `owner`), and
the list of `workspaces` to grant access to. Pragmata reads this file when
you pass `--users <path>` to `setup`.

`password` is **optional**. Omit it and pragmata auto-generates a password
per user, which is then echoed once to the terminal by the CLI so you can
record it. Don't commit passwords to git — the schema deliberately shows
the passwordless shape.

## `topic.example.jsonl` — input records

One JSON object per line, each a
[`QueryResponsePair`](../../src/pragmata/core/schemas/query_response_pair.py)
with the fields shown in the example (`query`, `answer`, `context_set`,
`chunks`, optional `language`). The bundled example mixes English and
German records to demonstrate the `language` field.

`pragmata annotation import path.jsonl` consumes this format.

## Customising via `--config` (optional)

If you want named workspaces (e.g. `bildung-retrieval` instead of the
default `retrieval`) or non-default overlap thresholds, write a YAML file
and pass `--config <path>` to `setup` / `import`:

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
[`AnnotationSettings`](../../src/pragmata/core/settings/annotation_settings.py).
Connection settings (`argilla.api_url`, `api_key`) belong on the env / CLI
layer, not in this YAML — see
[`docs/design/config-and-settings.md`](../../docs/design/config-and-settings.md)
for the precedence chain and secrets policy.
