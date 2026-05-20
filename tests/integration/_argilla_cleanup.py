"""Test helpers for clearing Argilla workspace state between integration tests.

Argilla refuses to delete a workspace while any dataset is linked. Tests that
exercise full teardown can leave orphan datasets behind across runs, so this
purge runs *before* `teardown_resources` in test fixtures to keep workspaces
deletable. Production callers should not need this — they own their dataset
ids and clean up via `teardown_resources(settings)`.
"""

import argilla as rg


def purge_workspace_datasets(client: rg.Argilla, ws_base: str) -> None:
    """Delete every dataset in workspace `ws_base`. No-op if workspace is missing."""
    workspace = client.workspaces(ws_base)
    if workspace is None:
        return
    for dataset in list(workspace.datasets):
        dataset.delete()
