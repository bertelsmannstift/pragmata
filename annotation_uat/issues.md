# Issues / Questions to address

- [ ] context: basic setup in argilla is that workspace has multi datasets, users assigned to workspaces (see all datasets in that workspace). We have workspace_prefix currently prefixing both workspace and dataset names (this ws by design so we can run completely isolated environments). But i think multiple runs into the same workspace makes more sense. e.g.
    ```bash
    Workspace: retrieval (permanent, users assigned here)
    Dataset run 1: uat_1_retrieval (inside retrieval workspace)
    Dataset run 2: uat_2_retrieval (inside retrieval workspace)
    ```
= users stay assigned to their workspaces and we can import fresh datasets without re-provisioning users?