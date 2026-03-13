# Annotation User Management

Argilla user account provisioning and credential management for the annotation platform.

## Responsibilities

**In scope:**
- Create and assign annotator accounts in Argilla
- Assign users to workspaces (workspace names are deployment config; defaults: `retrieval`, `grounding`, `generation`)
- Manage API key for import/export scripts
- Handle password resets

**Out of scope:**
- SSO/LDAP/OAuth (deferred — see [ADR-0008](../decisions/0008-annotation-interface-auth.md))
- Automated email-based password recovery

## Account Creation

Argilla v2 does not support self-registration. Accounts must be created by the Owner (see [ADR-0008](../decisions/0008-annotation-interface-auth.md)) via the Argilla Python SDK or CLI. Distribute initial passwords via secure channel; users should change on first login.


## API Key Management

Owner API key is generated automatically by the self-hosted Argilla instance on first run.

- Stored in `.env` (excluded from git); used for import/export scripts
- Rotate via Argilla admin interface → update `.env` → restart service

## Password Reset

Manual process: admin resets via Argilla UI or SDK → distributes new temporary password securely → user changes on login.

## Failure Modes

**Account creation failure:**
- SDK/CLI returns error; check Owner API key is valid and Argilla instance is reachable

**Workspace assignment failure:**
- User created but assigned to wrong workspace; verify via Argilla admin UI and reassign

**API key compromise:**
- Rotate immediately via Argilla admin interface; update `.env`; restart service

## References

- [Decision 0008: Authentication](../decisions/0008-annotation-interface-auth.md)
- [Workspace & Task Distribution](annotation-workspace-task-distribution.md) — workspace definitions
- Argilla user management docs: https://docs.argilla.io/latest/
