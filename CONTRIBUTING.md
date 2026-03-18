# Internal Contributing Guidelines

This document defines mandatory contribution rules for the `pRAGmata` repository.
It is intended for internal team members only.

External contributions are currently not accepted.
Pull requests from non-team members may be closed without review.

The goal is to ensure:
- regular, predictable development
- small, reviewable changes
- a clean and stable `main` branch

These rules are not optional.


## 1. Branching Model

- The `main` branch is protected.
- Direct commits to `main` by non-admins are not allowed.
- All work must be done on task-specific branches, created from `main`.
- Branches should be short-lived and scoped to a single PR.

### Branch naming

Branches for PRs should follow a simple, consistent naming scheme.

- feature/short-description
- fix/short-description
- docs/short-description
- refactor/short-description


## 2. Pull Requests (PRs)

### General Rules

- All changes must be merged via Pull Requests.
- Each PR should represent exactly one substantively coherent change.
- Large, mixed-scope PRs will be closed.

### PR Description Requirements

Every PR must include:

1. Brief summary (what this PR does)
2. Detailed description (rationale and approach)
3. Status flag at the end:
   - `Draft`
   - `Ready for review`

### Commit Messages

Commits within a PR should follow a simple, consistent naming scheme.

Expected prefixes:

- `feat:` new functionality
- `fix:` bug fixes
- `refactor:` code restructuring without behavior change
- `test:` adding or modifying tests
- `docs:` documentation only
- `ci:` CI or tooling changes
- `chore:` maintenance tasks without functional impact

### Testing Requirements

- Feature PRs must include tests.

If tests are not added, the PR must explicitly explain why.

### Merge Strategy

- PRs are merged into `main` using squash merges only.
- Merge commits are not allowed.


## 3. Review Policy

- Every PR must be reviewed by one other team member.
- Authors must assign a reviewer.
- Authors should not approve their own PRs.
- Review approval is required before merging.

Reviewers are expected to check:
- scope and clarity
- code quality
- tests (if applicable)
- CI checks (must pass)
