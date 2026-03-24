# Decision Records

This directory contains architectural and tooling decisions with rationale and consequences.

Each decision document follows this structure:
- **Status:** Draft / Accepted / Superseded
- **Decision:** What we chose
- **Rationale:** Why we chose it (with alternatives considered)
- **Consequences:** Implications (positive, negative, neutral)

## Index

- [0001: Annotation — Argilla Platform](0001-annotation-argilla-platform.md) — Accepted: use Argilla for human annotation interface
- [0002: Evaluation Approach](0002-evaluation-approach.md) — Accepted: supervised reference-based evaluation via fine-tuned cross-encoders
- [0003: Infra — Self-Hosted Deployment](0003-infra-self-hosted-only.md) — Accepted: cloud-hosted deployment out of scope
- [0004: Synthetic Query Generation](0004-synthetic-query-generation.md) — Accepted: staged, spec-driven LLM workflow for controlled synthetic query construction
- [0005: Contract Layer Tooling](0005-contract-layer-tooling.md) — Draft: Pydantic at boundaries, frozen dataclasses for runtime, CSV interchange, per-tool settings
- [0007: Invocation Surface](0007-packaging-invocation-surface.md) — Accepted: two supported invocation surfaces (Python API and CLI)
- [0008: Annotation Authentication Interface](0008-annotation-interface-auth.md) — Accepted: Argilla built-in auth with role mapping
- [0009: Annotation Schema Configurability](0009-annotation-schema-configurability.md) — Accepted: hardcoded schemas for v1.0, configurability deferred
- [0010: Multi-Dataset Architecture](0010-annotation-multi-dataset-architecture.md) — Draft: three separate Argilla datasets (one per task)
- [0011: Annotation Import Formats](0011-annotation-import-formats.md) — Draft: import accepts JSON, JSONL, CSV, HF Dataset, DataFrame
