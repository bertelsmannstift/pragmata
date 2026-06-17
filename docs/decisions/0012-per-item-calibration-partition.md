# 0012: Per-Item Calibration Partition

Status: Draft

## Decision

Calibration vs production assignment is **per annotation item**, not per `record_uuid`. The annotation item differs by task:

| Task | Annotation item | Items per `QueryResponsePair` |
|---|---|---|
| Grounding | `record_uuid` | 1 |
| Generation | `record_uuid` | 1 |
| Retrieval | `(record_uuid, chunk_id)` | N (one per chunk) |

## Rationale

- **Per-record calibration** preserves query-level bundling (TREC and its descendants partition per topic), but pragmata's current naive row-bootstrap (`bootstrap_alpha` in `core/annotation/iaa.py`) silently inflates Krippendorff α via cluster correlation. Adopting it would require also adopting cluster-aware IAA machinery (G-theory, per-query α aggregation, multilevel κ).
- **Per-item calibration** matches the actual annotation units, especially retrieval chunks (each chunk is its own Argilla record), and is internally consistent with the IAA layer pragmata uses today.
- Statistical analysis confirms the gap: Kish's design effect (ρ ≈ 0.2–0.4, m̄ = 5 chunks/query) gives per-record ≈ 38–56 effective samples vs per-item ≈ 100 at equal count — a 1.8–2.6× honest efficiency gain.
- Choosing per-record would imply also taking on cluster-aware IAA work; per-item avoids that dependency and is consistent with modern retrieval-evaluation practice (TREC-DL'19, TripJudge, D-MERIT).

## Consequences

**Positive**

- Statistically honest α and CIs under the current naive bootstrap; no silent inflation from cluster correlation.
- Per-task right-sizing: operators can size each task's calibration set independently via the inheritable `calibration_fraction` and `calibration_max_records`.
- Matches Argilla's per-item record creation (each retrieval chunk is its own Argilla record).
- Matches modern RAG eval annotation precedent (D-MERIT, TripJudge, ARES).

**Negative**

- Schema break for any pre-v0 on-disk manifests: legacy `calibration: bool` entries no longer load. Pragmata is pre-1.0; affected workspaces re-bootstrap by deleting `partition.meta.json` and re-importing.
- The implicit per-`record_uuid` bundling that ADR-0010 assumed weakens - the same `record_uuid` can have some chunks in retrieval-calibration and others in production. The dataset-per-task structure itself is preserved; what changes is that calibration is a property of the annotation item, not the parent record.
- **Order-dependence under binding cap**: when the cap is binding across multiple imports, the resulting calibration set is a function of `(corpus, seed, import_order)`, not `(corpus, seed)` alone. This is a consequence of the manifest-lock invariant - once an entry is in calibration, a later tightened cap cannot demote it. Documented in `assign_partitions` docstring.

## Out of scope - future work

- **Cluster-aware IAA** (G-theory, per-query α aggregation, cluster bootstrap, multilevel κ). The main alternative path that would make per-record calibration statistically defensible; if pragmata later wants a per-query IAA narrative (which the academic literature supports under correct statistical machinery), this is the layer to upgrade.

## References

- [ADR-0010](0010-annotation-multi-dataset-architecture.md) - modified in spirit (calibration as property of annotation item rather than property of `record_uuid`); the dataset-per-task structure is preserved.
- [PR #206 (`f1c6d9f`)](https://github.com/bertelsmannstift/pragmata/pull/206) - `Inherit` sentinel recipe for `production_min_submitted`, `calibration_min_submitted`, `locale`. The same pattern extends here to `calibration_fraction` and `calibration_max_records`.
- [Annotation Import Pipeline §Calibration partitioning](../design/annotation-import-pipeline.md#calibration-partitioning) - implementation-level description of the per-item bucketing algorithm, manifest schema, and order-dependence note.
- Faggioli et al. 2025, "Variations in Relevance Judgments and the Shelf Life of Test Collections" - [arXiv:2502.20937](https://arxiv.org/abs/2502.20937).
- Zhang et al. 2024, "D-MERIT: Evaluating D-MERIT of Partial-annotation on Information Retrieval" - [arXiv:2406.16048](https://arxiv.org/abs/2406.16048).
- Hofstätter et al., TripJudge: [arXiv:2208.06936](https://arxiv.org/abs/2208.06936).
- Saad-Falcon et al. 2023, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems" - [arXiv:2311.09476](https://arxiv.org/abs/2311.09476).
- Kish 1965, *Survey Sampling*, John Wiley & Sons - design effect formula.
