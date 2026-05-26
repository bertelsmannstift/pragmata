# 0012: Per-Item Calibration Partition

Status: Draft

## Decision

Calibration vs production assignment is **per annotation item**, not per `record_uuid`. The annotation item differs by task:

| Task | Annotation item | Items per `QueryResponsePair` |
|---|---|---|
| Grounding | `record_uuid` | 1 |
| Generation | `record_uuid` | 1 |
| Retrieval | `(record_uuid, chunk_id)` | N (one per chunk) |

The `PartitionManifestEntry` schema reflects this asymmetry:

```python
grounding_generation_calibration: dict[Task, bool]    # GROUNDING, GENERATION
retrieval_chunk_calibration: dict[str, bool]          # keyed by chunk_id
```

Per-task `calibration_fraction` and `calibration_max_records` are inheritable across deployment / workspace / task scopes via the `Inherit` sentinel pattern established in PR #206. Cap unit is the annotation item: a cap of 200 on retrieval means 200 chunks; on grounding/generation, 200 records.

## Context - design alternatives

Two viable partition granularities were weighed, with a genuine tension between IR convention and modern statistical practice:

**Option A - per-`record_uuid` partition** (the pre-existing model). Defensible on convention grounds: TREC and its descendants partition per topic; within-query chunks share annotator context; per-record bundling preserves a query-level IAA narrative. TREC's per-topic tradition assumes cluster-aware IAA analysis (G-theory, per-query α aggregation, multilevel κ) that pragmata does not currently implement.

**Option C - per annotation-item partition** (this ADR). Aligned with modern retrieval-evaluation practice: TREC-DL'19 (Faggioli et al. arXiv:2502.20937), TripJudge (Hofstätter et al. arXiv:2208.06936), and D-MERIT (Zhang et al. arXiv:2406.16048) all retreat from pure per-topic to per-(query, item) for budget and ranking-bias reasons.

Statistical analysis under pragmata's current naive row-bootstrap (`bootstrap_alpha` in `core/annotation/iaa.py`) settles it. Per-record partition silently inflates Krippendorff α and narrows confidence intervals via cluster correlation. Kish's design effect `N_eff = N_total / (1 + ρ(m̄ - 1))` with retrieval-plausible ρ ≈ 0.2–0.4 and m̄ = 5 chunks/query gives Option A's effective sample size ≈ 38–56 vs Option C's ≈ 100 at equal item count - Option C delivers ~1.8–2.6× the honest effective sample size. Adopting Option A would require also adopting cluster-aware IAA machinery; Option C is internally consistent with the IAA layer pragmata uses today.

Practical evidence further supports Option C: Label Studio, Prodigy, Argilla, and doccano have no "parent record" concept and default to per-item overlap by structural inertia. ARES uses per-(query, passage) human anchors for prediction-powered inference. RAGAS, TruLens, DeepEval, and Phoenix avoid the IAA question entirely by relying on LLM-as-judge.

## Strategic positioning

Pragmata is a deliberate niche - human-only annotation + IAA-overlap-based quality control - which differs from both:

- **Modern RAG eval mainstream** (LLM-as-judge dominant: RAGAS, TruLens, DeepEval, Phoenix)
- **Modern industry annotation services** (gold-standard insertion + per-annotator trust scores: Surge AI, Scale AI)

This positioning makes per-item partition (Option C) the right fit because it works with the naive Krippendorff bootstrap pragmata already uses, without requiring the cluster-aware analysis (G-theory, per-query α aggregation, multilevel κ) that the alternative would demand.

## Consequences

**Positive**

- Statistically honest α and CIs under the current naive bootstrap; no silent inflation from cluster correlation.
- Per-task right-sizing: operators can size each task's calibration set independently via the inheritable `calibration_fraction` and `calibration_max_records`.
- Matches Argilla's per-item record creation (each retrieval chunk is its own Argilla record).
- Matches modern RAG eval annotation precedent (D-MERIT, TripJudge, ARES).

**Negative**

- Manifest schema migration: legacy `calibration: bool` entries are read via a backward-compat `@model_validator(mode="before")` that expands the scalar to per-task dicts. Per-chunk retrieval calibration is **not reconstructible** from legacy entries; the migrator leaves `retrieval_chunk_calibration` empty so re-imports assign fresh per-chunk decisions for any chunks the new code encounters.
- The implicit per-`record_uuid` bundling that ADR-0010 assumed weakens - the same `record_uuid` can have some chunks in retrieval-calibration and others in production. The dataset-per-task structure itself is preserved; what changes is that calibration is a property of the annotation item, not the parent record.
- **Order-dependence under binding cap**: when the cap is binding across multiple imports, the resulting calibration set is a function of `(corpus, seed, import_order)`, not `(corpus, seed)` alone. This is a consequence of the manifest-lock invariant - once an entry is in calibration, a later tightened cap cannot demote it. Documented in `assign_partitions` docstring.

## Out of scope - future work

Tracked separately so they don't get lost:

- **Cluster-aware IAA** (G-theory, per-query α aggregation, cluster bootstrap, multilevel κ). Option C compensates by avoiding cluster correlation under the naive bootstrap; if pragmata later wants a per-query IAA narrative (which the academic literature supports under correct statistical machinery), this is the layer to upgrade.
- **Voorhees-style stratified per-(query, label-class) capping**: TREC's actual convention is not pure per-topic but stratified ("max 200 relevant + 200 non-relevant per topic"). Useful for heavily label-imbalanced corpora.
- **LLM-as-judge hybrid** (ARES-style PPI): strategic question outside this ADR.
- **Gold-standard / honey-pot QC** (Surge/Scale style): would complement the existing IAA-overlap QC.
- **Stratified sampling by difficulty / source / locale**: SOTA recommendation pragmata doesn't currently implement.
- **Annotator monopolisation prevention**: Argilla-imposed gap (no per-record overlap or per-user assignment in the native distribution mechanism).
- **Fix retrieval IAA pivot to compute per-query α aggregation** in addition to per-(query, chunk) α: the Commit 1 pivot bug fix corrects the keying; per-query rollups for operator-facing narrative are separate.

## References

- [ADR-0010](0010-annotation-multi-dataset-architecture.md) - modified in spirit (calibration as property of annotation item rather than property of `record_uuid`); the dataset-per-task structure is preserved.
- [PR #206 (`f1c6d9f`)](https://github.com/bertelsmannstift/pragmata/pull/206) - `Inherit` sentinel recipe for `production_min_submitted`, `calibration_min_submitted`, `locale`. The same pattern extends here to `calibration_fraction` and `calibration_max_records`.
- [Annotation Import Pipeline §Calibration partitioning](../design/annotation-import-pipeline.md#calibration-partitioning) - implementation-level description of the per-item bucketing algorithm, manifest schema, and order-dependence note.
- Faggioli et al. 2025, "Variations in Relevance Judgments and the Shelf Life of Test Collections" - [arXiv:2502.20937](https://arxiv.org/abs/2502.20937).
- Zhang et al. 2024, "D-MERIT: Evaluating D-MERIT of Partial-annotation on Information Retrieval" - [arXiv:2406.16048](https://arxiv.org/abs/2406.16048).
- Hofstätter et al., TripJudge: [arXiv:2208.06936](https://arxiv.org/abs/2208.06936).
- Saad-Falcon et al. 2023, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems" - [arXiv:2311.09476](https://arxiv.org/abs/2311.09476).
- Kish 1965, *Survey Sampling*, John Wiley & Sons - design effect formula.
