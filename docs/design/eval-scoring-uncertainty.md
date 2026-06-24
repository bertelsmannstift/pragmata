# Eval scoring layer: translate labels into corpus metrics with uncertainty

## Summary

Builds the eval scoring stage 
- turns per-row labels (either manually annotated or evaluator-predicted) into corpus metrics defined in the [metrics taxonomy](../methodology/metrics-taxonomy.md) 
- attach a confidence interval to every reported metric
- expose it via a API and a new `pragmata eval {__}` CLI, 
- reuses the existing IAA stats machinery. 

## Background - current state

```
 annotation export (CSV, per-row labels)
   │  import_eval_*_frame  (core/eval/imports.py)        DONE
   ▼
 build_tlmtc_frame + consolidate  (core/eval/transforms.py)  DONE
   ▼
 [tlmtc train / predict  (EXTERNAL, ADR-0002)]              → per-row predicted labels
   ▼
 ╔════════════════════════════════════════╗
 ║  SCORING  (per-row labels → metrics)   ║   MISSING - this issue
 ╚════════════════════════════════════════╝
   ▼
 *_scores.json  (core/schemas/eval_output.py)           NEEDS UPDATING CUFRENTLY:
                                                              shapes only, bare 0–1 floats,
                                                              no uncertainty, no compute code
```

- report shapes exist (`eval_output.py`) but each metric is a bare `Rate = float[0,1]` and nothing computes them.
- no eval CLI (`cli/commands/` has only `annotation.py`, `querygen.py`).
- IAA already reports bootstrap CIs on Krippendorff's alpha - `core/annotation/iaa.py:bootstrap_alpha` (seeded, computes percentiles, configurable `n_resamples`), surfaced through`iaa_report.py` (`ci_lower/ci_upper/ci_level/ n_bootstrap_resamples`), driven by `api/annotation_iaa.py:compute_iaa` -> `iaa_runner.py:run_iaa`.

## Goals

- compute all metrics (retrieval 6, grounding 5, generation 5) from per-row labels.
- attach CIs (boostrap sampling uncertainty).
- public API (`score_eval`??)
- CLI (`pragmata eval score`??)
- shared `core/stats.py` helpers, w/ IAA refactored to reuse 

## Non-goals (named, deferred)

- evaluator label noise: cross-encoder is imperfect; propagating its classification error is only relevant on the predict-from-model path (human-label scoring has none) and is a materially larger design/owned by tlmtc? Deferred.
- run-to-run comparison / significance: ee emit per-run CIs only; comparing two runs rigorously needs a *paired* difference-CI (overlap of two independent CIs is **not** a
  significance test). Downstream / out of scope.
- bias-corrected-&-accelerated intervals: ercentile for v1; BCa is the upgrade only if coverage proves poor.

## Design

### Grain and resampling unit

```
 corpus metric (reported) <- mean over queries <- per-query value  <─ per-row labels
```

- **Retrieval:** per-chunk labels -> per-query `@k` scalar -> mean over queries
- **Grounding / generation:** one row per query -> corpus metric is a proportion over queries

- unit everything averages over is **query** 
- query is also the resampling unit (resample whole queries w/ replacement, keeping each query's chunks attached (cluster
bootstrap). Resampling chunks would break the `/k` denominator and rank order and understate variance (= the hierarchical caveat in ADR-0012))

### Uncertainty method - per metric

| Family | Metrics | Method | n |
|---|---|---|---|
| Proportion (binary per query) | all grounding (5), all generation (5), `sufficiency_hit_at_k` | **Wilson** closed-form | #queries |
| Conditional proportion | `conditional_fabrication_rate` | **Wilson** on the cited subset | #cited queries |
| Continuous mean over queries | `ndcg_at_k`, `mean_reciprocal_rank_at_k`, `topical_precision_at_k`, `sufficiency_rate_at_k`, `misleading_context_rate_at_k` | **query-level percentile bootstrap** | #queries |

NB: ^^^`method` is recorded per metric. Wilson chosen for proportions because it is well-behaved at small n (cf wioth bootstrapping a proportion degenerates); bootstrap for the
continuous metrics because they have no clean closed form.

### New module: `core/stats.py`

Flat utility module (matches `core/atomic_io.py`, `csv_io.py`, `types.py`).

```python
def wilson_interval(successes: int, n: int, *, ci: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lower, upper)."""

def percentile_bootstrap(
    n_units: int,
    statistic: Callable[[NDArray[np.intp]], float],
    *, n_resamples: int = 1000, ci: float = 0.95, seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Resamples unit indices [0, n_units) with replacement,
    applies `statistic` to each resample, drops NaN replicates (as IAA does)."""
```


### Output contract - `core/schemas/eval_output.py`

Introduce a nested per-metric model (JSON artifact; named fields
kept because the metric set is a fixed taxonomy):

```python
class MetricScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    point: Rate
    ci_lower: Rate
    ci_upper: Rate
    method: Literal["wilson", "bootstrap"]
    n: PositiveInt            # effective denominator (queries; cited subset for conditional)
```

Each metric field changes from `Rate` -> `MetricScore` 

### Compute + orchestration

- **`core/eval/metrics.py`** (new) - pure, deterministic per-query metric functions
  matching the taxonomy formulas, plus corpus aggregation.
- **`core/eval/scoring.py`** (new) - orchestration, mirrors `iaa_runner.py:run_iaa`: validate -> group by query -> compute point + CI per metric (Wilson or bootstrap) -> assemble the task report -> write JSON.

```python
def run_scoring(
    frame: pd.DataFrame, *, task: Task,
    top_k: int | None = None, n_resamples: int = 1000, ci: float = 0.95, seed: int | None = None,
) -> RetrievalScoreReport | GroundingScoreReport | GenerationScoreReport: ...
```
^^^ pure compute - takes data itself


### Public API - `api/eval_score.py` (new), mirrors `compute_iaa`


the i/o wrapper:

```python
def score_eval(
    *, base_dir: str | Path | Unset = UNSET,
    score_id: str | None = None,
    labeled_input_path: str | Path | None = None,   
    prediction_run_id: str | None = None,         
    task: Task,
    top_k: int | None = None,
    n_resamples: int = 1000, ci: float = 0.95, seed: int | None = None,
    config_path: str | Path | Unset = UNSET,
) -> RetrievalScoreReport | GroundingScoreReport | GenerationScoreReport:
    """Resolve EvalScoreSettings + score paths, read the labeled/predicted frame,
run_scoring inside error_log, write *_scores.json, return the report."""
```

Reuses `EvalScoreSettings` (`settings/eval_settings.py`), `resolve_eval_score_paths`
(`paths/eval_paths.py`), `validate_eval_score_frame` (`eval_input.py`), `error_log`.

### CLI - `cli/commands/eval.py` (new) + wire into `cli/app.py`

`eval_app = typer.Typer()`; `app.add_typer(eval_app, name="eval")` (mirrors `annotation`).

```
pragmata eval score --task retrieval \
  [--labeled-input-path P | --prediction-run-id ID] \
  [--score-id ID] [--base-dir DIR] [--top-k K] \
  [--n-resamples 1000] [--ci 0.95] [--seed N] [--config FILE]
```

### TODO

Input-contract dependency (decision needed):

retrieval scoring contract currently lacks the columns the `@k` metrics need. `RETRIEVAL_*_SCHEMA` (`eval_input.py`) is `(query, chunk, labels)`; `strict=False` = lets extras pass but nothing is required/ordered. To score retrieval we need, per row:

- **`record_uuid`** - group chunks into queries (the per-query unit).
- **`rank`** (1..K) - required by NDCG@K and MRR@K and any top-K truncation.
- **`chunk_id`** - stable identity within a query (already a dup-key in `transforms.py`).

So 
- extend the retrieval scoring schema to 
  1. require `record_uuid`, `rank`, `chunk_id` (w/o `rank`,NDCG/MRR cannot be computed)
  2.  define how `K` is set (explicit `--top-k` vs inferred per-query chunkcount)???.
- grounding/generation need `record_uuid` 
