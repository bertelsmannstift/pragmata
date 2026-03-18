# 0002 Evaluation Approach

Status: Accepted


## Decision

This project evaluates RAG chatbots by scoring captured query–examples (primary evaluation unit, [see Metrics Taxonomy](../methodology/metrics-taxonomy.md)) using supervised evaluator models trained on human annotations.

The following architectural constraints apply:

- **Reference-based evaluation only**
  - Evaluation requires a human-annotated dataset.
  - Automated scoring is performed by fine-tuning and applying cross-encoder transformer models for multilabel classification.
  - Out of scope for v.1.*: LLM-as-judge, zero-shot evaluators, and other reference-free scoring approaches.
- **Offline batch evaluation**
  - The framework evaluates captured examples, retrospectively, in batch.
  - Out of scope: Real-time chatbot integration
- **Failure-mode decomposition**
  - Evaluation is decomposed into retrieval, grounding, and generation
- **Pipeline delegation**
  - Training and inference are executed via `tlmtc` (Transfer Learning for Multi-Label Text Classification) as the canonical transfer-learning pipeline.


## Rationale

- **Reference-based evaluation yields reproducible, auditable, and reliable scores** 
  - LLM-as-judge / zero-shot approaches depend on natural-language prompting and a served model/version, making scoring sensitive to prompt wording and model updates. Because they rely on generative decoding, scoring may vary across runs and incurs marginal per-example generation-token costs. This complicates longitudinal comparability unless the exact prompt, model version, and decoding configuration are strictly controlled.
  - Human-annotated labels provide an explicit ground truth for auditability and for validating the evaluator as a measurement instrument.
  - Supervised evaluators amortize cost into efficient fine-tuning and enable fast, cheap batch inference for repeatable runs at scale.
  - Cross-encoders learn relational scoring functions over (query, retrieved context, answer)—e.g., relevance and entailment/contradiction patterns. These signals often transfer within a domain and across closely related datasets, enabling consistent scoring of system variants (chunking/retrieval/reranking) without changing the evaluator. Significant domain shift may require incremental re-annotation and fine-tuning, which we treat as a controlled update to the pinned evaluator.
  - Multilabel classification matches the construct: within each metric family, evaluation dimensions can co-occur and be logically interdependent.
- **Offline batch evaluation supports reproducibility and keeps the system simple**
  - Human annotation is inherently asynchronous and produces labeled datasets in discrete snapshots rather than streams.
  - Batch runs enable controlled experimentation and make results comparable across runs and over time.
  - Avoids coupling evaluation to production systems and eliminates the need for real-time ingestion.
- **Failure-mode decomposition improves construct validity and actionability**
  - Separating retrieval, grounding, and generation keeps each task’s unit of annotation conceptually coherent and avoids NA-heavy schemas that dilute rater attention and agreement.
  - A single “overall quality” judgment conflates distinct error types, making scores harder to interpret and less diagnostically useful.
  - Decomposed outputs map directly to levers in a RAG system, supporting actionable debugging and regression triage.
- **Pipeline delegation enforces separation of concerns and reduces maintenance risk**
  - `tlmtc` encapsulates the transfer-learning lifecycle, allowing `pRAGmata` to focus on evaluation methodology, schema contracts, and reporting.
  - Reusing a dedicated transfer-learning pipeline avoids reimplementing complex and error-prone ML infrastructure, reducing technical debt.


## Consequences

- Evaluation requires an annotation workflow and a labeled dataset snapshot; the framework cannot operate in a reference-free mode.
- The scoring backend is constrained to supervised cross-encoder multilabel classification; metric computation assumes model outputs are aligned with the annotation protocol.
- Evaluator generalization is empirical. When applying the evaluator to substantially new domains, corpora, or query distributions, a labeled validation slice must be collected to assess performance. If degradation is observed, additional annotation and re-training are required.
- The annotation protocol must be multilabel-compatible (non–mutually-exclusive binary dimensions); continuous or ranking-style judgments are out of scope.
- `tlmtc` owns: data splitting and pre-processing, HPO, fine-tuning, inference, and model validation
