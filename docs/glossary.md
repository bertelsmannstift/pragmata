# Glossary

- **Chunk** — The atomic unit returned by the retriever: the independently retrievable, rankable segment prior to any grouping by source document. Retrieval-level annotation operates at this level.

- **Context set** ($C_i$) — The full evidence shown to the model at inference time: the complete set of retrieved passages as injected into the prompt, concatenated as a single string with `[CTX_SEP]` separators.

- **Inter-annotator agreement (IAA)** — Measure of consistency between independent annotators (e.g., Cohen's kappa, Krippendorff's alpha).

- **Annotation interface** — Web-based UI where domain experts label query-response pairs (powered by Argilla).

- **Batch/Task Distribution** — Collection of query-response pairs assigned to annotators as a single unit of work.

- **Evaluation framework** — ML pipeline for training and inference on evaluation models (DeBERTa, SBERT, cross-encoders).

- **MAMA cycle** (Model-Annotate-Measure-Adjust) — Iterative annotation quality loop: load a batch with full annotator overlap, measure inter-annotator agreement (Krippendorff's Alpha), revise guidelines if below threshold, repeat until convergence.

