# pRAGmata: Evidence-grounded evaluation for RAG systems

[![License](https://img.shields.io/github/license/bertelsmannstift/pragmata)](https://github.com/bertelsmannstift/pragmata/blob/main/LICENSE.md)
[![CI](https://img.shields.io/github/actions/workflow/status/bertelsmannstift/pragmata/ci.yml?label=ci)](https://github.com/bertelsmannstift/pragmata/actions)
![Status](https://img.shields.io/badge/status-WIP-orange)

**pRAGmata** (*πράγματα, Ancient Greek: roughly “matters of fact”*) is a Python framework for empirically evaluating retrieval-augmented generation (RAG) systems. It evaluates systems over annotated query–response examples with explicit, decomposed evaluation dimensions, rather than relying on opaque or heuristic scoring. The framework consists of three components:

- An AI workflow for controlled synthetic query generation and variation
- A web-based annotation interface for domain experts to label query–response pairs across retrieval, grounding, and generation aspects
- A label-based evaluation pipeline with model-assisted inference using transformer cross-encoders for scalable scoring

These components can be used independently or combined into an evaluation workflow:

*query generation* → *annotation* → *evaluation*

> [!NOTE]
The project is under active construction and the public API may change before the first stable release.
