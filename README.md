# pRAGmata: Evidence-grounded evaluation for RAG systems

[![License](https://img.shields.io/github/license/bertelsmannstift/pragmata)](https://github.com/bertelsmannstift/pragmata/blob/main/LICENSE.md)
[![CI](https://img.shields.io/github/actions/workflow/status/bertelsmannstift/pragmata/ci.yml?label=ci)](https://github.com/bertelsmannstift/pragmata/actions)
![Status](https://img.shields.io/badge/status-WIP-orange)

**pRAGmata** (*πράγματα, Ancient Greek: roughly “matters of fact”*) is a Python framework for empirically evaluating retrieval-augmented generation (RAG) systems. Grounded in real evidence rather than impressionistic scoring, it combines three components:

- A structured AI workflow for controlled synthetic query generation and variation
- A web-based annotation interface for domain experts to label query–response pairs across retrieval, grounding, and generation aspects
- An automated evaluation pipeline based on transformer cross-encoder models for efficient, repeatable evaluation at scale

> [!NOTE]
The project is under active construction and the public API may change before the first stable release.
