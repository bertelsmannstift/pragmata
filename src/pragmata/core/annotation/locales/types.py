"""Shared types for locale catalogs."""

from typing import Literal

from pragmata.core.schemas.annotation_task import Task

CatalogKind = Literal["field", "question", "guidelines", "label", "widget"]
# Naming convention for the ``name`` slot:
#   - ``"field"`` / ``"question"`` / ``"guidelines"``: the field/question
#     ``name=`` identifier, or ``""`` for guidelines.
#   - ``"label"``: ``"<question_name>.<label_value>"`` (e.g.
#     ``"topically_relevant.yes"``). Label *values* are machine identifiers
#     and never translated — only the display text varies by locale.
#   - ``"widget"``: free-form key for strings inside our injected HTML
#     widgets (e.g. ``discard_flow.html``). Shared across tasks: keyed by
#     ``Task.RETRIEVAL`` with ``("widget", "discard.panel_summary")`` style,
#     and identical for grounding/generation per locale.
type CatalogKey = tuple[Task, CatalogKind, str]
type Catalog = dict[CatalogKey, str]
