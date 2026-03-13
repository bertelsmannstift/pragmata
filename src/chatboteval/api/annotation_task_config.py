"""Hardcoded Argilla dataset definitions for the three annotation tasks.

These are Argilla rg.Settings objects — runtime task definitions consumed by
annotation_setup.py, NOT boundary schemas (those live in core/schemas/) or
configurable settings (those live in core/settings/). They encode the annotation
protocol (fields, questions, labels) and are hardcoded per ADR-0009: any change
requires a new ADR and major version bump.

Distribution (min_submitted) is intentionally omitted — it is an operational
setting controlled by AnnotationSetupSettings.min_submitted and applied at
dataset creation time in annotation_setup.py.
"""

from unittest.mock import MagicMock

import argilla as rg

from chatboteval.core.schemas.annotation_task import Task

# Argilla v2 requires a client connection to construct field/question objects.
# These are pure definitions — no server needed. Patch temporarily.
_needs_patch = rg.Argilla._default_client is None
if _needs_patch:
    rg.Argilla._default_client = MagicMock()

_COLLAPSIBLE_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<style>
  details {{
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 4px 8px;
    margin-top: 4px;
  }}
  summary {{
    color: #888;
    cursor: pointer;
    font-size: 0.9em;
    padding: 2px 0;
  }}
  #content {{
    white-space: pre-wrap;
    padding: 6px 0;
  }}
</style>
</head>
<body>
<details id="wrapper">
  <summary>{summary_text}</summary>
  <div id="content"></div>
</details>
<script>
  var field = record.fields["{field_name}"];
  document.getElementById("content").textContent = field || "";
  document.getElementById("wrapper").addEventListener("toggle", function () {{
    parent.postMessage({{ type: "resize" }}, "*");
  }});
</script>
</body>
</html>
"""


def _collapsible_field(name: str, title: str) -> rg.CustomField:
    return rg.CustomField(
        name=name,
        title=title,
        template=_COLLAPSIBLE_TEMPLATE.format(field_name=name, summary_text=title),
        advanced_mode=True,
        required=True,
    )


# Distribution is omitted — applied from AnnotationSetupSettings at creation time.

TASK1_RETRIEVAL_SETTINGS = rg.Settings(
    fields=[
        rg.TextField(name="query", title="Query", required=True),
        rg.TextField(name="chunk", title="Chunk", required=True),
        _collapsible_field("generated_answer", "Generated answer"),
    ],
    questions=[
        rg.LabelQuestion(
            name="topically_relevant",
            title="Does this passage contain information that is substantively relevant to the query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="evidence_sufficient",
            title="Does this passage provide sufficient evidence to support answering the query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="misleading",
            title="Could this passage plausibly lead to an incorrect or distorted answer?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.TextQuestion(name="notes", title="Notes (optional)", required=False),
    ],
    guidelines="Task 1 — Retrieval. TODO: Revisit after first annotation iteration.",
)

TASK2_GROUNDING_SETTINGS = rg.Settings(
    fields=[
        rg.TextField(name="answer", title="Answer", required=True),
        rg.TextField(name="context_set", title="Context set", required=True),
        _collapsible_field("query", "Query"),
    ],
    questions=[
        rg.LabelQuestion(
            name="support_present",
            title="Is at least one claim in the answer supported by the provided context?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="unsupported_claim_present",
            title="Does the answer contain claims not supported by the provided context?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="contradicted_claim_present",
            title="Does the provided context contradict any claim in the answer?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="source_cited",
            title="Does the answer contain a citation marker?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="fabricated_source",
            title="Does the answer cite a source not present in the retrieved context?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.TextQuestion(name="notes", title="Notes (optional)", required=False),
    ],
    guidelines="Task 2 — Grounding. TODO: Revisit after first annotation iteration.",
)

TASK3_GENERATION_SETTINGS = rg.Settings(
    fields=[
        rg.TextField(name="query", title="Query", required=True),
        rg.TextField(name="answer", title="Answer", required=True),
        _collapsible_field("context_set", "Context set"),
    ],
    questions=[
        rg.LabelQuestion(
            name="proper_action",
            title="Did the system choose the appropriate action for this query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="response_on_topic",
            title="Does the response substantively address the user's query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="helpful",
            title="Would this response enable a typical user to make progress on their task?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="incomplete",
            title="Does the response fail to cover required parts of the query?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.LabelQuestion(
            name="unsafe_content",
            title="Does the response contain unsafe or policy-violating content?",
            labels=["yes", "no"],
            required=True,
        ),
        rg.TextQuestion(name="notes", title="Notes (optional)", required=False),
    ],
    guidelines="Task 3 — Generation. TODO: Revisit after first annotation iteration.",
)

TASK_SETTINGS: dict[Task, rg.Settings] = {
    Task.RETRIEVAL: TASK1_RETRIEVAL_SETTINGS,
    Task.GROUNDING: TASK2_GROUNDING_SETTINGS,
    Task.GENERATION: TASK3_GENERATION_SETTINGS,
}

DATASET_NAMES: dict[Task, str] = {
    Task.RETRIEVAL: "task1_retrieval",
    Task.GROUNDING: "task2_grounding",
    Task.GENERATION: "task3_generation",
}

if _needs_patch:
    rg.Argilla._default_client = None
del _needs_patch
