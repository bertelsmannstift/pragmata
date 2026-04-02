pip install -e ".[annotation]"

docker info (check docker is rurnning)

cp deploy/annotation/.env.dev.example deploy/annotation/.env

make docker-up

python3 annotation_uat/01_setup.py

python3 annotation_uat/02_import.py


# annotate in browser: http://localhost:6900 (log in to diff user accs w/ generated passwords (saved in credentials.txt))
# NB owner account can also toggle dataset/workspace settings etc (as well as annotate)

python3 annotation_uat/03_export.py 
# NB idempotent -> each time creates new full set each time with all annotations by all users at that point

python3 annotation_uat/04_teardown.py

# optionally delete users too (not deleted by default on teardown)
python3 annotation_uat/04_teardown.py

make docker-down
# removes all data

3. Create workspaces, datasets, and users
Run this once to set up the Argilla environment:
import argilla as rg
from pragmata.annotation import setup, UserSpec

client = rg.Argilla(api_url="http://localhost:6900", api_key="argilla.apikey")

result = setup(
    client,
    users=[
        UserSpec(username="alice", role="annotator", workspaces=["retrieval"]),
        UserSpec(username="bob",   role="annotator", workspaces=["grounding", "generation"]),
    ]
)

print(result.created_workspaces)   # ['retrieval', 'grounding', 'generation']
print(result.created_datasets)     # ['task_retrieval', 'task_grounding', 'task_generation']
print(result.generated_passwords)  # {'alice': 'xxx', 'bob': 'yyy'}  ← save these

setup() is idempotent - safe to call again if something fails partway through.

4. Import records
from pragmata.annotation import import_records

sample_data = [
    {
        "query": "What are the main sources of renewable energy?",
        "answer": "The main sources are solar, wind, and hydroelectric power.",
        "context_set": "ctx-001",
        "language": "en",
        "chunks": [
            {"chunk_id": "c1", "doc_id": "doc1", "chunk_rank": 1,
             "text": "Solar energy is captured via photovoltaic panels..."},
            {"chunk_id": "c2", "doc_id": "doc1", "chunk_rank": 2,
             "text": "Wind turbines convert kinetic energy from wind..."},
        ]
    },
    {
        "query": "How does hydroelectric power work?",
        "answer": "It uses the flow of water to spin turbines and generate electricity.",
        "context_set": "ctx-002",
        "language": "en",
        "chunks": [
            {"chunk_id": "c3", "doc_id": "doc2", "chunk_rank": 1,
             "text": "Dams store water at height; releasing it drives generators..."},
        ]
    }
]

result = import_records(client, sample_data)
print(result.total_records)    # 2
print(result.dataset_counts)   # {'task_retrieval': 3, 'task_grounding': 2, 'task_generation': 2}
print(result.errors)           # [] — any validation failures appear here

Accepts a list of dicts (as above), or a file path to a JSON / JSONL / CSV file.
Record IDs are content-hashed, so re-importing the same data is a no-op.

5. Annotate in the UI
Go to http://localhost:6900
Log in as alice (password from step 3)
Select workspace retrieval → dataset task_retrieval
For each record, answer the three questions and click Submit
Retrieval questions:
Does this passage contain information that is substantively relevant to the query?
Does this passage provide sufficient evidence to support answering the query?
Could this passage plausibly lead to an incorrect or distorted answer?
Grounding questions (in workspace grounding):
Is at least one claim in the answer supported by the provided context?
Does the answer contain claims not supported by the provided context?
Does the provided context contradict any claim in the answer?
Does the answer contain a citation marker?
Does the answer cite a source not present in the retrieved context?
Generation questions (in workspace generation):
Did the system choose the appropriate action for this query?
Does the response substantively address the user's query?
Would this response enable a typical user to make progress on their task?
Does the response fail to cover required parts of the query?
Does the response contain unsafe or policy-violating content?
All questions are binary (yes/no). There's an optional Notes field on each.

6. Export results
Note: The public export() API is not yet wired up. Use the internal fetcher directly for now:
from pragmata.core.annotation.export_fetcher import build_user_lookup, fetch_task
from pragmata.core.annotation.export_helpers import write_export_csv
from pragmata.core.schemas.annotation_task import Task
from pragmata.core.settings.annotation_settings import AnnotationSettings
from pathlib import Path

settings = AnnotationSettings()
user_lookup = build_user_lookup(client)

for task in Task:
    rows = fetch_task(client, settings, task, user_lookup)
    out = Path(f"{task.value}.csv")
    write_export_csv(rows, out, task)
    print(f"{task.value}: {len(rows)} rows → {out}")

This produces retrieval.csv, grounding.csv, generation.csv. Each row includes all label columns plus constraint_violated (true/false) and constraint_details flagging logical inconsistencies (e.g. evidence_sufficient=yes but topically_relevant=no). Records are linked across the three files via record_uuid.

Things to look for / UAT checklist
 All three datasets appear in the Argilla UI after setup()
 Each dataset shows the correct fields and questions
 Import fan-out is correct: a record with N chunks creates N retrieval + 1 grounding + 1 generation records
 Annotator accounts can only see their assigned workspaces
 Submitting annotations in the UI marks records as "submitted"
 Export CSVs contain the correct columns and the submitted values
 Constraint violations are correctly flagged (e.g. submit evidence_sufficient=yes, topically_relevant=no and verify constraint_violated=true in the CSV)
 Re-running import_records() with the same data produces no duplicates

Workspace topology (default)
argilla server
├── workspace: retrieval   → dataset: task_retrieval
├── workspace: grounding   → dataset: task_grounding
└── workspace: generation  → dataset: task_generation

You can optionally namespace everything with a prefix (e.g. for parallel test runs):
setup(client, workspace_prefix="uat_", users=[...])
import_records(client, data, workspace_prefix="uat_")
# teardown when done:
teardown(client, workspace_prefix="uat_")


Questions? The design docs are in docs/design/annotation-*.md if you need more detail on the schema or export format.

A few things worth flagging before you share this:
Export is not yet a public API - the export step uses internal core/ modules directly. If that's landing in an open PR, let me know and I can update the snippet.
Guidelines text - the Argilla dataset guidelines currently say "TODO: Revisit after first annotation iteration" (argilla_task_definitions.py:83). Fine for UAT but worth flagging to the team.
Sharing format - given this references internal API details, sharing as a message/Notion/internal doc makes more sense than a PR to the main repo.


----

owner role: settings of datasets > can play around 