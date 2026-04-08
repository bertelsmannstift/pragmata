```bash
pip install -e ".[annotation]"

docker info  # check docker is running

cp deploy/annotation/.env.dev.example deploy/annotation/.env

make docker-up

# ── Setup (one-time per environment) ────────────────────────────────
# creates workspaces + users only — no datasets yet
python annotation_uat/01_setup.py

# ── Import ──────────────────────────────────────────────────────────
# no dataset_id — bare dataset names (retrieval, grounding, generation)
python annotation_uat/02_import.py

# with dataset_id — suffixed datasets (retrieval_run1, grounding_run1, ...)
python annotation_uat/02_import.py run1

# multiple independent runs in the same workspaces, same users
python annotation_uat/02_import.py run2

# re-import into same run — appends / upserts, does not duplicate
python annotation_uat/02_import.py run1

# ── Annotate ────────────────────────────────────────────────────────
# open http://localhost:6900 and log in with generated passwords (credentials.txt)
# NB: owner account can toggle dataset/workspace settings as well as annotate

# ── Export ──────────────────────────────────────────────────────────
# export a specific run (reads retrieval_run1, grounding_run1, generation_run1)
python annotation_uat/03_export.py run1

# export bare datasets (no dataset_id)
python annotation_uat/03_export.py

# NB idempotent — re-running overwrites the same export directory

# ── Teardown ────────────────────────────────────────────────────────
# scoped: delete only run1's datasets, workspaces + users preserved
python annotation_uat/04_teardown.py run1

# full: delete all datasets + workspaces (users still preserved)
python annotation_uat/04_teardown.py

# optionally delete individual users (not touched by teardown)
python annotation_uat/05_users_delete.py alice

make docker-down  # stops stack and removes all volumes
```
