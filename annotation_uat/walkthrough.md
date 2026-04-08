```bash
pip install -e ".[annotation]"

docker info  # check docker is running

cp deploy/annotation/.env.dev.example deploy/annotation/.env

make docker-up

# one-time setup: creates workspaces + users (no datasets yet)
python annotation_uat/01_setup.py

# default "uat" dataset_id — creates datasets on import
python annotation_uat/01_setup.py

# multiple runs with different dataset IDs (same workspaces, separate datasets)
python annotation_uat/02_import.py run1
python annotation_uat/02_import.py run2

# annotate in browser: http://localhost:6900
# log in with generated passwords (saved in credentials.txt)
# NB owner account can also toggle dataset/workspace settings (as well as annotate)

# export a specific run
python annotation_uat/03_export.py run1
# NB idempotent — each time creates new full set with all annotations at that point

# scoped teardown: delete only run1's datasets, keep workspaces + users
python annotation_uat/04_teardown.py run1

# full teardown: delete all datasets + workspaces (no arg)
python annotation_uat/04_teardown.py

# optionally delete users too (not deleted by default on teardown)
python annotation_uat/05_users_delete.py alice

make docker-down
# removes all data
```
