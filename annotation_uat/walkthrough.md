pip install -e ".[annotation]"

docker info (check docker is rurnning)

cp deploy/annotation/.env.dev.example deploy/annotation/.env

make docker-up

python3 annotation_uat/01_setup.py

# fefault uat prefix
python annotation_uat/01_setup.py

# parallel runs w/ different prefixes
python annotation_uat/01_setup.py uat_1
python annotation_uat/01_setup.py uat_2

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


# Questions to address

- context: basic setup in argilla is that workspace has multi datasets, users assigned to workspaces (see all datasets in that workspace). We have workspace_prefix currently prefixing both workspace and dataset names (this ws by design so we can run completely isolated environments). But i think multiple runs into the same workspace makes more sense. e.g.
    ```
    Workspace: retrieval (permanent, users assigned here)
    Dataset run 1: uat_1_retrieval (inside retrieval workspace)
    Dataset run 2: uat_2_retrieval (inside retrieval workspace)
    ```
= users stay assigned to their workspaces and we can import fresh datasets without re-provisioning users?