import sys

import argilla as rg

from pragmata.annotation import teardown

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"
DATASET_ID = sys.argv[1] if len(sys.argv) > 1 else ""


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    teardown(client, dataset_id=DATASET_ID)

    if DATASET_ID:
        print(f"\n=== Teardown complete (dataset_id={DATASET_ID}) ===")
        print("Matching datasets deleted. Workspaces and user accounts preserved.")
    else:
        print("\n=== Full teardown complete ===")
        print("All datasets and workspaces deleted. User accounts preserved.")


if __name__ == "__main__":
    main()
