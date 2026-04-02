import sys

import argilla as rg

from pragmata.annotation import teardown

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "uat"


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    teardown(client, workspace_prefix=PREFIX)
    print(f"\n=== Teardown complete (prefix={PREFIX}) ===")
    print("Datasets and workspaces deleted. User accounts are preserved.")


if __name__ == "__main__":
    main()
