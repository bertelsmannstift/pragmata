import argilla as rg

from pragmata.annotation import teardown

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    teardown(client)
    print("\n=== Teardown complete ===")
    print("Datasets and workspaces deleted. User accounts are preserved.")


if __name__ == "__main__":
    main()
