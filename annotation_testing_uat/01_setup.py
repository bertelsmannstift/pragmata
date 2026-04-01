from pathlib import Path

import argilla as rg

from pragmata.annotation import ImportResult, SetupResult, UserSpec, import_records, setup

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"
UAT_DIR = Path(__file__).parent
SAMPLE_DATA = UAT_DIR / "sample_data.json"
CREDENTIALS_FILE = UAT_DIR / "credentials.txt"

USERS = [
    UserSpec(username="alice", role="annotator", workspaces=["retrieval", "grounding"]),
    UserSpec(username="bob", role="annotator", workspaces=["grounding", "generation"]),
    UserSpec(username="ops_admin", role="owner"),
]


def write_credentials(result: SetupResult) -> None:
    """Write generated credentials to a local file for dev convenience."""
    lines = [
        "# Auto-generated credentials (do not commit)",
        f"# Default admin: argilla / {'argilla.apikey' if API_KEY == 'argilla.apikey' else '(custom)'}",
        "",
    ]
    for spec in USERS:
        pw = result.generated_passwords.get(spec.username, spec.password or "(unknown)")
        lines.append(f"{spec.username}  {spec.role}  {pw}")
    CREDENTIALS_FILE.write_text("\n".join(lines) + "\n")
    print(f"\nCredentials written to {CREDENTIALS_FILE}")


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)

    setup_result: SetupResult = setup(client, users=USERS)

    print("\n=== Setup complete ===")
    print(f"Workspaces created:  {setup_result.created_workspaces}")
    print(f"Workspaces skipped:  {setup_result.skipped_workspaces}")
    print(f"Datasets created:    {setup_result.created_datasets}")
    print(f"Datasets skipped:    {setup_result.skipped_datasets}")
    print(f"Users created:       {setup_result.created_users}")
    print(f"Users skipped:       {setup_result.skipped_users}")

    write_credentials(setup_result)

    import_result: ImportResult = import_records(client, SAMPLE_DATA)

    print("\n=== Import complete ===")
    print(f"Total input records: {import_result.total_records}")
    print(f"Records per dataset: {import_result.dataset_counts}")

    if import_result.errors:
        print(f"\nValidation errors ({len(import_result.errors)}):")
        for err in import_result.errors:
            print(f"  index={err.index}: {err.detail}")
    else:
        print("No validation errors.")

    print("\nNext: Open http://localhost:6900, log in as an annotator, and submit some annotations.")
    print("Then: python annotation_testing_uat/03_export.py")


if __name__ == "__main__":
    main()
