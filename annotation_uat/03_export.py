import sys

import argilla as rg

from pragmata.annotation import ExportResult, export_annotations

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "uat"


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    result: ExportResult = export_annotations(client, base_dir="annotation_uat", workspace_prefix=PREFIX)

    print(f"\n=== Export complete (prefix={PREFIX}) ===")
    print(f"Row counts: {dict(result.row_counts)}")

    if result.files:
        print("\nCSV files written:")
        for task, path in result.files.items():
            print(f"  {task.value}: {path}")
    else:
        print("No files written (no submitted annotations found).")

    if result.constraint_summary:
        print(f"\nConstraint violations: {result.constraint_summary}")
    else:
        print("No constraint violations.")


if __name__ == "__main__":
    main()
