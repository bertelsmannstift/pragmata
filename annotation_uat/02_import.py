import sys

import argilla as rg

from pragmata.annotation import ImportResult, import_records

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"
DATASET_ID = sys.argv[1] if len(sys.argv) > 1 else ""
SAMPLE_DATA = "annotation_uat/sample_data.json"


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    result: ImportResult = import_records(client, SAMPLE_DATA, format="json", dataset_id=DATASET_ID)

    print(f"\n=== Import complete (dataset_id={DATASET_ID}) ===")
    print(f"Total input records: {result.total_records}")
    print(f"Records per dataset: {result.dataset_counts}")

    if result.errors:
        print(f"\nValidation errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  index={err.index}: {err.detail}")
    else:
        print("No validation errors.")


if __name__ == "__main__":
    main()
