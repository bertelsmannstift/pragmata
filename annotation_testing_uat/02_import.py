import argilla as rg

from pragmata.annotation import ImportResult, import_records

API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"
SAMPLE_DATA = "annotation_testing_uat/sample_data.json"


def main() -> None:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    result: ImportResult = import_records(client, SAMPLE_DATA, format="json")

    print("\n=== Import complete ===")
    print(f"Total input records: {result.total_records}")
    print(f"Records per dataset: {result.dataset_counts}")

    if result.errors:
        print(f"\nValidation errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  index={err.index}: {err.detail}")
    else:
        print("No validation errors.")

    print("\nNext: Open http://localhost:6900, log in as an annotator, and submit some annotations.")
    print("Then: python annotation_testing_uat/03_export.py")


if __name__ == "__main__":
    main()
