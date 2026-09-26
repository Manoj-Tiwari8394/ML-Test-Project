"""Create a local Iris CSV and prepare it for the DVC pipeline."""

import argparse
from pathlib import Path

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, load_iris_data
from src.paths import PROJECT_ROOT

DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "iris.csv"


def create_raw_dataset(output_path: Path = DEFAULT_RAW_PATH) -> None:
    """Write the bundled Iris data to a CSV file for local DVC tracking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_iris_data().to_csv(output_path, index=False)
    print(f"Created {output_path}")


def prepare_dataset(input_path: Path, output_path: Path) -> None:
    """Read, validate, and write the expected columns in a consistent order."""
    data = load_dataset(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.loc[:, (*FEATURE_COLUMNS, TARGET_COLUMN)].to_csv(output_path, index=False)
    print(f"Prepared {len(data)} rows in {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create-raw",
        action="store_true",
        help="Export the bundled dataset to the raw data path instead of preparing a CSV.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_PATH)
    args = parser.parse_args()

    if args.create_raw:
        create_raw_dataset(args.raw_output)
    else:
        prepare_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
