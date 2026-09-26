"""Predict an Iris species from four flower measurements."""

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import FEATURE_COLUMNS
from src.paths import DEFAULT_MODEL_PATH


def predict_species(model: Pipeline, measurements: dict[str, float]) -> str:
    """Return the predicted species for one named set of measurements."""
    missing_features = [name for name in FEATURE_COLUMNS if name not in measurements]
    if missing_features:
        raise ValueError(f"Missing feature measurements: {missing_features}")

    values = [measurements[name] for name in FEATURE_COLUMNS]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("Feature measurements must be numbers.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Feature measurements must be finite numbers.")

    row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    return str(model.predict(row)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sepal-length", type=float, required=True, help="Sepal length in cm.")
    parser.add_argument("--sepal-width", type=float, required=True, help="Sepal width in cm.")
    parser.add_argument("--petal-length", type=float, required=True, help="Petal length in cm.")
    parser.add_argument("--petal-width", type=float, required=True, help="Petal width in cm.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found at {args.model_path}; run python -m src.train first.")

    model = joblib.load(args.model_path)
    result = predict_species(
        model,
        {
            FEATURE_COLUMNS[0]: args.sepal_length,
            FEATURE_COLUMNS[1]: args.sepal_width,
            FEATURE_COLUMNS[2]: args.petal_length,
            FEATURE_COLUMNS[3]: args.petal_width,
        },
    )
    print(f"Predicted species: {result}")


if __name__ == "__main__":
    main()
