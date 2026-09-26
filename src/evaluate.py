"""Evaluate a trained classifier on a held-out Iris test set."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.paths import DEFAULT_MODEL_PATH
from src.preprocess import split_train_test


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Return accuracy and macro-averaged F1 for model predictions."""
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "macro_f1": float(f1_score(target, predictions, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to a model created by src.train.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for the data split.")
    parser.add_argument("--data-path", type=Path, help="Optional CSV to use instead of bundled Iris.")
    parser.add_argument("--metrics-path", type=Path, help="Optional path to write metrics as JSON.")
    args = parser.parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found at {args.model_path}; run python -m src.train first.")

    _, X_test, _, y_test = split_train_test(
        load_dataset(args.data_path), test_size=args.test_size, random_state=args.random_state
    )
    model = joblib.load(args.model_path)
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    if args.metrics_path:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
