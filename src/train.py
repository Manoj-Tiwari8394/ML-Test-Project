"""Train and save a small logistic-regression Iris classifier."""

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.evaluate import evaluate_model
from src.paths import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MODEL_PATH,
    MLFLOW_TRACKING_DIR,
)
from src.preprocess import split_train_test


def log_mlflow_run(
    model,
    features: pd.DataFrame,
    params: dict[str, int | float | str],
    metrics: dict[str, float],
) -> str:
    """Log one training run and its model to the local MLflow store."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    from src.paths import DEFAULT_EXPERIMENT_NAME, MLFLOW_TRACKING_DIR

    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment(params.get("experiment_name", DEFAULT_EXPERIMENT_NAME))
    with mlflow.start_run() as run:
        mlflow.log_params({key: value for key, value in params.items() if key != "experiment_name"})
        mlflow.log_metrics(metrics)
        signature = infer_signature(features, model.predict(features))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=features.head(3),
        )
    return run.info.run_id


def build_model(c: float = 1.0) -> Pipeline:
    """Build scaling and logistic regression as one reusable model pipeline."""
    if not math.isfinite(c) or c <= 0:
        raise ValueError("C must be a finite number greater than zero.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


def train_model(
    features: pd.DataFrame, target: pd.Series, c: float = 1.0
) -> Pipeline:
    """Fit the model on training features and labels."""
    model = build_model(c=c)
    model.fit(features, target)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for the data split.")
    parser.add_argument("--data-path", type=Path, help="Optional CSV to use instead of bundled Iris.")
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment for this training run.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model.",
    )
    args = parser.parse_args()

    data = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = split_train_test(
        data, test_size=args.test_size, random_state=args.random_state
    )
    model = train_model(X_train, y_train, c=args.c)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(model, X_test, y_test)

    run_id = log_mlflow_run(
        model,
        X_train,
        {
            "c": args.c,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "max_iter": 1000,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "experiment_name": args.experiment_name,
        },
        metrics,
    )
    joblib.dump(model, args.model_path)

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Saved model to: {args.model_path}")
    print(f"MLflow run ID: {run_id}")
    print(f"MLflow tracking directory: {MLFLOW_TRACKING_DIR}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
