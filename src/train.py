"""Train and save a small logistic-regression Iris classifier."""

import argparse
import math
import os
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
from src.run_metadata import collect_run_metadata


def log_mlflow_run(
    model,
    features: pd.DataFrame,
    params: dict[str, int | float | str],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    registered_model_name: str | None = None,
) -> tuple[str, str | None]:
    """Log one training run and optionally register its model."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = str(params.get("experiment_name", DEFAULT_EXPERIMENT_NAME))
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
        if artifact_root and artifact_root.startswith("file://"):
            Path(artifact_root.removeprefix("file://")).mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=(
                f"{artifact_root.rstrip('/')}/{experiment_name}"
                if artifact_root
                else None
            ),
        )
    else:
        experiment_id = experiment.experiment_id

    run_id = ""
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_params({key: value for key, value in params.items() if key != "experiment_name"})
        mlflow.log_metrics(metrics)
        run_tags = dict(tags or {})
        run_tags.update(collect_run_metadata())
        if run_tags:
            mlflow.set_tags(run_tags)
        signature = infer_signature(features, model.predict(features))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=features.head(3),
        )

    model_version = None
    if registered_model_name:
        version = mlflow.register_model(
            f"runs:/{run_id}/model",
            registered_model_name,
            await_registration_for=120,
        )
        model_version = version.version
        client.set_model_version_tag(
            registered_model_name,
            model_version,
            "git.commit",
            (tags or {}).get("git.commit", os.getenv("GITHUB_SHA", "local")),
        )
    return run_id, model_version


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
    parser.add_argument(
        "--register-model-name",
        help="Optional MLflow Registry model name; requires a database-backed tracking store.",
    )
    args = parser.parse_args()

    data = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = split_train_test(
        data, test_size=args.test_size, random_state=args.random_state
    )
    model = train_model(X_train, y_train, c=args.c)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(model, X_test, y_test)

    run_id, model_version = log_mlflow_run(
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
        registered_model_name=args.register_model_name,
    )
    joblib.dump(model, args.model_path)

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Saved model to: {args.model_path}")
    print(f"MLflow run ID: {run_id}")
    print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_DIR.as_uri())}")
    if model_version:
        print(f"Registered model: {args.register_model_name} version {model_version}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
