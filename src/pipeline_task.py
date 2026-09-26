"""Train and evaluate Iris for a Kubeflow Pipelines component."""

import argparse
import json
from pathlib import Path

import joblib

from src.data_loader import load_iris_data
from src.evaluate import evaluate_model
from src.preprocess import split_train_test
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = split_train_test(load_iris_data())
    model = train_model(X_train, y_train, c=args.c)
    metrics = evaluate_model(model, X_test, y_test)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    kfp_metrics = {
        "metrics": [{"name": name, "numberValue": value} for name, value in metrics.items()]
    }
    args.metrics_output.write_text(json.dumps(kfp_metrics) + "\n", encoding="utf-8")
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
