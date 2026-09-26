"""Train the lightweight deployment model without MLflow or DVC."""

import argparse
from pathlib import Path

import joblib

from src.data_loader import load_iris_data
from src.paths import DEFAULT_MODEL_PATH
from src.preprocess import split_train_test
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    X_train, _, y_train, _ = split_train_test(load_iris_data())
    model = train_model(X_train, y_train, c=args.c)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved deployment model to {args.output}")


if __name__ == "__main__":
    main()
