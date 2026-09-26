"""Load and validate the Iris dataset bundled with scikit-learn."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

FEATURE_COLUMNS = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)
TARGET_COLUMN = "species"


def validate_dataset(data: pd.DataFrame) -> None:
    """Raise ValueError when required data is missing or invalid."""
    required_columns = (*FEATURE_COLUMNS, TARGET_COLUMN)
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if data.loc[:, required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in FEATURE_COLUMNS):
        raise ValueError("All Iris feature columns must be numeric.")


def load_iris_data() -> pd.DataFrame:
    """Return all 150 Iris examples as a validated pandas DataFrame."""
    iris = load_iris(as_frame=True)
    data = iris.data.copy()
    data[TARGET_COLUMN] = iris.target.map(dict(enumerate(iris.target_names)))
    validate_dataset(data)
    return data


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load bundled Iris data or a CSV containing the same columns."""
    if path is None:
        return load_iris_data()

    data = pd.read_csv(path)
    validate_dataset(data)
    return data
