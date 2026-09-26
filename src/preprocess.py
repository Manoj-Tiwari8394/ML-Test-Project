"""Prepare the Iris DataFrame for model training."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the input measurements (X) from the species label (y)."""
    return data.loc[:, FEATURE_COLUMNS], data[TARGET_COLUMN]


def split_train_test(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split examples while keeping each species represented in both sets."""
    features, target = split_features_target(data)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
