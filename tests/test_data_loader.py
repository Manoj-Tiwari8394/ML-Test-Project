import pandas as pd
import pytest

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    load_iris_data,
    validate_dataset,
)


def test_load_bundled_iris_has_expected_shape_and_columns():
    data = load_iris_data()

    assert data.shape == (150, 5)
    assert tuple(data.columns) == (*FEATURE_COLUMNS, TARGET_COLUMN)
    assert set(data[TARGET_COLUMN]) == {"setosa", "versicolor", "virginica"}


def test_load_dataset_reads_csv(tmp_path):
    expected = load_iris_data()
    csv_path = tmp_path / "iris.csv"
    expected.to_csv(csv_path, index=False)

    actual = load_dataset(csv_path)

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.drop(columns=[TARGET_COLUMN]), "missing required columns"),
        (lambda data: data.assign(**{TARGET_COLUMN: data[TARGET_COLUMN].where(data.index != 0)}),
         "missing values"),
        (lambda data: data.assign(**{FEATURE_COLUMNS[0]: "not numeric"}), "must be numeric"),
    ],
)
def test_validate_dataset_rejects_invalid_data(change, message):
    data = change(load_iris_data())

    with pytest.raises(ValueError, match=message):
        validate_dataset(data)
