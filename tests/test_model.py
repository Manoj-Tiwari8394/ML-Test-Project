import numpy as np
import pytest

from src.data_loader import FEATURE_COLUMNS, load_iris_data
from src.evaluate import evaluate_model
from src.predict import predict_species
from src.preprocess import split_train_test
from src.train import build_model, train_model


def test_model_trains_and_predicts_expected_shape_and_species():
    X_train, X_test, y_train, y_test = split_train_test(load_iris_data())

    model = train_model(X_train, y_train, c=1.0)
    predictions = model.predict(X_test)
    metrics = evaluate_model(model, X_test, y_test)

    assert predictions.shape == (len(X_test),)
    assert set(predictions).issubset(set(y_train))
    assert set(metrics) == {"accuracy", "macro_f1"}
    assert all(0.0 <= score <= 1.0 for score in metrics.values())


def test_prediction_accepts_named_measurements():
    X_train, _, y_train, _ = split_train_test(load_iris_data())
    model = train_model(X_train, y_train)
    measurements = dict(zip(FEATURE_COLUMNS, [5.1, 3.5, 1.4, 0.2]))

    prediction = predict_species(model, measurements)

    assert prediction == "setosa"
    assert isinstance(prediction, str)


def test_prediction_rejects_missing_measurement():
    X_train, _, y_train, _ = split_train_test(load_iris_data())
    model = build_model().fit(X_train, y_train)

    with pytest.raises(ValueError, match="Missing feature measurements"):
        predict_species(model, {"sepal length (cm)": 5.1})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), True, "5.1"],
)
def test_prediction_rejects_non_finite_or_non_numeric_measurements(value):
    X_train, _, y_train, _ = split_train_test(load_iris_data())
    model = train_model(X_train, y_train)
    measurements = dict(zip(FEATURE_COLUMNS, [5.1, 3.5, 1.4, 0.2]))
    measurements[FEATURE_COLUMNS[0]] = value

    with pytest.raises(ValueError, match="measurements must"):
        predict_species(model, measurements)


@pytest.mark.parametrize("c", [0, -1, np.nan, np.inf])
def test_model_rejects_invalid_regularization_parameter(c):
    with pytest.raises(ValueError, match="finite number greater than zero"):
        build_model(c)
