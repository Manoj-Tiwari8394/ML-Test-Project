import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.data_loader import FEATURE_COLUMNS, load_iris_data
from src.preprocess import split_train_test
from src.train import log_mlflow_run, train_model
from src.evaluate import evaluate_model


def test_training_run_logs_tags_metrics_and_registered_model(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = (tmp_path / "artifacts").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", artifact_root)

    X_train, X_test, y_train, y_test = split_train_test(load_iris_data())
    model = train_model(X_train, y_train)
    run_id, version = log_mlflow_run(
        model,
        X_train,
        {"c": 1.0, "experiment_name": "registry-test"},
        evaluate_model(model, X_test, y_test),
        tags={"git.commit": "test-sha", "github.actor": "test-user"},
        registered_model_name="iris-classifier-test",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["git.commit"] == "test-sha"
    assert run.data.tags["github.actor"] == "test-user"
    assert run.data.params["c"] == "1.0"
    assert 0.0 <= run.data.metrics["accuracy"] <= 1.0
    assert version is not None

    registered = client.get_model_version("iris-classifier-test", version)
    assert registered.run_id == run_id
    assert registered.tags["git.commit"] == "test-sha"

    logged_model = mlflow.sklearn.load_model(f"models:/iris-classifier-test/{version}")
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURE_COLUMNS)
    assert logged_model.predict(sample).shape == (1,)
