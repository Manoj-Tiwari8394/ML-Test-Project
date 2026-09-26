"""Paths shared by the command-line entry points."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
DEFAULT_EXPERIMENT_NAME = "iris-logistic-regression"
