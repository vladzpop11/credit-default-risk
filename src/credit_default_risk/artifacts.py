"""Where the pipeline's artifacts live, and how their integrity is recorded.

models/model.joblib            the local model (git-ignored)
models/train_summary.json      how the local model was trained (git-ignored)
reports/test_metrics.json      the one evaluation on the test set, with the model's SHA-256
reports/train_summary.json     training summary of the evaluated model (tracked record)

Retraining writes only to models/, so it never overwrites the tracked record of the
evaluated model.
"""

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from credit_default_risk.config import project_path

MODEL_FILE = "model.joblib"
TRAIN_SUMMARY_FILE = "train_summary.json"
TEST_METRICS_FILE = "test_metrics.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(cfg: dict) -> Path:
    return project_path(cfg["paths"]["models_dir"]) / MODEL_FILE


def local_train_summary_path(cfg: dict) -> Path:
    return project_path(cfg["paths"]["models_dir"]) / TRAIN_SUMMARY_FILE


def evaluated_train_summary_path(cfg: dict) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / TRAIN_SUMMARY_FILE


def test_metrics_path(cfg: dict) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / TEST_METRICS_FILE


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def evaluated_model_sha(cfg: dict) -> str | None:
    metrics = read_json(test_metrics_path(cfg))
    return metrics.get("model_sha256") if metrics else None


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": f"{sys.platform}-{platform.machine()}"}
    for lib in ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = version(f"{lib}-cpu")  # xgboost-cpu on Linux
    return out
