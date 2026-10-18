"""Train the chosen model on the full training split.

1. Out-of-fold probabilities (stratified 5-fold) on the training split.
2. Decision threshold that minimises the configured error cost on those probabilities.
3. Refit on the whole training split and save the model bundle, with its SHA-256.

The test split is not used here; see evaluate.py. Training refuses to replace a model
that has already been evaluated on the test set (set FORCE=1 to override).
"""

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit_default_risk import __version__
from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import prepare
from credit_default_risk.models import chosen_pipeline
from credit_default_risk.predict import input_features, save_bundle
from credit_default_risk.split import load_split
from credit_default_risk.threshold import choose_threshold, metrics_at


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": f"{sys.platform}-{platform.machine()}"}
    for lib in ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = version(f"{lib}-cpu")  # xgboost-cpu on Linux
    return out


def out_of_fold(X: pd.DataFrame, y: pd.Series, cfg: dict) -> np.ndarray:
    """Probabilities where each row is scored by a pipeline that never saw it.

    The whole pipeline, preprocessing included, is refitted inside every fold.
    """
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    return cross_val_predict(chosen_pipeline(cfg), X, y, cv=cv, method="predict_proba")[:, 1]


def evaluated_model_sha(cfg: dict) -> str | None:
    path = project_path(cfg["paths"]["reports_dir"]) / "test_metrics.json"
    return json.loads(path.read_text()).get("model_sha256") if path.exists() else None


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    cost_fn, cost_fp = cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"]
    model_path = project_path(cfg["paths"]["models_dir"]) / "model.joblib"

    evaluated = evaluated_model_sha(cfg)
    if (
        model_path.exists()
        and evaluated == file_sha256(model_path)
        and os.environ.get("FORCE") != "1"
    ):
        sys.exit(
            f"{model_path} is the model evaluated on the test set (reports/test_metrics.json). "
            "Retraining would replace it. Set FORCE=1 to override."
        )

    train = load_split("train", cfg)
    X, y = prepare(train), train[target]

    oof = out_of_fold(X, y, cfg)
    threshold = choose_threshold(y, oof, cost_fn, cost_fp)
    oof_metrics = metrics_at(y, oof, threshold, cost_fn, cost_fp)
    pd.DataFrame({target: y, "proba": oof}, index=X.index).to_csv(
        project_path(cfg["data"]["processed_dir"]) / "oof_chosen.csv"
    )

    pipeline = chosen_pipeline(cfg).fit(X, y)
    metadata = {
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": cfg["model"],
        "cost": {"fn": cost_fn, "fp": cost_fp},
        "train_rows": len(X),
        "seed": cfg["seed"],
        "versions": library_versions(),
        "oof_metrics": oof_metrics,
    }
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "input_features": input_features(cfg["model"]["exclude_features"]),
        "metadata": metadata,
    }
    save_bundle(bundle, model_path)

    summary = {**metadata, "model_sha256": file_sha256(model_path)}
    summary_path = project_path(cfg["paths"]["reports_dir"]) / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"threshold {threshold:.2f} (cost FN:FP = {cost_fn}:{cost_fp})")
    for key in ["pr_auc", "roc_auc", "recall", "precision", "flagged_share", "cost_per_client"]:
        print(f"  OOF {key}: {oof_metrics[key]:.4f}")
    print(f"  OOF cost per client with no model: {oof_metrics['cost_per_client_flag_none']:.4f}")
    print(f"  model sha256: {summary['model_sha256'][:16]}…")


if __name__ == "__main__":
    main()
