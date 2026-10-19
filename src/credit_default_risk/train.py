"""Train the chosen model on the full training split.

1. Out-of-fold probabilities (stratified k-fold) on the training split.
2. Decision threshold that minimises the configured error cost on those probabilities.
3. Refit on the whole training split; save the bundle to models/model.joblib and its
   summary (metrics, versions, SHA-256) to models/train_summary.json.

The test split is not used here; see evaluate.py. Training refuses to replace a model
that has already been evaluated on the test set (set FORCE=1 to override).
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from credit_default_risk import __version__, artifacts
from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import prepare
from credit_default_risk.models import chosen_pipeline, make_cv
from credit_default_risk.predict import input_features, save_bundle
from credit_default_risk.split import load_split
from credit_default_risk.threshold import choose_threshold, metrics_at, print_metrics


def out_of_fold(X: pd.DataFrame, y: pd.Series, cfg: dict) -> np.ndarray:
    """Probabilities where each row is scored by a pipeline that never saw it.

    The whole pipeline, preprocessing included, is refitted inside every fold.
    """
    return cross_val_predict(chosen_pipeline(cfg), X, y, cv=make_cv(cfg), method="predict_proba")[
        :, 1
    ]


def ensure_not_evaluated(model_path: Path, evaluated_sha: str | None) -> None:
    """Refuse to overwrite the model file whose test results are on record."""
    if (
        evaluated_sha is not None
        and model_path.exists()
        and artifacts.sha256(model_path) == evaluated_sha
        and os.environ.get("FORCE") != "1"
    ):
        sys.exit(
            f"{model_path} is the model evaluated on the test set (reports/test_metrics.json). "
            "Retraining would replace it. Set FORCE=1 to override."
        )


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    cost_fn, cost_fp = cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"]
    model_path = artifacts.model_path(cfg)
    ensure_not_evaluated(model_path, artifacts.evaluated_model_sha(cfg))

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
        "versions": artifacts.library_versions(),
        "oof_metrics": oof_metrics,
    }
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "input_features": input_features(cfg["model"]["exclude_features"]),
        "metadata": metadata,
    }
    save_bundle(bundle, model_path)
    summary = {**metadata, "model_sha256": artifacts.sha256(model_path)}
    artifacts.write_json(artifacts.local_train_summary_path(cfg), summary)

    print(f"threshold {threshold:.2f} (cost FN:FP = {cost_fn}:{cost_fp})")
    print_metrics("OOF", oof_metrics)
    print(f"  model sha256: {summary['model_sha256'][:16]}… -> {model_path}")


if __name__ == "__main__":
    main()
