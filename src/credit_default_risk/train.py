"""Train the chosen model on the full training split.

1. Out-of-fold probabilities (stratified 5-fold) on the training split.
2. Decision threshold that minimises the configured error cost on those probabilities.
3. Refit on the whole training split and save the model bundle.

The test split is not used here; see evaluate.py.
"""

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit_default_risk import __version__
from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import prepare
from credit_default_risk.models import chosen_pipeline
from credit_default_risk.predict import input_features, save_bundle
from credit_default_risk.split import load_split
from credit_default_risk.threshold import choose_threshold, metrics_at


def library_versions() -> dict[str, str]:
    libs = ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]
    out = {"python": platform.python_version()}
    for lib in libs:
        try:
            out[lib] = version(lib)
        except Exception:
            out[lib] = version(f"{lib}-cpu")  # xgboost-cpu on Linux
    return out


def main() -> None:
    cfg = load_config()
    seed, target = cfg["seed"], cfg["data"]["target"]
    cost_fn, cost_fp = cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"]

    train = load_split("train", cfg)
    X, y = prepare(train), train[target]
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)

    oof = cross_val_predict(chosen_pipeline(cfg), X, y, cv=cv, method="predict_proba")[:, 1]
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
        "seed": seed,
        "versions": library_versions(),
        "oof_metrics": oof_metrics,
    }
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "input_features": input_features(cfg["model"]["exclude_features"]),
        "metadata": metadata,
    }
    save_bundle(bundle, project_path(cfg["paths"]["models_dir"]) / "model.joblib")

    summary_path = project_path(cfg["paths"]["reports_dir"]) / "train_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"threshold {threshold:.2f} (cost FN:FP = {cost_fn}:{cost_fp})")
    for key in ["pr_auc", "roc_auc", "recall", "precision", "flagged_share", "cost_per_client"]:
        print(f"  OOF {key}: {oof_metrics[key]:.4f}")
    print(f"  OOF cost per client with no model: {oof_metrics['cost_per_client_flag_none']:.4f}")


if __name__ == "__main__":
    main()
