"""Final, one-time evaluation of the saved model on the test split.

Refuses to run again once reports/test_metrics.json exists (set FORCE=1 to override):
re-running after changing the model would turn the test set into a validation set.
"""

import json
import os
import sys

from credit_default_risk.config import load_config, project_path
from credit_default_risk.predict import load_bundle, predict
from credit_default_risk.split import load_split
from credit_default_risk.threshold import metrics_at

MAX_PR_AUC_GAP = 0.03


def main() -> None:
    cfg = load_config()
    out = project_path(cfg["paths"]["reports_dir"]) / "test_metrics.json"
    if out.exists() and os.environ.get("FORCE") != "1":
        sys.exit(f"{out} already exists: the test set has been used. Set FORCE=1 to override.")

    bundle = load_bundle(project_path(cfg["paths"]["models_dir"]) / "model.joblib")
    test = load_split("test", cfg)
    y = test[cfg["data"]["target"]]
    proba = predict(bundle, test)["default_probability"]

    cost = bundle["metadata"]["cost"]
    metrics = metrics_at(y, proba, bundle["threshold"], cost["fn"], cost["fp"])
    oof = bundle["metadata"]["oof_metrics"]
    gap = metrics["pr_auc"] - oof["pr_auc"]
    metrics["oof_pr_auc"] = oof["pr_auc"]
    metrics["pr_auc_gap_vs_oof"] = gap
    metrics["model_trained_at"] = bundle["metadata"]["trained_at"]
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    for key in ["pr_auc", "roc_auc", "recall", "precision", "flagged_share", "cost_per_client"]:
        print(f"test {key}: {metrics[key]:.4f}   (OOF {oof[key]:.4f})")
    print(f"test cost per client with no model: {metrics['cost_per_client_flag_none']:.4f}")
    if abs(gap) > MAX_PR_AUC_GAP:
        print(f"WARNING: PR-AUC differs from CV by {gap:+.3f}; investigate, do not retune on test.")


if __name__ == "__main__":
    main()
