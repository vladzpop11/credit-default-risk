"""Final evaluation of the saved model on the test split.

Refuses to run again once reports/test_metrics.json exists (set FORCE=1 to override):
re-running after changing the model would turn the test set into a validation set.
The result records the SHA-256 of the evaluated model file, and the model's training
summary is copied to reports/train_summary.json as the tracked record.
"""

import os
import sys
from pathlib import Path

from credit_default_risk import artifacts
from credit_default_risk.config import load_config
from credit_default_risk.predict import load_bundle, predict
from credit_default_risk.split import load_split
from credit_default_risk.threshold import metrics_at, print_metrics

MAX_PR_AUC_GAP = 0.03


def ensure_first_use(metrics_path: Path) -> None:
    if metrics_path.exists() and os.environ.get("FORCE") != "1":
        sys.exit(
            f"{metrics_path} already exists: the test set has been used. Set FORCE=1 to override."
        )


def main() -> None:
    cfg = load_config()
    out = artifacts.test_metrics_path(cfg)
    ensure_first_use(out)

    model_path = artifacts.model_path(cfg)
    model_sha = artifacts.sha256(model_path)
    summary = artifacts.read_json(artifacts.local_train_summary_path(cfg))
    if summary is None or summary["model_sha256"] != model_sha:
        sys.exit("models/train_summary.json does not describe models/model.joblib; run make train")

    bundle = load_bundle(model_path, expected_sha256=model_sha)
    test = load_split("test", cfg)
    y = test[cfg["data"]["target"]]
    proba = predict(bundle, test)["default_probability"]

    cost = bundle["metadata"]["cost"]
    metrics = metrics_at(y, proba, bundle["threshold"], cost["fn"], cost["fp"])
    oof = bundle["metadata"]["oof_metrics"]
    gap = metrics["pr_auc"] - oof["pr_auc"]
    metrics |= {
        "oof_pr_auc": oof["pr_auc"],
        "pr_auc_gap_vs_oof": gap,
        "model_trained_at": bundle["metadata"]["trained_at"],
        "model_platform": bundle["metadata"]["versions"]["platform"],
        "model_sha256": model_sha,
    }
    artifacts.write_json(out, metrics)
    artifacts.write_json(artifacts.evaluated_train_summary_path(cfg), summary)

    print_metrics("test", metrics, reference=oof)
    if abs(gap) > MAX_PR_AUC_GAP:
        print(f"WARNING: PR-AUC differs from CV by {gap:+.3f}; investigate, do not retune on test.")


if __name__ == "__main__":
    main()
