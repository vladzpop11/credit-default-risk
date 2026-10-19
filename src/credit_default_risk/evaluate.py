"""Final evaluation of the saved model on the test split.

Refuses to run again once reports/test_metrics.json exists (set FORCE=1 to override):
re-running after changing the model would turn the test set into a validation set.
The result records the SHA-256 of the evaluated model file, and the model's training
summary is copied to reports/train_summary.json as the tracked record.
"""

import os
import sys
from pathlib import Path
from typing import Any

from credit_default_risk import artifacts
from credit_default_risk.config import load_config
from credit_default_risk.predict import predict, read_bundle
from credit_default_risk.split import load_split
from credit_default_risk.threshold import metrics_at, print_metrics

MAX_PR_AUC_GAP = 0.03


def ensure_first_use(metrics_path: Path) -> None:
    if metrics_path.exists() and os.environ.get("FORCE") != "1":
        sys.exit(
            f"{metrics_path} already exists: the test set has been used. Set FORCE=1 to override."
        )


def build_record(metrics: dict[str, Any], bundle: dict[str, Any], model_sha: str) -> dict[str, Any]:
    """Test metrics plus what identifies the evaluated model."""
    oof = bundle["metadata"]["oof_metrics"]
    return metrics | {
        "oof_pr_auc": oof["pr_auc"],
        "pr_auc_gap_vs_oof": metrics["pr_auc"] - oof["pr_auc"],
        "model_trained_at": bundle["metadata"]["trained_at"],
        "model_platform": bundle["metadata"]["versions"]["platform"],
        "model_sha256": model_sha,
    }


def main() -> None:
    cfg = load_config()
    out = artifacts.test_metrics_path(cfg)
    ensure_first_use(out)

    summary = artifacts.read_json(artifacts.local_train_summary_path(cfg))
    if summary is None:
        sys.exit("no models/train_summary.json; run make train")
    # Only the model the summary describes is evaluated (hash checked before unpickling).
    bundle, model_sha = read_bundle(artifacts.model_path(cfg), summary["model_sha256"])
    test = load_split("test", cfg)
    y = test[cfg["data"]["target"]]
    proba = predict(bundle, test)["default_probability"]

    cost = bundle["metadata"]["cost"]
    record = build_record(
        metrics_at(y, proba, bundle["threshold"], cost["fn"], cost["fp"]), bundle, model_sha
    )
    artifacts.write_json(out, record)
    artifacts.write_json(artifacts.evaluated_train_summary_path(cfg), summary)

    print_metrics("test", record, reference=bundle["metadata"]["oof_metrics"])
    if abs(record["pr_auc_gap_vs_oof"]) > MAX_PR_AUC_GAP:
        gap = record["pr_auc_gap_vs_oof"]
        print(f"WARNING: PR-AUC differs from CV by {gap:+.3f}; investigate, do not retune on test.")


if __name__ == "__main__":
    main()
