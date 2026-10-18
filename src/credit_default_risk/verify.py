"""Check a trained model against the evaluated one recorded in reports/test_metrics.json.

python -m credit_default_risk.verify model       models/model.joblib is the evaluated file
python -m credit_default_risk.verify reproduce   a retrain (e.g. in CI) is within tolerance

Retraining on a different platform gives a slightly different model (the macOS and Linux
builds of xgboost differ in floating-point detail), so `reproduce` compares metrics within
a tolerance instead of comparing files.
"""

import json
import sys

from credit_default_risk.config import load_config, project_path
from credit_default_risk.train import file_sha256

PR_AUC_TOLERANCE = 0.005
THRESHOLD_TOLERANCE = 0.02


def check_model(cfg: dict) -> bool:
    evaluated = json.loads(
        (project_path(cfg["paths"]["reports_dir"]) / "test_metrics.json").read_text()
    )
    actual = file_sha256(project_path(cfg["paths"]["models_dir"]) / "model.joblib")
    ok = actual == evaluated["model_sha256"]
    status = "IS" if ok else "is NOT"
    print(f"models/model.joblib {status} the model evaluated on the test set")
    print(f"  evaluated {evaluated['model_sha256'][:16]}…  actual {actual[:16]}…")
    return ok


def check_reproduce(cfg: dict) -> bool:
    reports = project_path(cfg["paths"]["reports_dir"])
    evaluated = json.loads((reports / "test_metrics.json").read_text())
    retrained = json.loads((reports / "train_summary.json").read_text())["oof_metrics"]
    pr_gap = retrained["pr_auc"] - evaluated["oof_pr_auc"]
    thr_gap = retrained["threshold"] - evaluated["threshold"]
    ok = abs(pr_gap) <= PR_AUC_TOLERANCE and abs(thr_gap) <= THRESHOLD_TOLERANCE + 1e-9
    print(
        f"retrained OOF PR-AUC {retrained['pr_auc']:.4f} "
        f"vs evaluated {evaluated['oof_pr_auc']:.4f} "
        f"(gap {pr_gap:+.4f}, tolerance {PR_AUC_TOLERANCE})"
    )
    print(
        f"retrained threshold {retrained['threshold']:.2f} "
        f"vs evaluated {evaluated['threshold']:.2f} (tolerance {THRESHOLD_TOLERANCE})"
    )
    print("reproduce check:", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    checks = {"model": check_model, "reproduce": check_reproduce}
    if len(sys.argv) != 2 or sys.argv[1] not in checks:
        sys.exit(f"usage: python -m credit_default_risk.verify {{{'|'.join(checks)}}}")
    sys.exit(0 if checks[sys.argv[1]](load_config()) else 1)


if __name__ == "__main__":
    main()
