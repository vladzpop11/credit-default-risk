"""Check the local model against the evaluated one recorded in reports/test_metrics.json.

python -m credit_default_risk.verify model       models/model.joblib is the evaluated file
python -m credit_default_risk.verify reproduce   a local retrain is within tolerance of it

Both exit with status 1 when the check fails. The model file's hash changes on every
retrain (the bundle stores its training time), and retraining on another platform gives a
slightly different model (the macOS and Linux builds of xgboost differ in floating-point
detail), so `reproduce` compares metrics within a tolerance rather than files.
"""

import sys

from credit_default_risk import artifacts
from credit_default_risk.config import load_config

PR_AUC_TOLERANCE = 0.005
THRESHOLD_TOLERANCE = 0.02


def check_model(cfg: dict) -> bool:
    evaluated = artifacts.evaluated_model_sha(cfg)
    path = artifacts.model_path(cfg)
    if evaluated is None or not path.exists():
        print(f"cannot verify: need {path} and {artifacts.test_metrics_path(cfg)}")
        return False
    actual = artifacts.sha256(path)
    ok = actual == evaluated
    print(f"{path.name} {'IS' if ok else 'is NOT'} the model evaluated on the test set")
    print(f"  evaluated {evaluated[:16]}…  local {actual[:16]}…")
    return ok


def check_reproduce(cfg: dict) -> bool:
    evaluated = artifacts.read_json(artifacts.test_metrics_path(cfg))
    summary = artifacts.read_json(artifacts.local_train_summary_path(cfg))
    path = artifacts.model_path(cfg)
    if evaluated is None or summary is None or not path.exists():
        print("reproduce check: FAIL (no local retrain to compare; run make train first)")
        return False
    if summary["model_sha256"] != artifacts.sha256(path):
        print("reproduce check: FAIL (models/train_summary.json does not describe model.joblib)")
        return False
    if summary["model_sha256"] == evaluated["model_sha256"]:
        print("reproduce check: PASS (the local model is the evaluated model itself)")
        return True

    retrained = summary["oof_metrics"]
    pr_gap = retrained["pr_auc"] - evaluated["oof_pr_auc"]
    thr_gap = retrained["threshold"] - evaluated["threshold"]
    ok = abs(pr_gap) <= PR_AUC_TOLERANCE and abs(thr_gap) <= THRESHOLD_TOLERANCE + 1e-9
    print(
        f"retrained OOF PR-AUC {retrained['pr_auc']:.4f} vs evaluated "
        f"{evaluated['oof_pr_auc']:.4f} (gap {pr_gap:+.4f}, tolerance {PR_AUC_TOLERANCE})"
    )
    print(
        f"retrained threshold {retrained['threshold']:.2f} vs evaluated "
        f"{evaluated['threshold']:.2f} (tolerance {THRESHOLD_TOLERANCE})"
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
