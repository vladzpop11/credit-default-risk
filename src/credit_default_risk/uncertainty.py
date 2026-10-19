"""Bootstrap confidence intervals for the recorded test result.

Re-scores the test split with the evaluated model (refusing any other model file), checks
that the scores reproduce reports/test_metrics.json exactly, then resamples test clients
to put 95% intervals on the headline metrics. Nothing is retrained or re-chosen: this adds
uncertainty to the existing result, it is not a new use of the test set for decisions.
Writes reports/test_uncertainty.json.
"""

import sys

import numpy as np
from sklearn.metrics import average_precision_score

from credit_default_risk import artifacts
from credit_default_risk.config import load_config
from credit_default_risk.predict import load_bundle, predict
from credit_default_risk.split import load_split
from credit_default_risk.threshold import metrics_at, total_cost

N_BOOTSTRAP = 1000


def interval(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def reproduces(again: dict, recorded: dict) -> bool:
    """Same decisions and the same PR-AUC: the same model scored the same test clients."""
    return (
        again["confusion"] == recorded["confusion"]
        and abs(again["pr_auc"] - recorded["pr_auc"]) <= 1e-12
    )


def main() -> None:
    cfg = load_config()
    recorded = artifacts.read_json(artifacts.test_metrics_path(cfg))
    if recorded is None:
        sys.exit("no recorded test result (reports/test_metrics.json)")
    bundle = load_bundle(artifacts.model_path(cfg), expected_sha256=recorded["model_sha256"])

    test = load_split("test", cfg)
    y = test[cfg["data"]["target"]].to_numpy().astype(bool)
    proba = predict(bundle, test)["default_probability"].to_numpy()
    cost = bundle["metadata"]["cost"]
    t = bundle["threshold"]
    if not reproduces(metrics_at(y, proba, t, cost["fn"], cost["fp"]), recorded):
        sys.exit("re-scored test metrics do not match the recorded ones; refusing to continue")

    rng = np.random.default_rng(cfg["seed"])
    metrics = ("pr_auc", "recall", "cut_vs_flag_all", "cut_vs_approve_all")
    samples: dict[str, list[float]] = {k: [] for k in metrics}
    n = len(y)
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, n)
        yb, pb = y[i], proba[i]
        model = total_cost(yb, pb >= t, cost["fn"], cost["fp"])
        samples["pr_auc"].append(average_precision_score(yb, pb))
        samples["recall"].append((pb[yb] >= t).mean())
        samples["cut_vs_flag_all"].append(1 - model / (cost["fp"] * (~yb).sum()))
        samples["cut_vs_approve_all"].append(1 - model / (cost["fn"] * yb.sum()))

    point = {
        "pr_auc": recorded["pr_auc"],
        "recall": recorded["recall"],
        "cut_vs_flag_all": 1 - recorded["cost_per_client"] / recorded["cost_per_client_flag_all"],
        "cut_vs_approve_all": 1
        - recorded["cost_per_client"] / recorded["cost_per_client_flag_none"],
    }
    result = {
        "method": f"percentile bootstrap over test clients, {N_BOOTSTRAP} resamples, "
        f"seed {cfg['seed']}; threshold fixed at {t}",
        "model_sha256": recorded["model_sha256"],
        **{k: {"point": point[k], "ci95": interval(v)} for k, v in samples.items()},
    }
    gap = [s - recorded["oof_pr_auc"] for s in samples["pr_auc"]]
    result["pr_auc_gap_vs_oof"] = {"point": recorded["pr_auc_gap_vs_oof"], "ci95": interval(gap)}
    artifacts.write_json(
        artifacts.test_metrics_path(cfg).with_name("test_uncertainty.json"), result
    )
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['point']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
