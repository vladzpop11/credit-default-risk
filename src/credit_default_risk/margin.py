"""How much the model adds over the "ever late" rule, with a 95% interval.

The rule "flag anyone who was late in the last 6 months" has no fitted parameters. On the
training split's out-of-fold predictions (data/processed/oof_chosen.csv, from make train or
make oof), this compares the expected error cost per client of the model at its threshold
with the rule's, and bootstraps clients (paired: the same resample for both) for an
interval. The test split is not used. Writes reports/rule_margin.json.
"""

import numpy as np
import pandas as pd

from credit_default_risk import artifacts
from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import prepare
from credit_default_risk.split import load_split
from credit_default_risk.threshold import total_cost

N_BOOTSTRAP = 2000


def paired_margin(y: np.ndarray, flagged: np.ndarray, rule: np.ndarray, cost_fn: float,
                  cost_fp: float, seed: int) -> dict:  # fmt: skip
    """Relative cost cut of `flagged` over `rule`, and its paired bootstrap interval."""

    def cut(i: np.ndarray) -> float:
        return 1 - total_cost(y[i], flagged[i], cost_fn, cost_fp) / total_cost(
            y[i], rule[i], cost_fn, cost_fp
        )

    rng = np.random.default_rng(seed)
    everyone = np.arange(len(y))
    samples = [cut(rng.integers(0, len(y), len(y))) for _ in range(N_BOOTSTRAP)]
    return {
        "cut": cut(everyone),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "resamples": N_BOOTSTRAP,
    }


def main() -> None:
    cfg = load_config()
    cost_fn, cost_fp = cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"]
    oof_path = project_path(cfg["data"]["processed_dir"]) / "oof_chosen.csv"
    if not oof_path.exists():
        raise SystemExit("no out-of-fold predictions (data/processed/oof_chosen.csv): run make oof")
    summary = artifacts.read_json(artifacts.evaluated_train_summary_path(cfg))
    threshold = summary["oof_metrics"]["threshold"]
    train = load_split("train", cfg)
    X = prepare(train)
    y = train[cfg["data"]["target"]].to_numpy().astype(bool)
    oof = pd.read_csv(oof_path, index_col="ID")
    if not X.index.equals(oof.index):
        raise SystemExit("oof_chosen.csv does not match the training split: run make oof")
    flagged = oof["proba"].to_numpy() >= threshold
    rule = (X["months_late"] > 0).to_numpy()
    result = paired_margin(y, flagged, rule, cost_fn, cost_fp, cfg["seed"])
    result.update(threshold=threshold, cost={"fn": cost_fn, "fp": cost_fp},
                  data="training split, out-of-fold predictions")  # fmt: skip
    artifacts.write_json(project_path(cfg["paths"]["reports_dir"]) / "rule_margin.json", result)
    low, high = result["ci95"]
    print(f"model cuts the 'ever late' rule's cost by {result['cut']:.1%} "
          f"(95% CI {low:.1%}-{high:.1%}, paired bootstrap)")  # fmt: skip


if __name__ == "__main__":
    main()
