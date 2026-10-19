"""Cost of decisions, the cost-minimising threshold, and metrics at a threshold.

Cost model: a missed defaulter (false negative) costs `cost_fn`, a wrongly flagged good
client (false positive) costs `cost_fp`, and correct decisions cost nothing. Costs are
reported per client, in units of one false-positive cost when cost_fp = 1.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def total_cost(y: ArrayLike, flagged: ArrayLike, cost_fn: float, cost_fp: float) -> float:
    y, flagged = np.asarray(y, bool), np.asarray(flagged, bool)
    fn = np.sum(y & ~flagged)
    fp = np.sum(~y & flagged)
    return float(cost_fn * fn + cost_fp * fp)


def cost_curve(
    y: ArrayLike, proba: ArrayLike, cost_fn: float, cost_fp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average cost per client for every threshold in GRID."""
    proba = np.asarray(proba)
    costs = np.array([total_cost(y, proba >= t, cost_fn, cost_fp) for t in GRID]) / len(proba)
    return GRID, costs


def choose_threshold(y: ArrayLike, proba: ArrayLike, cost_fn: float, cost_fp: float) -> float:
    grid, costs = cost_curve(y, proba, cost_fn, cost_fp)
    return float(grid[np.argmin(costs)])


def metrics_at(
    y: ArrayLike, proba: ArrayLike, threshold: float, cost_fn: float, cost_fp: float
) -> dict[str, Any]:
    y, proba = np.asarray(y, bool), np.asarray(proba)
    flagged = proba >= threshold
    tp = int(np.sum(y & flagged))
    fp = int(np.sum(~y & flagged))
    fn = int(np.sum(y & ~flagged))
    tn = int(np.sum(~y & ~flagged))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n = len(y)
    return {
        "n": n,
        "threshold": threshold,
        # PR-AUC is computed as average precision.
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flagged_share": float(flagged.mean()),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "cost_per_client": total_cost(y, flagged, cost_fn, cost_fp) / n,
        # Reference policies: approve everyone (no model) and flag everyone.
        "cost_per_client_flag_none": cost_fn * int(y.sum()) / n,
        "cost_per_client_flag_all": cost_fp * int((~y).sum()) / n,
    }


def print_metrics(label: str, m: dict[str, Any], reference: dict[str, Any] | None = None) -> None:
    for key in ["pr_auc", "roc_auc", "recall", "precision", "flagged_share", "cost_per_client"]:
        ref = f"   (OOF {reference[key]:.4f})" if reference else ""
        print(f"  {label} {key}: {m[key]:.4f}{ref}")
    print(
        f"  {label} cost per client if everyone is approved: {m['cost_per_client_flag_none']:.4f}"
        f", if everyone is flagged: {m['cost_per_client_flag_all']:.4f}"
    )
