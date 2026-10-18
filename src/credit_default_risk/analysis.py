"""Step 5 analysis on the training split: cost curve, permutation importance, error segments.

Uses the out-of-fold probabilities written by train.py, so every prediction comes from a
model that did not see that client. Writes reports/error_analysis_tables.md and figures.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import feature_columns, prepare
from credit_default_risk.models import chosen_pipeline
from credit_default_risk.split import load_split
from credit_default_risk.threshold import cost_curve


def plot_cost_curve(y, proba, cfg, threshold: float, path) -> None:
    cost_fn, cost_fp = cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"]
    grid, costs = cost_curve(y, proba, cost_fn, cost_fp)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, costs, color="#4C72B0")
    ax.axvline(threshold, color="#C44E52", linestyle="--", label=f"chosen {threshold:.2f}")
    ax.axhline(cost_fn * np.mean(y), color="grey", linestyle=":", label="approve everyone")
    ax.axhline(cost_fp * (1 - np.mean(y)), color="grey", linestyle="--", label="flag everyone")
    ax.set(
        title=f"Expected cost per client (FN cost {cost_fn}, FP cost {cost_fp})",
        xlabel="decision threshold",
        ylabel="cost per client",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fold_permutation_importance(X, y, cfg) -> pd.DataFrame:
    """Fit on each training fold, permute features on its validation fold."""
    categorical, amounts, numeric = feature_columns(cfg["model"]["exclude_features"])
    used = categorical + amounts + numeric
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    per_fold = []
    for train_idx, val_idx in cv.split(X, y):
        pipeline = chosen_pipeline(cfg).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            pipeline,
            X.iloc[val_idx][used],
            y.iloc[val_idx],
            scoring="average_precision",
            n_repeats=5,
            random_state=cfg["seed"],
            n_jobs=1,
        )
        per_fold.append(pd.Series(result.importances_mean, index=used))
    folds = pd.concat(per_fold, axis=1)
    return pd.DataFrame({"mean": folds.mean(axis=1), "std": folds.std(axis=1)}).sort_values(
        "mean", ascending=False
    )


def plot_importance(importance: pd.DataFrame, path, top: int = 12) -> None:
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(data.index, data["mean"], xerr=data["std"], color="#4C72B0")
    ax.set(
        title="Permutation importance (drop in PR-AUC, 5 validation folds)",
        xlabel="mean drop in PR-AUC",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def segment_table(df: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """Per-segment default rate, flag rate, recall, false-positive rate and calibration."""
    rows = []
    for segment, g in df.groupby(by, observed=True):
        pos, neg = g["y"], ~g["y"]
        rows.append(
            {
                "segment": segment,
                "clients": len(g),
                "default rate": g["y"].mean(),
                "flagged": g["flagged"].mean(),
                "recall": g.loc[pos, "flagged"].mean() if pos.any() else np.nan,
                "false positive rate": g.loc[neg, "flagged"].mean() if neg.any() else np.nan,
                "mean predicted": g["proba"].mean(),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for _, r in table.iterrows():
        cells = [
            f"{v:.3f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
            for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    figures = project_path(cfg["paths"]["figures_dir"])
    reports = project_path(cfg["paths"]["reports_dir"])

    summary = json.loads((reports / "train_summary.json").read_text())
    threshold = summary["oof_metrics"]["threshold"]
    oof = pd.read_csv(project_path(cfg["data"]["processed_dir"]) / "oof_chosen.csv", index_col="ID")
    train = load_split("train", cfg)
    X, y = prepare(train), train[target]
    assert X.index.equals(oof.index)

    plot_cost_curve(y, oof["proba"], cfg, threshold, figures / "cost_curve.png")

    importance = fold_permutation_importance(X, y, cfg)
    plot_importance(importance, figures / "permutation_importance.png")

    df = pd.DataFrame(
        {"y": y.astype(bool), "proba": oof["proba"], "flagged": oof["proba"] >= threshold}
    )
    pay1 = pd.cut(
        X["PAY_1"],
        [-3, 0, 1, 99],
        labels=["paid / revolving (≤0)", "1 month late", "2+ months late"],
    )
    limit = pd.qcut(X["LIMIT_BAL"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    late = pd.cut(
        X["months_late"], [-1, 0, 2, 6], labels=["never late", "1-2 months", "3-6 months"]
    )
    education = X["EDUCATION"].map(
        {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
    )
    # Fairness audit: the model does not use these, but its errors may still differ by group.
    sex = train["SEX"].map({1: "male", 2: "female"})
    age = pd.cut(train["AGE"], [20, 25, 40, 60, 80], labels=["21-25", "26-40", "41-60", "61+"])

    sections = [
        ("Most recent repayment status (PAY_1)", pay1),
        ("Months late in the last 6", late),
        ("Credit limit quintile", limit),
        ("Education", education),
        ("Sex (not a model input; fairness audit)", sex),
        ("Age band (not a model input; fairness audit)", age),
    ]
    missed = df["y"] & ~df["flagged"]
    lines = [
        "# Error analysis tables (generated by `make analysis`)",
        "",
        f"Out-of-fold predictions on the training split, threshold {threshold:.2f}. "
        "Recall = share of defaulters flagged; false positive rate = share of "
        "non-defaulters flagged; mean predicted vs default rate shows calibration.",
        "",
        "## Permutation importance (mean drop in PR-AUC over 5 validation folds)",
        "",
        to_markdown(importance.reset_index(names="feature").head(12)),
        "",
    ]
    for title, by in sections:
        lines += [f"## {title}", "", to_markdown(segment_table(df, by)), ""]
    lines += [
        "## Missed defaulters",
        "",
        f"{int(missed.sum())} of {int(df['y'].sum())} defaulters were not flagged. "
        f"{(X.loc[missed, 'months_late'] == 0).mean():.1%} of them were never late "
        f"in the last 6 months (vs {(X.loc[df['y'], 'months_late'] == 0).mean():.1%} "
        "of all defaulters).",
        "",
    ]
    (reports / "error_analysis_tables.md").write_text("\n".join(lines))
    print(f"written {reports / 'error_analysis_tables.md'} and figures in {figures}")


if __name__ == "__main__":
    main()
