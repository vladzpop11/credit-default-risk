"""Analysis on the training split: threshold, baselines, cost sensitivity, calibration,
permutation importance, error segments and a fairness audit.

Uses the out-of-fold probabilities written by train.py (data/processed/oof_chosen.csv), so
every prediction comes from a model that did not see that client, and the local model's
threshold (models/train_summary.json). Writes reports/error_analysis_tables.md and
figures. The test split is not used.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss

from credit_default_risk import artifacts
from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import feature_columns, prepare
from credit_default_risk.models import chosen_pipeline, make_cv
from credit_default_risk.split import load_split
from credit_default_risk.threshold import GRID, choose_threshold, cost_curve, total_cost

COST_RATIOS = [2, 3, 4, 5, 7, 10]


def plot_cost_curve(
    y: pd.Series, proba: pd.Series, cfg: dict, threshold: float, path: Path
) -> None:
    cost_fn, cost_fp = cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"]
    grid, costs = cost_curve(y, proba, cost_fn, cost_fp)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, costs, color="#4C72B0", label="model")
    ax.axvline(threshold, color="#C44E52", linestyle="--", label=f"chosen {threshold:.2f}")
    ax.axhline(cost_fn * np.mean(y), color="grey", linestyle=":", label="approve everyone")
    ax.axhline(cost_fp * (1 - np.mean(y)), color="grey", linestyle="--", label="flag everyone")
    ax.set(
        title=f"Expected error cost per client (missed defaulter {cost_fn}, false alarm {cost_fp})",
        xlabel="decision threshold",
        ylabel="cost per client",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fold_permutation_importance(X: pd.DataFrame, y: pd.Series, cfg: dict) -> pd.DataFrame:
    """Fit on each training fold, permute features on its validation fold."""
    categorical, amounts, numeric = feature_columns(cfg["model"]["exclude_features"])
    used = categorical + amounts + numeric
    per_fold = []
    for train_idx, val_idx in make_cv(cfg).split(X, y):
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


def plot_importance(importance: pd.DataFrame, n_splits: int, path: Path, top: int = 12) -> None:
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(data.index, data["mean"], xerr=data["std"], color="#4C72B0")
    ax.set(
        title=f"Permutation importance (drop in PR-AUC, {n_splits} validation folds)",
        xlabel="mean drop in PR-AUC",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def policy_row(
    name: str, y: np.ndarray, flagged: np.ndarray, cost_fn: float, cost_fp: float
) -> dict:
    cost = total_cost(y, flagged, cost_fn, cost_fp) / len(y)
    flag_all = cost_fp * (~y).mean()
    return {
        "policy": name,
        "flagged": flagged.mean(),
        "recall": flagged[y].mean(),
        "precision": y[flagged].mean() if flagged.any() else np.nan,
        "cost per client": cost,
        "cut vs flag everyone": 1 - cost / flag_all,
    }


def baselines(
    y: np.ndarray, X: pd.DataFrame, proba: np.ndarray, threshold: float, cfg: dict
) -> pd.DataFrame:
    """The model against policies with no fitted parameters."""
    c = (cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"])
    return pd.DataFrame(
        [
            policy_row("approve everyone", y, np.zeros_like(y), *c),
            policy_row("flag everyone", y, np.ones_like(y), *c),
            policy_row("rule: late last month (PAY_1 ≥ 1)", y, (X["PAY_1"] >= 1).to_numpy(), *c),
            policy_row("rule: ever late in 6 months", y, (X["months_late"] > 0).to_numpy(), *c),
            policy_row(f"model, threshold {threshold:.2f}", y, proba >= threshold, *c),
        ]
    )


def cost_sensitivity(y: np.ndarray, proba: np.ndarray, ever_late: np.ndarray) -> pd.DataFrame:
    """Re-choose the threshold for other cost ratios (false alarm cost fixed at 1)."""
    rows = []
    for r in COST_RATIOS:
        t = choose_threshold(y, proba, r, 1)
        model = total_cost(y, proba >= t, r, 1) / len(y)
        best_trivial = min(r * y.mean(), (~y).mean())
        rule = total_cost(y, ever_late, r, 1) / len(y)
        rows.append(
            {
                "missed : false alarm": f"{r}:1",
                "threshold": t,
                "model cost": model,
                "best trivial policy cost": best_trivial,
                "model cut vs best trivial": 1 - model / best_trivial,
                "'ever late' rule cut vs best trivial": 1 - rule / best_trivial,
                "model cut vs 'ever late' rule": 1 - model / rule,
            }
        )
    return pd.DataFrame(rows)


def reliability(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    decile = pd.qcut(proba, bins, labels=False, duplicates="drop")
    table = (
        pd.DataFrame({"decile": decile + 1, "predicted": proba, "observed": y})
        .groupby("decile")
        .agg(
            clients=("observed", "size"),
            predicted=("predicted", "mean"),
            observed=("observed", "mean"),
        )
        .reset_index()
    )
    table["gap (pp)"] = (table["predicted"] - table["observed"]) * 100
    weights = table["clients"] / table["clients"].sum()
    scores = {
        "brier": brier_score_loss(y, proba),
        "brier_prior": brier_score_loss(y, np.full_like(proba, y.mean())),
        "ece": float((weights * (table["predicted"] - table["observed"]).abs()).sum()),
    }
    return table, scores


def fold_thresholds(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, cfg: dict) -> list[float]:
    c = (cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"])
    return [choose_threshold(y.iloc[v], proba[v], *c) for _, v in make_cv(cfg).split(X, y)]


def near_optimal_range(
    y: np.ndarray, proba: np.ndarray, cfg: dict, tolerance: float
) -> tuple[float, float]:
    """Thresholds whose cost is within `tolerance` (relative) of the minimum."""
    _, costs = cost_curve(y, proba, cfg["threshold"]["cost_fn"], cfg["threshold"]["cost_fp"])
    ok = GRID[costs <= (1 + tolerance) * costs.min()]
    return float(ok.min()), float(ok.max())


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


def group_calibration(df: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    """Test, per audited group, whether the mean prediction matches the observed rate.

    Under the hypothesis that the predictions are calibrated, the number of defaults in a
    group has variance sum(p_i (1 - p_i)), so the standard error of the gap comes from the
    predictions themselves. Significance is Bonferroni-corrected for the number of groups.
    """
    z, z_bonf = norm.ppf(0.975), norm.ppf(1 - 0.05 / (2 * len(groups)))
    rows = []
    for name, mask in groups.items():
        g = df[mask]
        se = np.sqrt((g["proba"] * (1 - g["proba"])).sum()) / len(g)
        gap = g["proba"].mean() - g["y"].mean()
        rows.append(
            {
                "group": name,
                "clients": len(g),
                "observed (%)": g["y"].mean() * 100,
                "mean predicted (%)": g["proba"].mean() * 100,
                "gap (pp)": gap * 100,
                "z": gap / se,
                "95% half-width (pp)": z * se * 100,
                "corrected half-width (pp)": z_bonf * se * 100,
                "significant after correction": bool(abs(gap) > z_bonf * se),
            }
        )
    return pd.DataFrame(rows)


def reweighted_fpr(
    df: pd.DataFrame, group: pd.Series, reference: pd.Series, strata: pd.Series
) -> float:
    """False-positive rate of `group` if its good clients had `reference`'s mix of `strata`."""
    good = ~df["y"]
    g_share = strata[group & good].value_counts(normalize=True)
    r_share = strata[reference & good].value_counts(normalize=True)
    weights = (r_share / g_share).reindex(strata[group & good]).to_numpy()
    return float(np.average(df.loc[group & good, "flagged"], weights=weights))


def format_cell(v: object, digits: int) -> str:
    if isinstance(v, bool | np.bool_):
        return "yes" if v else "no"
    if isinstance(v, int | np.integer):
        return f"{v:,}"
    if isinstance(v, float | np.floating):
        return "–" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def to_markdown(table: pd.DataFrame, digits: int = 3) -> str:
    # itertuples keeps each column's dtype; iterrows would upcast mixed rows to float.
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(format_cell(v, digits) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    target, n_splits = cfg["data"]["target"], cfg["cv"]["n_splits"]
    figures = project_path(cfg["paths"]["figures_dir"])
    reports = project_path(cfg["paths"]["reports_dir"])

    summary = artifacts.read_json(artifacts.local_train_summary_path(cfg))
    if summary is None:
        raise SystemExit("no models/train_summary.json: run make train first")
    threshold = summary["oof_metrics"]["threshold"]
    oof = pd.read_csv(project_path(cfg["data"]["processed_dir"]) / "oof_chosen.csv", index_col="ID")
    train = load_split("train", cfg)
    X, y_series = prepare(train), train[target]
    if not X.index.equals(oof.index):
        raise ValueError("oof_chosen.csv does not match the training split; run make train")
    y, proba = y_series.to_numpy().astype(bool), oof["proba"].to_numpy()

    plot_cost_curve(y_series, oof["proba"], cfg, threshold, figures / "cost_curve.png")
    importance = fold_permutation_importance(X, y_series, cfg)
    plot_importance(importance, n_splits, figures / "permutation_importance.png")

    df = pd.DataFrame({"y": y, "proba": proba, "flagged": proba >= threshold}, index=X.index)
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
    groups = {"female": sex == "female", "male": sex == "male"} | {
        f"age {band}": age == band for band in age.cat.categories
    }

    rel, scores = reliability(y, proba)
    per_fold = fold_thresholds(X, y_series, proba, cfg)
    within2, within5 = (near_optimal_range(y, proba, cfg, tol) for tol in (0.02, 0.05))
    young_fpr = df.loc[(age == "21-25") & ~df["y"], "flagged"].mean()
    young_fpr_reweighted = reweighted_fpr(df, age == "21-25", age == "26-40", limit)

    missed = df["y"] & ~df["flagged"]
    lines = [
        "# Error analysis tables (generated by `make analysis`)",
        "",
        f"Out-of-fold predictions on the training split ({len(df):,} clients), threshold "
        f"{threshold:.2f}. Costs are per client, in units of one false alarm; a missed "
        f"defaulter costs {cfg['threshold']['cost_fn']}. Recall = share of defaulters flagged; "
        "false positive rate = share of non-defaulters flagged.",
        "",
        "## Policies compared (no parameters fitted except the model)",
        "",
        to_markdown(baselines(y, X, proba, threshold, cfg)),
        "",
        "## Cost-ratio sensitivity (threshold re-chosen for each ratio)",
        "",
        to_markdown(cost_sensitivity(y, proba, (X["months_late"] > 0).to_numpy())),
        "",
        "## Threshold stability",
        "",
        f"- Cost-optimal threshold within each of the {n_splits} folds: "
        + ", ".join(f"{t:.2f}" for t in per_fold),
        f"- Thresholds within 2% of the minimum cost: {within2[0]:.2f}–{within2[1]:.2f}; "
        f"within 5%: {within5[0]:.2f}–{within5[1]:.2f}",
        "",
        "## Calibration (deciles of predicted probability)",
        "",
        to_markdown(rel),
        "",
        f"Brier score {scores['brier']:.4f} (prior-only {scores['brier_prior']:.4f}); "
        f"expected calibration error {scores['ece']:.4f}.",
        "",
        f"## Permutation importance (mean drop in PR-AUC over {n_splits} validation folds)",
        "",
        to_markdown(importance.reset_index(names="feature").head(12)),
        "",
    ]
    for title, by in [
        ("Most recent repayment status (PAY_1)", pay1),
        ("Months with a positive repayment status in the last 6", late),
        ("Credit limit quintile", limit),
        ("Education", education),
        ("Sex (not a model input; fairness audit)", sex),
        ("Age band (not a model input; fairness audit)", age),
    ]:
        lines += [f"## {title}", "", to_markdown(segment_table(df, by)), ""]
    lines += [
        "## Fairness audit: calibration by group",
        "",
        "Gap = mean predicted minus observed. The standard error assumes the predictions are "
        f"calibrated (variance from the predicted probabilities); 95% intervals are ±1.96 SE, "
        f"and the corrected ones are Bonferroni-adjusted for {len(groups)} groups.",
        "",
        to_markdown(group_calibration(df, groups), digits=2),
        "",
        "## Fairness audit: young clients and credit limits",
        "",
        f"False positive rate of good clients aged 21–25: {young_fpr:.1%}. Reweighted to the "
        f"credit-limit quintile mix of good clients aged 26–40: {young_fpr_reweighted:.1%}. "
        f"Median credit limit: NT${train.loc[age == '21-25', 'LIMIT_BAL'].median():,.0f} "
        f"(21–25) vs NT${train.loc[age == '26-40', 'LIMIT_BAL'].median():,.0f} (26–40).",
        "",
        "## Missed defaulters",
        "",
        f"{int(missed.sum()):,} of {int(df['y'].sum()):,} defaulters were not flagged. "
        f"{(X.loc[missed, 'months_late'] == 0).mean():.1%} of them had no positive repayment "
        f"status in the last 6 months (vs {(X.loc[df['y'], 'months_late'] == 0).mean():.1%} "
        "of all defaulters).",
        "",
    ]
    (reports / "error_analysis_tables.md").write_text("\n".join(lines))
    print(f"written {reports / 'error_analysis_tables.md'} and figures in {figures}")


if __name__ == "__main__":
    main()
