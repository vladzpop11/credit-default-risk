"""Exploratory plots on the training split only. Conclusions are in reports/eda.md."""

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_default_risk.config import load_config, project_path
from credit_default_risk.split import load_split


def rate_bar(series: pd.Series, title: str, xlabel: str, overall: float, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot.bar(ax=ax, color="#4C72B0", rot=0, label="_nolegend_")
    ax.axhline(overall, color="#C44E52", linestyle="--", label=f"overall {overall:.1%}")
    ax.set(title=title, xlabel=xlabel, ylabel="default rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    out = project_path(cfg["paths"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)

    df = load_split("train", cfg)
    overall = df[target].mean()

    pay1 = df["PAY_1"].clip(upper=3)
    rate_bar(
        df.groupby(pay1)[target].mean().rename(index={3: "3+"}),
        "Default rate by most recent repayment status (PAY_1)",
        "PAY_1 (-2/-1/0 = paid or revolving, 1+ = months late)",
        overall,
        out / "default_by_pay1.png",
    )

    edges = df["LIMIT_BAL"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
    labels = [f"{lo / 1000:.0f}k-{hi / 1000:.0f}k" for lo, hi in pairwise(edges)]
    quintiles = pd.cut(df["LIMIT_BAL"], edges, labels=labels, include_lowest=True)
    by_limit = df.groupby(quintiles, observed=True)[target].mean()
    rate_bar(
        by_limit,
        "Default rate by credit limit quintile",
        "LIMIT_BAL (NT$)",
        overall,
        out / "default_by_limit.png",
    )

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.set_title("Feature correlations (train)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "correlations.png", dpi=120)
    plt.close(fig)

    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
