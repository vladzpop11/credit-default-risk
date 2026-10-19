"""Stratified k-fold cross-validation of all candidates on the training split.

Writes reports/cv_results.{csv,md}. Every model is scored on the same folds, so models are
compared with paired fold differences (the honest noise yardstick), not the spread of each
model's own scores. The test split is not touched.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import SENSITIVE, prepare
from credit_default_risk.models import candidates, make_cv, make_pipeline
from credit_default_risk.split import load_split

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1@0.5": "f1",
    "recall@0.5": "recall",
    "precision@0.5": "precision",
}
CHOSEN = "xgboost, without SEX/MARRIAGE/AGE"


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict[str, np.ndarray]:
    start = time.perf_counter()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    out = {m: scores[f"test_{m}"] for m in SCORING}
    out["seconds"] = np.array(time.perf_counter() - start)
    return out


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Mean and standard deviation of fold-wise a - b, and the paired t-test p-value."""
    d = a - b
    return float(d.mean()), float(d.std(ddof=1)), float(stats.ttest_rel(a, b).pvalue)


def to_markdown(results: dict[str, dict], n_splits: int) -> str:
    lines = [
        f"# Cross-validation results (training split, stratified {n_splits}-fold)",
        "",
        f"Mean ± standard deviation (ddof=1) over the {n_splits} folds. All models use fixed, "
        "untuned settings. F1, recall and precision use the default 0.5 threshold; the "
        "decision threshold is chosen separately, from the cost of errors.",
        "",
        "| Model | PR-AUC | ROC-AUC | F1@0.5 | Recall@0.5 | Precision@0.5 | Time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, s in results.items():
        cells = [f"{s[m].mean():.3f} ± {s[m].std(ddof=1):.3f}" for m in SCORING]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {float(s['seconds']):.0f} |")
    lines += [
        "",
        f"## Paired PR-AUC differences against the chosen model ({CHOSEN})",
        "",
        "Fold-wise difference (chosen minus other), its standard deviation, and the paired "
        "t-test p-value. With 5 folds the test has little power: treat p > 0.05 as a tie.",
        "",
        "| Other model | Mean difference | SD | p-value |",
        "| --- | --- | --- | --- |",
    ]
    chosen = results[CHOSEN]["pr_auc"]
    for name, s in results.items():
        if name != CHOSEN:
            mean, sd, p = paired(chosen, s["pr_auc"])
            lines.append(f"| {name} | {mean:+.4f} | {sd:.4f} | {p:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_config()
    train = load_split("train", cfg)
    X, y = prepare(train), train[cfg["data"]["target"]]
    cv = make_cv(cfg)
    models = candidates(cfg)

    pipelines = {name: make_pipeline(model) for name, model in models.items()}
    # The repayment codes -2/-1/0 are categories; a linear model needs them one-hot encoded.
    pipelines["logistic regression, one-hot PAY codes"] = make_pipeline(
        clone(models["logistic regression"]), pay_onehot=True
    )
    pipelines["logistic regression, one-hot PAY, without SEX/MARRIAGE/AGE"] = make_pipeline(
        clone(models["logistic regression"]), exclude=SENSITIVE, pay_onehot=True
    )
    pipelines[CHOSEN] = make_pipeline(clone(models["xgboost"]), exclude=SENSITIVE)

    results = {}
    for name, pipeline in pipelines.items():
        results[name] = evaluate(pipeline, X, y, cv)
        print(f"{name:60s} PR-AUC {results[name]['pr_auc'].mean():.4f}")

    reports = project_path(cfg["paths"]["reports_dir"])
    rows = [
        {"model": name, "fold": k, **{m: s[m][k] for m in SCORING}}
        for name, s in results.items()
        for k in range(cfg["cv"]["n_splits"])
    ]
    pd.DataFrame(rows).to_csv(reports / "cv_results.csv", index=False)
    (reports / "cv_results.md").write_text(to_markdown(results, cfg["cv"]["n_splits"]))
    print(f"results written to {reports}")


if __name__ == "__main__":
    main()
