"""Stratified 5-fold cross-validation of all candidates on the training split.

Writes reports/cv_results.{csv,md} and the out-of-fold probabilities that the threshold
choice uses later. The test split is not touched.
"""

import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import SENSITIVE, prepare
from credit_default_risk.models import candidates, make_pipeline
from credit_default_risk.split import load_split

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1@0.5": "f1",
    "recall@0.5": "recall",
    "precision@0.5": "precision",
}


def evaluate(name: str, pipeline, X, y, cv) -> dict:
    start = time.perf_counter()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    row = {"model": name}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()
    row["seconds"] = time.perf_counter() - start
    print(f"{name:40s} PR-AUC {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}")
    return row


def to_markdown(results: pd.DataFrame) -> str:
    lines = [
        "# Cross-validation results (training split, stratified 5-fold)",
        "",
        "Mean ± standard deviation over the 5 folds. "
        "F1, recall and precision use the default 0.5 threshold; "
        "the decision threshold is chosen separately, from the cost of errors.",
        "",
        "| Model | PR-AUC | ROC-AUC | F1@0.5 | Recall@0.5 | Precision@0.5 | Time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in results.iterrows():
        cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}" for m in SCORING]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + f" | {r['seconds']:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_config()
    seed, target = cfg["seed"], cfg["data"]["target"]
    train = load_split("train", cfg)
    X, y = prepare(train), train[target]
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)

    models = candidates(seed)
    rows = [evaluate(name, make_pipeline(model), X, y, cv) for name, model in models.items()]
    for name in ["logistic regression", "xgboost"]:
        pipeline = make_pipeline(clone(models[name]), exclude=SENSITIVE)
        rows.append(evaluate(f"{name}, without SEX/MARRIAGE/AGE", pipeline, X, y, cv))
    results = pd.DataFrame(rows)

    reports = project_path(cfg["paths"]["reports_dir"])
    results.to_csv(reports / "cv_results.csv", index=False)
    (reports / "cv_results.md").write_text(to_markdown(results))

    oof = pd.DataFrame({target: y}, index=X.index)
    for name in ["logistic regression", "xgboost"]:
        pipeline = make_pipeline(clone(models[name]))
        oof[name] = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    oof.to_csv(project_path(cfg["data"]["processed_dir"]) / "oof_predictions.csv")
    print(f"results written to {reports}")


if __name__ == "__main__":
    main()
