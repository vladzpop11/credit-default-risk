"""Candidate models, each wrapped in the same preprocessing pipeline."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_default_risk.features import build_preprocessor


def make_cv(cfg: dict) -> StratifiedKFold:
    """The one cross-validation splitter used everywhere, so all results share folds."""
    return StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])


def xgboost(cfg: dict) -> XGBClassifier:
    return XGBClassifier(**cfg["model"]["params"], n_jobs=-1, random_state=cfg["seed"])


def candidates(cfg: dict) -> dict[str, BaseEstimator]:
    """Models compared in cross-validation, all with fixed (untuned) settings."""
    return {
        "baseline (prior)": DummyClassifier(strategy="prior"),
        "logistic regression": LogisticRegression(C=1.0, max_iter=3000),
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=cfg["seed"]
        ),
        "xgboost": xgboost(cfg),
    }


def make_pipeline(
    model: BaseEstimator, exclude: list[str] | None = None, pay_onehot: bool = False
) -> Pipeline:
    return Pipeline(
        [("preprocess", build_preprocessor(exclude, pay_onehot=pay_onehot)), ("model", model)]
    )


def chosen_pipeline(cfg: dict) -> Pipeline:
    """The model selected at the step 4 checkpoint, as configured in configs/config.yaml."""
    spec = cfg["model"]
    if spec["name"] != "xgboost":
        raise ValueError(f"unsupported model: {spec['name']}")
    return make_pipeline(xgboost(cfg), exclude=spec["exclude_features"])
