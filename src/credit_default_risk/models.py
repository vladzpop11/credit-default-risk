"""Candidate models, each wrapped in the same preprocessing pipeline."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_default_risk.features import build_preprocessor


def candidates(seed: int) -> dict:
    return {
        "baseline (prior)": DummyClassifier(strategy="prior"),
        "logistic regression": LogisticRegression(C=1.0, max_iter=2000),
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def make_pipeline(model, exclude: list[str] | None = None) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(exclude)), ("model", model)])
