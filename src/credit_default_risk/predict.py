"""Save, load and use the trained model bundle (pipeline + threshold + metadata)."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_default_risk.features import RAW_FEATURES, prepare


def input_features(exclude: list[str]) -> list[str]:
    """Raw columns a client record must provide."""
    return [c for c in RAW_FEATURES if c not in exclude]


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    missing = {"pipeline", "threshold", "input_features", "metadata"} - bundle.keys()
    if missing:
        raise ValueError(f"model bundle {path} is missing {sorted(missing)}")
    return bundle


def predict(bundle: dict[str, Any], records: pd.DataFrame) -> pd.DataFrame:
    """Default probability and the thresholded decision for each client record."""
    missing = set(bundle["input_features"]) - set(records.columns)
    if missing:
        raise ValueError(f"records are missing columns {sorted(missing)}")
    proba = bundle["pipeline"].predict_proba(prepare(records))[:, 1]
    return pd.DataFrame(
        {"default_probability": proba, "default_prediction": proba >= bundle["threshold"]},
        index=records.index,
    )
