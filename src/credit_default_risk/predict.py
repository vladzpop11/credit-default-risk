"""Save, load and use the trained model bundle (pipeline + threshold + metadata).

The bundle is a joblib pickle: loading it runs code stored in the file. Only load model
files you produced or trust; pass `expected_sha256` to check the file before unpickling.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_default_risk.artifacts import sha256
from credit_default_risk.features import RAW_FEATURES, prepare

BUNDLE_KEYS = {"pipeline", "threshold", "input_features", "metadata"}


def input_features(exclude: list[str]) -> list[str]:
    """Raw columns a client record must provide."""
    return [c for c in RAW_FEATURES if c not in exclude]


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f"{path} does not match the expected SHA-256; refusing to unpickle it")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or BUNDLE_KEYS - bundle.keys():
        raise ValueError(f"{path} is not a model bundle with keys {sorted(BUNDLE_KEYS)}")
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
