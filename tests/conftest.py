from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_default_risk.config import load_config
from credit_default_risk.features import BILL_AMTS, PAY_STATUS, RAW_FEATURES, SENSITIVE, prepare
from credit_default_risk.models import chosen_pipeline
from credit_default_risk.predict import input_features, save_bundle


def _make_clients(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic clients with realistic value ranges and a target tied to repayment status.

    Bills include negative values (overpayments), as in the real data.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 50_000, n) for c in RAW_FEATURES})
    df[BILL_AMTS] = rng.integers(-5_000, 50_000, (n, len(BILL_AMTS)))
    df["SEX"] = rng.integers(1, 3, n)
    df["EDUCATION"] = rng.integers(0, 7, n)
    df["MARRIAGE"] = rng.integers(0, 4, n)
    df["AGE"] = rng.integers(21, 70, n)
    for col in PAY_STATUS:
        df[col] = rng.integers(-2, 9, n)
    df["LIMIT_BAL"] = rng.integers(10_000, 500_000, n)
    df["default"] = (df["PAY_1"] > 0).astype(int) ^ (rng.random(n) < 0.1)
    return df


@pytest.fixture
def make_clients() -> Callable[..., pd.DataFrame]:
    return _make_clients


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model bundle trained once on synthetic data, saved like train.py saves it."""
    cfg = load_config()
    cfg["model"]["params"] = {**cfg["model"]["params"], "n_estimators": 10}
    clients = _make_clients(n=300, seed=1)
    pipeline = chosen_pipeline(cfg).fit(prepare(clients), clients["default"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    save_bundle(
        {
            "pipeline": pipeline,
            "threshold": 0.16,
            "input_features": input_features(SENSITIVE),
            "metadata": {"trained_at": "test", "package_version": "test"},
        },
        path,
    )
    return path
