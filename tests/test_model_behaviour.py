"""Behavioural tests of the trained model in models/model.joblib.

They check what the model does, not how well it scores: which inputs must not matter,
which changes must move the risk in a known direction, and how fast it answers. Clients
come from the training split, so the test split stays reserved for the one evaluation.

Thresholds sit below the values measured on the evaluated (macOS) model, with room for
the slightly different model CI retrains on Linux; MODEL_CARD.md lists both.

Skipped until a model and the real split exist. `make test-model` sets REQUIRE_MODEL=1,
which turns the skip into a failure, so CI cannot pass by skipping them.
"""

import os
import time

import numpy as np
import pandas as pd
import pytest

from credit_default_risk import artifacts
from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import SENSITIVE
from credit_default_risk.predict import load_bundle, predict
from credit_default_risk.split import load_split

CFG = load_config()


@pytest.fixture(scope="module")
def bundle() -> dict:
    path = artifacts.model_path(CFG)
    split = project_path(CFG["data"]["processed_dir"]) / "train_ids.csv"
    if not (path.exists() and split.exists()):
        message = "no trained model or real split (make data split train)"
        if os.environ.get("REQUIRE_MODEL") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return load_bundle(path)


@pytest.fixture(scope="module")
def clients(bundle) -> pd.DataFrame:
    return load_split("train", CFG)


def risk(bundle: dict, records: pd.DataFrame) -> np.ndarray:
    return predict(bundle, records)["default_probability"].to_numpy()


def effect(bundle: dict, records: pd.DataFrame, **changes) -> dict[str, float]:
    """How each client's risk moves when `changes` are applied (callables get the frame)."""
    changed = records.copy()
    for col, value in changes.items():
        changed[col] = value(changed) if callable(value) else value
    diff = risk(bundle, changed) - risk(bundle, records)
    return {"up": np.mean(diff > 0), "down": np.mean(diff < 0), "mean": diff.mean()}


# Invariance: inputs that must not matter


def test_sensitive_attributes_have_no_effect(bundle, clients) -> None:
    """Sex, marital status and age are excluded, so any value, or none, gives the same risk."""
    rng = np.random.default_rng(0)
    scrambled = clients.assign(
        SEX=rng.integers(1, 3, len(clients)),
        MARRIAGE=rng.integers(0, 4, len(clients)),
        AGE=rng.integers(21, 80, len(clients)),
    )
    expected = risk(bundle, clients)
    np.testing.assert_array_equal(risk(bundle, scrambled), expected)
    np.testing.assert_array_equal(risk(bundle, clients.drop(columns=SENSITIVE)), expected)


def test_risk_does_not_depend_on_batch_or_order(bundle, clients) -> None:
    sample = clients.sample(100, random_state=0)
    batch = risk(bundle, sample)
    one_by_one = np.array([risk(bundle, sample.loc[[i]])[0] for i in sample.index])
    np.testing.assert_array_equal(one_by_one, batch)
    shuffled = sample.sample(frac=1, random_state=1)
    np.testing.assert_array_equal(
        predict(bundle, shuffled).loc[sample.index, "default_probability"], batch
    )


# Direction: changes that must move the risk one way


@pytest.mark.parametrize(
    ("before", "after", "min_share_up", "min_mean_increase"),
    [
        ((-2, -1, 0), 1, 0.90, 0.02),  # paid on time -> one month late
        ((1,), 2, 0.95, 0.10),  # one month late -> two months late
    ],
    ids=["on-time-to-1-month-late", "1-to-2-months-late"],
)
def test_being_later_raises_risk(
    bundle, clients, before, after, min_share_up, min_mean_increase
) -> None:
    group = clients[clients["PAY_1"].isin(before)]
    moved = effect(bundle, group, PAY_1=after)
    assert moved["up"] >= min_share_up
    assert moved["mean"] >= min_mean_increase


@pytest.mark.xfail(
    strict=True,
    reason="known flaw (README, Limitations): risk peaks at 3 months late and falls for "
    "longer delays, where training data is thin. Fixing it (monotonic constraints) makes "
    "this test pass; strict=True then fails the run so the xfail gets removed.",
)
def test_longer_delays_do_not_lower_risk(bundle, clients) -> None:
    three_months_late = clients.assign(PAY_1=3)
    assert effect(bundle, three_months_late, PAY_1=6)["down"] <= 0.10


def test_paying_the_previous_bill_in_full_lowers_risk_on_average(bundle, clients) -> None:
    """Holds on average and for most clients, not for every client (see MODEL_CARD.md)."""
    owing = clients[(clients["BILL_AMT2"] > 0) & (clients["PAY_AMT1"] < clients["BILL_AMT2"])]
    moved = effect(bundle, owing, PAY_AMT1=lambda d: d["BILL_AMT2"])
    assert moved["mean"] < 0
    assert moved["down"] >= 0.65


def test_stopping_payments_raises_risk_on_average(bundle, clients) -> None:
    paying = clients[clients["PAY_AMT1"] > 0]
    moved = effect(bundle, paying, PAY_AMT1=0)
    assert moved["mean"] > 0
    assert moved["up"] >= 0.70


# Speed


def test_single_client_latency(bundle, clients) -> None:
    """In-process, one client at a time: about 24 ms median on a 2-core 2018 MacBook Air.

    The budget is generous so a slow CI runner does not fail it; it catches order-of-
    magnitude regressions, such as a feature that scans the whole dataset per request.
    """
    one = clients.iloc[[0]]
    for _ in range(3):
        predict(bundle, one)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        predict(bundle, one)
        times.append(time.perf_counter() - start)
    assert np.median(times) < 0.25
