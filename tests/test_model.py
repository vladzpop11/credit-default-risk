import numpy as np
import pandas as pd
import pytest

from credit_default_risk.config import load_config, project_path
from credit_default_risk.features import AMOUNTS, SENSITIVE, build_preprocessor, prepare
from credit_default_risk.predict import load_bundle, predict
from credit_default_risk.threshold import choose_threshold, metrics_at


def test_preprocessing_is_fitted_on_train_only(make_clients) -> None:
    """Scaling statistics must come from the rows the pipeline was fitted on.

    The "test" rows have amounts 1000x larger. If they leaked into fitting, the
    training rows would not be centred and the test rows would not be far off-centre.
    """
    train = prepare(make_clients(n=200, seed=1))
    test_raw = make_clients(n=200, seed=2)
    test_raw[AMOUNTS] = test_raw[AMOUNTS] * 1000
    test = prepare(test_raw)

    pre = build_preprocessor(exclude=SENSITIVE).fit(train)
    names = list(pre.get_feature_names_out())
    cols = [names.index(c) for c in AMOUNTS]

    assert np.allclose(pre.transform(train)[:, cols].mean(axis=0), 0, atol=1e-8)
    assert (pre.transform(test)[:, cols].mean(axis=0) > 1).all()


def test_train_and_test_ids_are_disjoint() -> None:
    processed = project_path(load_config()["data"]["processed_dir"])
    if not (processed / "test_ids.csv").exists():
        pytest.skip("data not downloaded (run make data split)")
    train = set(pd.read_csv(processed / "train_ids.csv")["ID"])
    test = set(pd.read_csv(processed / "test_ids.csv")["ID"])
    assert train.isdisjoint(test)


def test_saved_model_predicts_one_client(bundle_path, make_clients) -> None:
    bundle = load_bundle(bundle_path)
    client = make_clients(n=1, seed=3)[bundle["input_features"]]
    out = predict(bundle, client)
    assert out.shape == (1, 2)
    assert 0.0 <= out["default_probability"].iloc[0] <= 1.0
    assert out["default_prediction"].iloc[0] == (out["default_probability"].iloc[0] >= 0.16)


def test_predictions_shape_and_range(bundle_path, make_clients) -> None:
    clients = make_clients(n=50, seed=4)
    out = predict(load_bundle(bundle_path), clients)
    assert len(out) == 50
    assert out["default_probability"].between(0, 1).all()


def test_predict_rejects_missing_columns(bundle_path, make_clients) -> None:
    clients = make_clients(n=2).drop(columns=["PAY_1"])
    with pytest.raises(ValueError, match="PAY_1"):
        predict(load_bundle(bundle_path), clients)


def test_threshold_matches_decision_theory() -> None:
    """With calibrated probabilities the cost-optimal threshold is FP / (FP + FN)."""
    rng = np.random.default_rng(0)
    proba = rng.random(200_000)
    y = rng.random(200_000) < proba
    assert choose_threshold(y, proba, cost_fn=5, cost_fp=1) == pytest.approx(1 / 6, abs=0.02)


def test_metrics_at_threshold() -> None:
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.2])
    m = metrics_at(y, proba, 0.5, cost_fn=5, cost_fp=1)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["cost_per_client"] == pytest.approx((5 + 1) / 4)
