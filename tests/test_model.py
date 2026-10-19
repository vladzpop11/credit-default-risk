import joblib
import numpy as np
import pytest

from credit_default_risk.artifacts import sha256
from credit_default_risk.predict import load_bundle, predict
from credit_default_risk.threshold import choose_threshold, metrics_at


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


def test_load_bundle_checks_the_hash_before_unpickling(tmp_path) -> None:
    # Not a pickle at all: if it were unpickled, joblib would raise a different error.
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="refusing to unpickle"):
        load_bundle(path, expected_sha256="0" * 64)


def test_load_bundle_accepts_the_expected_hash(bundle_path) -> None:
    assert load_bundle(bundle_path, expected_sha256=sha256(bundle_path))["threshold"] == 0.16


@pytest.mark.parametrize("obj", [["a list"], {"pipeline": None, "threshold": 0.5}])
def test_load_bundle_rejects_non_bundles(tmp_path, obj) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump(obj, path)
    with pytest.raises(ValueError, match="not a model bundle"):
        load_bundle(path)


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
    assert m["cost_per_client_flag_none"] == pytest.approx(5 * 2 / 4)
    assert m["cost_per_client_flag_all"] == pytest.approx(1 * 2 / 4)
