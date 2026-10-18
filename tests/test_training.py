"""Leakage and artifact tests on the real training code path (train.out_of_fold)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_default_risk import train, verify
from credit_default_risk.config import load_config
from credit_default_risk.features import prepare
from credit_default_risk.models import chosen_pipeline


@pytest.fixture
def small_cfg() -> dict:
    cfg = load_config()
    cfg["model"]["params"] = {**cfg["model"]["params"], "n_estimators": 30}
    return cfg


def test_out_of_fold_has_no_skill_on_shuffled_labels(make_clients, small_cfg) -> None:
    """With labels shuffled, nothing can be learned honestly.

    Catches a training path that scores rows with a model that saw them (e.g. fitting on
    all rows instead of per fold): that gives ROC-AUC ~0.85 here. It cannot catch target
    information inside the features, because shuffled labels are unrelated to them;
    test_model_inputs_do_not_depend_on_the_target covers that.
    """
    clients = make_clients(n=2000, seed=5)
    X = prepare(clients)
    y = pd.Series(
        np.random.default_rng(0).permutation(clients["default"].to_numpy()), index=X.index
    )
    oof = train.out_of_fold(X, y, small_cfg)
    assert roc_auc_score(y, oof) < 0.56


def test_model_inputs_do_not_depend_on_the_target(make_clients) -> None:
    """Feature preparation must give identical inputs whatever the labels are."""
    clients = make_clients(n=200, seed=7)
    flipped = clients.assign(default=1 - clients["default"])
    out = prepare(clients)
    assert "default" not in out.columns
    pd.testing.assert_frame_equal(out, prepare(flipped))


def test_out_of_fold_learns_real_signal(make_clients, small_cfg) -> None:
    clients = make_clients(n=2000, seed=5)
    X, y = prepare(clients), clients["default"]
    assert roc_auc_score(y, train.out_of_fold(X, y, small_cfg)) > 0.80


def test_each_fold_is_scored_by_a_pipeline_fitted_without_it(make_clients, small_cfg) -> None:
    """Out-of-fold scores equal a pipeline fitted from scratch on the other folds only."""
    clients = make_clients(n=600, seed=6)
    X, y = prepare(clients), clients["default"]
    oof = train.out_of_fold(X, y, small_cfg)
    cv = StratifiedKFold(5, shuffle=True, random_state=small_cfg["seed"])
    fit_idx, val_idx = next(cv.split(X, y))
    manual = chosen_pipeline(small_cfg).fit(X.iloc[fit_idx], y.iloc[fit_idx])
    np.testing.assert_allclose(oof[val_idx], manual.predict_proba(X.iloc[val_idx])[:, 1], rtol=1e-6)


def write_reports(tmp_path, cfg, test_metrics: dict, train_summary: dict | None = None) -> dict:
    cfg = {
        **cfg,
        "paths": {**cfg["paths"], "reports_dir": str(tmp_path), "models_dir": str(tmp_path)},
    }
    (tmp_path / "test_metrics.json").write_text(json.dumps(test_metrics))
    if train_summary:
        (tmp_path / "train_summary.json").write_text(json.dumps(train_summary))
    return cfg


def test_verify_model_detects_a_replaced_file(tmp_path, small_cfg, monkeypatch) -> None:
    monkeypatch.setattr(verify, "project_path", lambda p: tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"evaluated model")
    sha = train.file_sha256(tmp_path / "model.joblib")
    cfg = write_reports(tmp_path, small_cfg, {"model_sha256": sha})
    assert verify.check_model(cfg)
    (tmp_path / "model.joblib").write_bytes(b"retrained model")
    assert not verify.check_model(cfg)


@pytest.mark.parametrize(("pr_auc", "threshold", "ok"), [(0.5575, 0.17, True), (0.54, 0.16, False)])
def test_reproduce_check_tolerance(tmp_path, small_cfg, monkeypatch, pr_auc, threshold, ok) -> None:
    monkeypatch.setattr(verify, "project_path", lambda p: tmp_path)
    cfg = write_reports(
        tmp_path,
        small_cfg,
        {"oof_pr_auc": 0.5588, "threshold": 0.16},
        {"oof_metrics": {"pr_auc": pr_auc, "threshold": threshold}},
    )
    assert verify.check_reproduce(cfg) is ok
