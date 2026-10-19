"""Leakage tests on the real training path, and the guards around the evaluated model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_default_risk import artifacts, evaluate, train, verify
from credit_default_risk.config import load_config
from credit_default_risk.features import prepare
from credit_default_risk.models import chosen_pipeline, make_cv


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
    assert roc_auc_score(y, train.out_of_fold(X, y, small_cfg)) < 0.56


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
    for fit_idx, val_idx in make_cv(small_cfg).split(X, y):
        manual = chosen_pipeline(small_cfg).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        expected = manual.predict_proba(X.iloc[val_idx])[:, 1]
        np.testing.assert_allclose(oof[val_idx], expected, rtol=1e-6)


# --- guards and verification ---------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch, small_cfg) -> dict:
    """Point all artifact paths at tmp_path/models and tmp_path/reports."""
    monkeypatch.setattr(artifacts, "project_path", lambda relative: tmp_path / relative)
    monkeypatch.delenv("FORCE", raising=False)
    return small_cfg


def write_model(cfg: dict, content: bytes) -> str:
    path = artifacts.model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return artifacts.sha256(path)


def record_evaluation(cfg: dict, sha: str, oof_pr_auc: float = 0.5588, threshold: float = 0.16):
    artifacts.write_json(
        artifacts.test_metrics_path(cfg),
        {"model_sha256": sha, "oof_pr_auc": oof_pr_auc, "threshold": threshold},
    )


def record_local_training(cfg: dict, sha: str, pr_auc: float, threshold: float) -> None:
    artifacts.write_json(
        artifacts.local_train_summary_path(cfg),
        {"model_sha256": sha, "oof_metrics": {"pr_auc": pr_auc, "threshold": threshold}},
    )


def test_train_refuses_to_replace_the_evaluated_model(workspace, monkeypatch) -> None:
    sha = write_model(workspace, b"evaluated model")
    with pytest.raises(SystemExit, match="evaluated on the test set"):
        train.ensure_not_evaluated(artifacts.model_path(workspace), sha)
    monkeypatch.setenv("FORCE", "1")
    train.ensure_not_evaluated(artifacts.model_path(workspace), sha)


def test_train_may_replace_a_model_that_was_not_evaluated(workspace) -> None:
    write_model(workspace, b"some retrain")
    train.ensure_not_evaluated(artifacts.model_path(workspace), "0" * 64)
    train.ensure_not_evaluated(artifacts.model_path(workspace), None)


def test_evaluate_refuses_a_second_use_of_the_test_set(workspace, monkeypatch) -> None:
    path = artifacts.test_metrics_path(workspace)
    evaluate.ensure_first_use(path)  # no record yet: fine
    record_evaluation(workspace, "abc")
    with pytest.raises(SystemExit, match="test set has been used"):
        evaluate.ensure_first_use(path)
    monkeypatch.setenv("FORCE", "1")
    evaluate.ensure_first_use(path)


def test_verify_model_detects_a_replaced_file(workspace) -> None:
    assert not verify.check_model(workspace)  # nothing to compare yet
    record_evaluation(workspace, write_model(workspace, b"evaluated model"))
    assert verify.check_model(workspace)
    write_model(workspace, b"retrained model")
    assert not verify.check_model(workspace)


def test_reproduce_check_needs_a_real_local_retrain(workspace) -> None:
    record_evaluation(workspace, "evaluated-sha")
    assert not verify.check_reproduce(workspace)  # no local model or summary
    write_model(workspace, b"retrain")
    record_local_training(workspace, "stale-sha", 0.5588, 0.16)
    assert not verify.check_reproduce(workspace)  # summary describes another file


def test_reproduce_check_passes_for_the_evaluated_model_itself(workspace) -> None:
    sha = write_model(workspace, b"evaluated model")
    record_evaluation(workspace, sha)
    record_local_training(workspace, sha, 0.5588, 0.16)
    assert verify.check_reproduce(workspace)


@pytest.mark.parametrize(
    ("pr_auc", "threshold", "ok"),
    [(0.5573, 0.16, True), (0.5575, 0.18, True), (0.54, 0.16, False), (0.5588, 0.20, False)],
)
def test_reproduce_check_tolerance(workspace, pr_auc, threshold, ok) -> None:
    record_evaluation(workspace, "evaluated-sha")
    sha = write_model(workspace, b"retrain")
    record_local_training(workspace, sha, pr_auc, threshold)
    assert verify.check_reproduce(workspace) is ok
