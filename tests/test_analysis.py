"""The functions behind the reported calibration, fairness, baseline and comparison claims."""

import numpy as np
import pandas as pd
import pytest

from credit_default_risk.analysis import (
    baselines,
    cost_sensitivity,
    group_calibration,
    near_optimal_range,
    reliability,
    reweighted_fpr,
)
from credit_default_risk.config import load_config
from credit_default_risk.cv import paired
from credit_default_risk.evaluate import build_record
from credit_default_risk.uncertainty import reproduces


@pytest.fixture
def calibrated() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    proba = rng.beta(2, 6, 50_000)
    return rng.random(proba.size) < proba, proba


def test_group_calibration_flags_only_a_miscalibrated_group(calibrated) -> None:
    y, proba = calibrated
    df = pd.DataFrame({"y": y, "proba": proba})
    biased = df.copy()
    biased.loc[: len(df) // 2, "proba"] *= 0.9  # under-predict the first half by 10%
    first = pd.Series(df.index <= len(df) // 2, index=df.index)
    groups = {"first": first, "second": ~first}

    honest = group_calibration(df, groups)
    assert not honest["significant after correction"].any()
    result = group_calibration(biased, groups).set_index("group")
    assert result.loc["first", "significant after correction"]
    assert result.loc["first", "z"] < 0  # predicted below observed
    assert not result.loc["second", "significant after correction"]


def test_reweighted_fpr_removes_a_composition_effect() -> None:
    # Within each stratum the group and the reference have the same false positive rate;
    # only their mix of strata differs. Reweighting must recover the reference's rate.
    strata = pd.Series(["low"] * 100 + ["high"] * 100 + ["low"] * 20 + ["high"] * 180)
    group = pd.Series([True] * 200 + [False] * 200)
    flagged = np.r_[
        np.arange(100) < 60, np.arange(100) < 10, np.arange(20) < 12, np.arange(180) < 18
    ]
    df = pd.DataFrame({"y": False, "flagged": flagged})
    raw = df.loc[group, "flagged"].mean()
    adjusted = reweighted_fpr(df, group, ~group, strata)
    reference = df.loc[~group, "flagged"].mean()
    assert raw == pytest.approx(0.35)
    assert adjusted == pytest.approx(reference)


def test_reliability_of_calibrated_scores(calibrated) -> None:
    y, proba = calibrated
    table, scores = reliability(y, proba)
    assert len(table) == 10 and table["clients"].sum() == y.size
    assert table["gap (pp)"].abs().max() < 1.5
    assert scores["ece"] < 0.01
    assert scores["brier"] < scores["brier_prior"]


def test_baselines_and_rule_costs() -> None:
    y = np.array([True, True, False, False, False])
    X = pd.DataFrame({"PAY_1": [1, 0, 1, 0, 0], "months_late": [1, 0, 2, 0, 0]})
    cfg = load_config()  # 5:1 costs
    table = baselines(y, X, np.array([0.9, 0.2, 0.1, 0.05, 0.3]), 0.16, cfg).set_index("policy")
    assert table.loc["approve everyone", "cost per client"] == pytest.approx(5 * 2 / 5)
    assert table.loc["flag everyone", "cost per client"] == pytest.approx(3 / 5)
    # Rule "ever late" flags rows 0 and 2: misses row 1 (5) and one false alarm (1).
    assert table.loc["rule: ever late in 6 months", "cost per client"] == pytest.approx(6 / 5)


def test_cost_sensitivity_columns_are_consistent(calibrated) -> None:
    y, proba = calibrated
    ever_late = proba > 0.3
    table = cost_sensitivity(y, proba, ever_late)
    for _, row in table.iterrows():
        rule_cost = row["best trivial policy cost"] * (
            1 - row["'ever late' rule cut vs best trivial"]
        )
        assert row["model cut vs 'ever late' rule"] == pytest.approx(
            1 - row["model cost"] / rule_cost
        )
    assert (table["model cut vs best trivial"] > 0).all()


def test_near_optimal_range_contains_the_optimum(calibrated) -> None:
    y, proba = calibrated
    lo, hi = near_optimal_range(y, proba, load_config(), tolerance=0.02)
    assert lo <= 1 / 6 <= hi


def test_paired_test_is_more_conservative_than_a_plain_t_test() -> None:
    from scipy import stats

    a = np.array([0.56, 0.58, 0.57, 0.55, 0.55])
    b = a - np.array([0.010, 0.004, 0.012, 0.002, 0.007])
    mean, sd, p = paired(a, b)
    assert mean == pytest.approx(0.007)
    assert p > stats.ttest_rel(a, b).pvalue
    assert paired(a, a - 0.5)[2] < 1e-6  # a clear difference stays clear


def test_build_record_identifies_the_model() -> None:
    bundle = {
        "metadata": {
            "oof_metrics": {"pr_auc": 0.56},
            "trained_at": "t",
            "versions": {"platform": "darwin-x86_64"},
        }
    }
    record = build_record({"pr_auc": 0.55, "confusion": {}}, bundle, "abc")
    assert record["model_sha256"] == "abc"
    assert record["pr_auc_gap_vs_oof"] == pytest.approx(-0.01)
    assert record["model_platform"] == "darwin-x86_64"


def test_uncertainty_requires_an_exact_reproduction() -> None:
    recorded = {"confusion": {"tp": 1, "fp": 2, "fn": 3, "tn": 4}, "pr_auc": 0.554}
    assert reproduces(dict(recorded), recorded)
    assert not reproduces({**recorded, "pr_auc": 0.5541}, recorded)
    assert not reproduces({**recorded, "confusion": {"tp": 2, "fp": 1, "fn": 3, "tn": 4}}, recorded)


def test_latency_record_is_only_updated_for_the_evaluated_model() -> None:
    from credit_default_risk.latency import output_path

    cfg = load_config()
    assert output_path(cfg, evaluated_model=True).parent.name == "reports"
    assert output_path(cfg, evaluated_model=False).parent.name == "models"
