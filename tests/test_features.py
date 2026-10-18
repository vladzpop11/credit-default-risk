import numpy as np
import pandas as pd

from credit_default_risk.features import (
    ENGINEERED,
    RAW_FEATURES,
    SENSITIVE,
    build_preprocessor,
    prepare,
)


def make_clients(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 50_000, n) for c in RAW_FEATURES})
    df["SEX"] = rng.integers(1, 3, n)
    df["EDUCATION"] = rng.integers(0, 7, n)
    df["MARRIAGE"] = rng.integers(0, 4, n)
    df["AGE"] = rng.integers(21, 70, n)
    for m in range(1, 7):
        df[f"PAY_{m}"] = rng.integers(-2, 9, n)
    df["LIMIT_BAL"] = rng.integers(10_000, 500_000, n)
    return df


def test_undocumented_codes_are_merged() -> None:
    df = make_clients()
    df.loc[0, ["EDUCATION", "MARRIAGE"]] = [0, 0]
    df.loc[1, "EDUCATION"] = 6
    out = prepare(df)
    assert set(out["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"]) <= {1, 2, 3}


def test_prepare_is_row_wise() -> None:
    df = make_clients()
    assert prepare(df.iloc[[3]]).equals(prepare(df).iloc[[3]])


def test_engineered_features() -> None:
    df = make_clients()
    df.loc[0, [f"PAY_{m}" for m in range(1, 7)]] = [2, 0, 1, -1, -2, 3]
    out = prepare(df)
    assert set(ENGINEERED) <= set(out.columns)
    assert out.loc[0, "months_late"] == 3
    assert out.loc[0, "max_months_late"] == 3
    assert out["payment_ratio_1"].le(1.5).all()


def test_preprocessor_can_exclude_sensitive_features() -> None:
    X = prepare(make_clients())
    names = build_preprocessor(exclude=SENSITIVE).fit(X).get_feature_names_out()
    assert not any(n.startswith(("SEX", "MARRIAGE", "AGE")) for n in names)
