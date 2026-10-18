from credit_default_risk.features import ENGINEERED, SENSITIVE, build_preprocessor, prepare


def test_undocumented_codes_are_merged(make_clients) -> None:
    df = make_clients()
    df.loc[0, ["EDUCATION", "MARRIAGE"]] = [0, 0]
    df.loc[1, "EDUCATION"] = 6
    out = prepare(df)
    assert set(out["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"]) <= {1, 2, 3}


def test_prepare_is_row_wise(make_clients) -> None:
    df = make_clients()
    assert prepare(df.iloc[[3]]).equals(prepare(df).iloc[[3]])


def test_prepare_accepts_records_without_sensitive_columns(make_clients) -> None:
    df = make_clients()
    out = prepare(df.drop(columns=SENSITIVE))
    assert out[SENSITIVE].isna().all().all()


def test_engineered_features(make_clients) -> None:
    df = make_clients()
    df.loc[0, [f"PAY_{m}" for m in range(1, 7)]] = [2, 0, 1, -1, -2, 3]
    out = prepare(df)
    assert set(ENGINEERED) <= set(out.columns)
    assert out.loc[0, "months_late"] == 3
    assert out.loc[0, "max_months_late"] == 3
    assert out["payment_ratio_1"].le(1.5).all()


def test_payment_ratio_uses_the_previous_statement(make_clients) -> None:
    df = make_clients(n=1)
    df.loc[0, ["PAY_AMT1", "BILL_AMT1", "BILL_AMT2"]] = [5_000, 40_000, 10_000]
    assert prepare(df).loc[0, "payment_ratio_1"] == 0.5


def test_preprocessor_can_exclude_sensitive_features(make_clients) -> None:
    X = prepare(make_clients())
    names = build_preprocessor(exclude=SENSITIVE).fit(X).get_feature_names_out()
    assert not any(n.startswith(("SEX", "MARRIAGE", "AGE")) for n in names)
