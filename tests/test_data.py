import pandas as pd
import pytest

from credit_default_risk.artifacts import sha256
from credit_default_risk.data import RAW_TARGET, rename_columns, verify_sha256
from credit_default_risk.split import make_split


def test_rename_columns_maps_uci_names() -> None:
    raw = pd.DataFrame({"ID": [1, 2], "PAY_0": [0, 2], "PAY_2": [0, 1], RAW_TARGET: [0, 1]})
    out = rename_columns(raw, "default")
    assert list(out.columns) == ["PAY_1", "PAY_2", "default"]
    assert out.index.name == "ID"


def test_rename_columns_fails_loudly_on_a_changed_layout() -> None:
    raw = pd.DataFrame({"ID": [1], "PAY_1": [0], RAW_TARGET: [0]})  # no PAY_0
    with pytest.raises(ValueError, match="PAY_0"):
        rename_columns(raw, "default")


def test_verify_sha256(tmp_path) -> None:
    path = tmp_path / "raw.xls"
    path.write_bytes(b"data")
    verify_sha256(path, sha256(path))
    with pytest.raises(ValueError, match="pins"):
        verify_sha256(path, "0" * 64)


def test_split_is_disjoint_complete_and_stratified(make_clients) -> None:
    df = make_clients(n=1000, seed=8).rename_axis("ID")
    train_ids, test_ids = make_split(df, "default", test_size=0.2, seed=42)
    assert train_ids.intersection(test_ids).empty
    assert len(train_ids) + len(test_ids) == len(df)
    assert len(test_ids) == 200
    rate = df["default"].mean()
    assert abs(df.loc[train_ids, "default"].mean() - rate) < 0.01
    assert abs(df.loc[test_ids, "default"].mean() - rate) < 0.01


def test_split_is_reproducible(make_clients) -> None:
    df = make_clients(n=500, seed=9)
    a = make_split(df, "default", 0.2, seed=42)
    b = make_split(df, "default", 0.2, seed=42)
    assert all(x.equals(y) for x, y in zip(a, b, strict=True))
