"""Checks on the real data and split. Skipped until `make data split` has run (CI runs it)."""

import pandas as pd
import pytest

from credit_default_risk.artifacts import sha256
from credit_default_risk.config import load_config, project_path

CFG = load_config()
PROCESSED = project_path(CFG["data"]["processed_dir"])

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "test_ids.csv").exists(), reason="real data not downloaded (make data split)"
)


def test_split_files_match_the_pinned_split() -> None:
    for name, expected in CFG["data"]["split_sha256"].items():
        assert sha256(PROCESSED / f"{name}_ids.csv") == expected


def test_split_sizes_rates_and_disjointness() -> None:
    clients = pd.read_csv(PROCESSED / "clients.csv", index_col="ID")
    train = pd.read_csv(PROCESSED / "train_ids.csv")["ID"]
    test = pd.read_csv(PROCESSED / "test_ids.csv")["ID"]
    assert (len(clients), len(train), len(test)) == (30_000, 24_000, 6_000)
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(clients.index)
    target = CFG["data"]["target"]
    for ids in (train, test):
        assert clients.loc[ids, target].mean() == pytest.approx(0.2212, abs=0.0005)
