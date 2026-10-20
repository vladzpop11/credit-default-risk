import pytest

from credit_default_risk.artifacts import sha256
from credit_default_risk.fetch_model import fetch


@pytest.fixture
def published(tmp_path):
    source = tmp_path / "release" / "model.joblib"
    source.parent.mkdir()
    source.write_bytes(b"the evaluated model")
    return source


def test_fetch_installs_a_file_with_the_expected_hash(published, tmp_path) -> None:
    dest = tmp_path / "models" / "model.joblib"
    fetch(published.as_uri(), dest, sha256(published))
    assert dest.read_bytes() == b"the evaluated model"
    assert list(dest.parent.iterdir()) == [dest]  # no temporary file left behind


def test_fetch_refuses_a_different_file_and_keeps_the_old_one(published, tmp_path) -> None:
    dest = tmp_path / "models" / "model.joblib"
    dest.parent.mkdir()
    dest.write_bytes(b"local retrain")
    with pytest.raises(ValueError, match="refusing to use it"):
        fetch(published.as_uri(), dest, "0" * 64)
    assert dest.read_bytes() == b"local retrain"
    assert list(dest.parent.iterdir()) == [dest]


def test_fetch_leaves_nothing_when_the_download_fails(tmp_path) -> None:
    dest = tmp_path / "models" / "model.joblib"
    with pytest.raises(OSError):
        fetch((tmp_path / "missing.joblib").as_uri(), dest, "0" * 64)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []
