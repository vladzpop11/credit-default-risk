from credit_default_risk import __version__
from credit_default_risk.config import load_config


def test_package_imports() -> None:
    assert __version__


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg["seed"] == 42
    assert 0 < cfg["data"]["test_size"] < 1
