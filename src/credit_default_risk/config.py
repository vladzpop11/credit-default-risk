"""Load the project configuration and resolve paths relative to the project root."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative
