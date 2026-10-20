"""Download the evaluated model from the GitHub release and verify it before use.

The downloaded file must match the SHA-256 recorded in reports/test_metrics.json; any
other file is deleted without being unpickled. The evaluated model's training summary
(reports/train_summary.json) is copied next to it, so the local records describe it.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

from credit_default_risk import artifacts
from credit_default_risk.config import load_config


def fetch(url: str, dest: Path, expected_sha256: str, timeout: int = 60) -> None:
    """Download `url` to `dest` only if its SHA-256 matches; never leave a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".download")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        actual = artifacts.sha256(tmp)
        if actual != expected_sha256:
            raise ValueError(
                f"{url} has SHA-256 {actual}, but the evaluated model is {expected_sha256}; "
                "refusing to use it"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    cfg = load_config()
    expected = artifacts.evaluated_model_sha(cfg)
    summary = artifacts.read_json(artifacts.evaluated_train_summary_path(cfg))
    if expected is None or summary is None or summary["model_sha256"] != expected:
        sys.exit("reports/ does not record an evaluated model to fetch")

    dest = artifacts.model_path(cfg)
    if dest.exists() and artifacts.sha256(dest) == expected:
        print(f"{dest} is already the evaluated model")
    else:
        if dest.exists():
            print(f"replacing the local model (a retrain) in {dest} with the evaluated model")
        fetch(cfg["release"]["model_url"], dest, expected)
        print(f"downloaded {cfg['release']['model_url']}")
    artifacts.write_json(artifacts.local_train_summary_path(cfg), summary)
    print(f"SHA-256 verified: {expected[:16]}… (reports/test_metrics.json)")


if __name__ == "__main__":
    main()
