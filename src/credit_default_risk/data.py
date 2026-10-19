"""Download the UCI "Default of Credit Card Clients" dataset and convert it to CSV.

The downloaded file must match the SHA-256 pinned in configs/config.yaml, so a changed
upstream file cannot silently change the data, the split or the results.
"""

import io
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_default_risk.artifacts import sha256
from credit_default_risk.config import load_config, project_path

RAW_TARGET = "default payment next month"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path} has SHA-256 {actual}, but configs/config.yaml pins {expected}. "
            "The file is not the one the recorded results were produced from; restore it, "
            "or update the pin knowingly."
        )


def download(url: str, raw_dir: Path, raw_file: str, expected_sha256: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls_path = raw_dir / raw_file
    if not xls_path.exists():
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extract(raw_file, raw_dir)
        (raw_dir / "SOURCE.md").write_text(
            f"# Source\n\n"
            f"- Dataset: Default of Credit Card Clients (Yeh & Lien, 2009)\n"
            f"- Page: {DATASET_PAGE}\n"
            f"- Download URL: {url}\n"
            f"- License: CC BY 4.0\n"
            f"- Downloaded (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"- File: {raw_file}\n"
            f"- SHA-256: {sha256(xls_path)}\n"
        )
    verify_sha256(xls_path, expected_sha256)
    return xls_path


def rename_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """The UCI file names the most recent repayment status PAY_0; this project calls it PAY_1,
    matching BILL_AMT1 and PAY_AMT1 (all September 2005)."""
    expected = {"ID", RAW_TARGET, "PAY_0"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"unexpected raw file layout, missing columns {sorted(missing)}")
    return df.rename(columns={RAW_TARGET: target, "PAY_0": "PAY_1"}).set_index("ID")


def read_raw(xls_path: Path, target: str) -> pd.DataFrame:
    # Row 0 holds generic X1..X23 labels; the real column names are on row 1.
    return rename_columns(pd.read_excel(xls_path, header=1), target)


def load_clients(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(project_path(cfg["data"]["processed_dir"]) / "clients.csv", index_col="ID")


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    xls_path = download(d["url"], project_path(d["raw_dir"]), d["raw_file"], d["sha256"])
    df = read_raw(xls_path, d["target"])

    out = project_path(d["processed_dir"]) / "clients.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"{len(df)} rows x {df.shape[1]} columns -> {out} (SHA-256 verified)")
    print(f"default rate: {df[d['target']].mean():.4f}")


if __name__ == "__main__":
    main()
