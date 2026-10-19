"""Split the data once into stratified train/test sets and save the client IDs.

The test IDs are not used again until the final evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_risk.config import load_config, project_path
from credit_default_risk.data import load_clients, verify_sha256


def make_split(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> tuple[pd.Index, pd.Index]:
    """Stratified train/test client IDs, sorted."""
    train_ids, test_ids = train_test_split(
        df.index, test_size=test_size, stratify=df[target], random_state=seed
    )
    return train_ids.sort_values(), test_ids.sort_values()


def load_split(name: str, cfg: dict | None = None) -> pd.DataFrame:
    """Return the rows of the `train` or `test` split."""
    cfg = cfg or load_config()
    ids = pd.read_csv(project_path(cfg["data"]["processed_dir"]) / f"{name}_ids.csv")["ID"]
    return load_clients(cfg).loc[ids]


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    df = load_clients(cfg)
    splits = make_split(df, target, cfg["data"]["test_size"], cfg["seed"])

    out_dir = project_path(cfg["data"]["processed_dir"])
    for name, ids in zip(["train", "test"], splits, strict=True):
        path = out_dir / f"{name}_ids.csv"
        pd.Series(ids, name="ID").to_csv(path, index=False)
        verify_sha256(path, cfg["data"]["split_sha256"][name])
        print(f"{name}: {len(ids)} rows, default rate {df.loc[ids, target].mean():.4f}")
    print("split SHA-256 verified")


if __name__ == "__main__":
    main()
