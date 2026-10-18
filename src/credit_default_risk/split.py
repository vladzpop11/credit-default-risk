"""Split the data once into stratified train/test sets and save the client IDs.

The test IDs are not used again until the final evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_risk.config import load_config, project_path
from credit_default_risk.data import load_clients


def load_split(name: str, cfg: dict | None = None) -> pd.DataFrame:
    """Return the rows of the `train` or `test` split."""
    cfg = cfg or load_config()
    ids = pd.read_csv(project_path(cfg["data"]["processed_dir"]) / f"{name}_ids.csv")["ID"]
    return load_clients(cfg).loc[ids]


def main() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    df = load_clients(cfg)

    train_ids, test_ids = train_test_split(
        df.index,
        test_size=cfg["data"]["test_size"],
        stratify=df[target],
        random_state=cfg["seed"],
    )
    out_dir = project_path(cfg["data"]["processed_dir"])
    for name, ids in [("train", train_ids), ("test", test_ids)]:
        pd.Series(sorted(ids), name="ID").to_csv(out_dir / f"{name}_ids.csv", index=False)
        rate = df.loc[ids, target].mean()
        print(f"{name}: {len(ids)} rows, default rate {rate:.4f}")


if __name__ == "__main__":
    main()
