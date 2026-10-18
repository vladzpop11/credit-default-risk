"""Feature preparation shared by training and serving.

`clean` and `add_features` are row-wise and stateless, so they cannot leak information
between rows. Everything learned from data (imputation, scaling, encoding) lives in the
scikit-learn pipeline built by `build_preprocessor`, which is fitted on training folds only.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MONTHS = range(1, 7)
PAY_STATUS = [f"PAY_{m}" for m in MONTHS]
BILL_AMTS = [f"BILL_AMT{m}" for m in MONTHS]
PAY_AMTS = [f"PAY_AMT{m}" for m in MONTHS]

SENSITIVE = ["SEX", "MARRIAGE", "AGE"]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
AMOUNTS = ["LIMIT_BAL", *BILL_AMTS, *PAY_AMTS]
ENGINEERED = ["utilization_1", "payment_ratio_1", "months_late", "max_months_late"]
RAW_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS,
    *BILL_AMTS,
    *PAY_AMTS,
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Merge undocumented category codes into the documented "other" categories."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["utilization_1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    # Share of the previous statement that was paid. The payment in month m settles the
    # bill of month m+1: PAY_AMT1 equals BILL_AMT2 exactly for 15.2% of clients with a
    # positive bill, against 2.4% for BILL_AMT1 (training split). Clipped because bills
    # can be ~0 or negative. When there was no previous bill (BILL_AMT2 <= 0, 10.4% of
    # clients) this is just PAY_AMT1 clipped, usually 0, the same value as "had a bill and
    # paid nothing"; the trees can tell the two apart through BILL_AMT2. An explicit
    # no-previous-bill flag changed CV PR-AUC by +0.0011 ± 0.0016, so it was not adopted.
    df["payment_ratio_1"] = (df["PAY_AMT1"] / df["BILL_AMT2"].clip(lower=1)).clip(upper=1.5)
    late = df[PAY_STATUS].clip(lower=0)
    df["months_late"] = (late > 0).sum(axis=1)
    df["max_months_late"] = late.max(axis=1)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw client rows -> model input rows. Used identically in training and serving.

    Columns the model does not use (the excluded sensitive attributes) may be absent;
    they are filled with NaN and dropped by the preprocessor.
    """
    return add_features(clean(df.reindex(columns=RAW_FEATURES)))


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def feature_columns(exclude: list[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """Return (categorical, amount, other numeric) column lists, minus `exclude`."""
    exclude = set(exclude or [])
    categorical = [c for c in CATEGORICAL if c not in exclude]
    amounts = [c for c in AMOUNTS if c not in exclude]
    numeric = [c for c in ["AGE", *PAY_STATUS, *ENGINEERED] if c not in exclude]
    return categorical, amounts, numeric


def build_preprocessor(exclude: list[str] | None = None) -> ColumnTransformer:
    categorical, amounts, numeric = feature_columns(exclude)
    amount_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("amounts", amount_pipe, amounts),
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        verbose_feature_names_out=False,
    )
