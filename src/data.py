"""Dataset loading and cleaning (missing values, duplicates, outliers)."""

from pathlib import Path

import pandas as pd

from .config import OUTLIER_COLUMNS, RAW_DIR


def find_dataset() -> Path:
    """Return the first CSV placed in data/raw/."""
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No dataset found in {RAW_DIR}. Download "
            "'Student Habits and Academic Performance Dataset' from Kaggle "
            "(aryan208/student-habits-and-academic-performance-dataset) and "
            "place the CSV there."
        )
    return candidates[0]


def load_raw() -> pd.DataFrame:
    path = find_dataset()
    df = pd.read_csv(path)
    return df


def missing_report(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    return {col: int(n) for col, n in missing.items() if n > 0}


def remove_outliers(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Drop rows outside the 1.5 * IQR fences of the given numeric columns."""
    columns = [c for c in (columns or OUTLIER_COLUMNS) if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for col in columns:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[col].isna() | df[col].between(low, high)
    return df[mask]


def clean(df: pd.DataFrame) -> tuple:
    """Standard cleaning: exact duplicates and outlier rows are removed.

    Remaining missing values are kept as NaN and imputed inside the model
    pipelines, so inference time inputs may also contain blanks.
    """
    before = len(df)
    df = df.drop_duplicates()
    deduped = len(df)
    df = remove_outliers(df)
    after = len(df)
    stats = {
        "rows_raw": before,
        "duplicates_removed": before - deduped,
        "outliers_removed": deduped - after,
        "rows_clean": after,
        "missing_values": missing_report(df),
    }
    return df.reset_index(drop=True), stats


def risk_label(probability: float, bands: dict) -> str:
    if probability >= bands["High"]:
        return "High"
    if probability >= bands["Medium"]:
        return "Medium"
    return "Low"
