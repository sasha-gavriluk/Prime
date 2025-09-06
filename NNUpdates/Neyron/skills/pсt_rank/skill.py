"""
pct_rank skill
---------------
Перцентильна позиція поточного значення відносно rolling-вікна (0..1).

Параметри:
- window: int = 100
- column: str = "close"
- out_col: Optional[str] = None

Повертає: df з колонкою pct_rank.
"""
from typing import Optional
import numpy as np
import pandas as pd


def _window_pct_rank(arr: np.ndarray) -> float:
    n = arr.size
    if n <= 1 or not np.isfinite(arr[-1]):
        return np.nan
    x = arr[-1]
    # включаємо поточне значення; віднімаємо 1, щоб не отримувати 1.0 завжди при рівності самому собі
    le = np.sum(arr <= x) - 1
    return le / max(n - 1, 1)


def transform(df: pd.DataFrame, window: int = 100, column: str = "close",
              out_col: Optional[str] = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if window <= 1:
        raise ValueError("window must be > 1")

    res = df.copy()
    x = res[column].astype(float).to_numpy()
    pr = pd.Series(x).rolling(window=window, min_periods=window).apply(_window_pct_rank, raw=True)

    out = out_col or f"pct_rank_{window}"
    res[out] = pr.ffill().fillna(0.0)
    return res