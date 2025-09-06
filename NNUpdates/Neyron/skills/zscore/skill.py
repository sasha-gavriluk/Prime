"""
zscore skill
------------
Z-score для вхідної колонки, з опціональним лог-перетворенням і кліпом.

Параметри:
- window: int = 32
- column: str = "close"
- clip: float = 4.0  # обрізання |z|, якщо >0
- log: bool = False  # брати log(column)
- center: str = "sma"  # або "ema"
- adjust: bool = False
- eps: float = 1e-12
- out_col: Optional[str] = None

Повертає: df з колонкою zscore.
"""
from typing import Optional
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame, window: int = 32, column: str = "close",
              clip: float = 4.0, log: bool = False, center: str = "sma",
              adjust: bool = False, eps: float = 1e-12, out_col: Optional[str] = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if window <= 1:
        raise ValueError("window must be > 1")

    res = df.copy()
    x = res[column].astype(float)
    if log:
        x = np.log(x.clip(min=eps))

    if center.lower() == "ema":
        mean = x.ewm(span=window, adjust=adjust, min_periods=window).mean()
    else:
        mean = x.rolling(window=window, min_periods=window).mean()
    std = x.rolling(window=window, min_periods=window).std()

    z = (x - mean) / (std + eps)
    if clip and clip > 0:
        z = z.clip(lower=-clip, upper=clip)

    out = out_col or f"zscore_{window}"
    res[out] = z.ffill().fillna(0.0)
    return res