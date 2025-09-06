"""
roc_norm skill
--------------
Нормалізований темп зміни (ROC) відносно поточної волатильності.

Параметри:
- window: int = 12 — період ROC
- column: str = "close"
- normalize_by: str = "atr"  # або "std"
- atr_window: int = 14
- std_window: int = 20
- eps: float = 1e-8
- out_col: Optional[str] = None
- clip: Optional[float] = None  # напр., 10.0

Повертає: df з колонкою roc_norm.
"""
from typing import Optional
import numpy as np
import pandas as pd


def _atr_like(s: pd.Series, window: int) -> pd.Series:
    # спрощена ATR з close, якщо H/L недоступні
    tr = (s - s.shift(1)).abs()
    return tr.rolling(window=window, min_periods=window).mean()


def transform(df: pd.DataFrame, window: int = 12, column: str = "close",
              normalize_by: str = "atr", atr_window: int = 14, std_window: int = 20,
              eps: float = 1e-8, out_col: Optional[str] = None, clip: Optional[float] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if window <= 0:
        raise ValueError("window must be > 0")

    res = df.copy()
    price = res[column].astype(float)
    roc = price.pct_change(periods=window)

    if normalize_by.lower() == "atr":
        # намагаємось взяти повний ATR, якщо є 'high'/'low'
        if {"high", "low"}.issubset(res.columns):
            high = res["high"].astype(float)
            low = res["low"].astype(float)
            prev_close = price.shift(1)
            tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            vol = tr.rolling(window=atr_window, min_periods=atr_window).mean() / (price.abs() + eps)
        else:
            vol = _atr_like(price, atr_window) / (price.abs() + eps)
    else:
        # std of simple returns
        rets = price.pct_change()
        vol = rets.rolling(window=std_window, min_periods=std_window).std()

    denom = (vol.abs() + eps)
    rn = roc / denom
    if clip is not None and clip > 0:
        rn = rn.clip(lower=-clip, upper=clip)
    out = out_col or f"roc_norm_{window}"
    res[out] = rn.ffill().fillna(0.0)
    return res