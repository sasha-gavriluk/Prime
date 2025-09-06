"""
ATR skill
---------
Рахує Average True Range (ATR). Переважно потребує 'high','low','close'.
Якщо high/low відсутні, можна приблизити через абсолютну зміну close.

Параметри:
- window: int = 14
- high_col: str = "high"
- low_col: str = "low"
- close_col: str = "close"
- fallback_with_close: bool = True — якщо немає high/low, використати |close-close.shift(1)|
- out_col: Optional[str] = None

Повертає: df з колонкою ATR (у тих же одиницях, що й ціна).
"""
from typing import Optional
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame, window: int = 14,
              high_col: str = "high", low_col: str = "low", close_col: str = "close",
              fallback_with_close: bool = True, out_col: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found")
    has_hl = (high_col in df.columns) and (low_col in df.columns)
    if not has_hl and not fallback_with_close:
        raise ValueError("'high'/'low' not found and fallback_with_close=False")
    if not isinstance(window, int) or window <= 1:
        raise ValueError("window must be int > 1")

    res = df.copy()
    out = out_col or f"atr_{window}"
    if has_hl:
        high = res[high_col].astype(float)
        low = res[low_col].astype(float)
        close = res[close_col].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
    else:
        c = res[close_col].astype(float)
        tr = (c - c.shift(1)).abs()

    atr = tr.rolling(window=window, min_periods=window).mean()
    res[out] = atr.ffill().fillna(0.0)
    return res