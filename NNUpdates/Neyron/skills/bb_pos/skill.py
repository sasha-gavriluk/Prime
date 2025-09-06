"""
bb_pos skill
------------
Позиція ціни в смугах Боллінджера: (close - MA) / (k * std).

Параметри:
- window: int = 20
- std_mult: float = 2.0  # k
- column: str = "close"
- ma_type: str = "sma"  # або "ema"
- adjust: bool = False  # для EMA
- eps: float = 1e-8
- out_col: Optional[str] = None

Повертає: df з колонкою bb_pos.
"""
from typing import Optional
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame, window: int = 20, std_mult: float = 2.0,
              column: str = "close", ma_type: str = "sma", adjust: bool = False,
              eps: float = 1e-8, out_col: Optional[str] = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if window <= 1:
        raise ValueError("window must be > 1")

    res = df.copy()
    x = res[column].astype(float)
    if ma_type.lower() == "ema":
        ma = x.ewm(span=window, adjust=adjust, min_periods=window).mean()
    else:
        ma = x.rolling(window=window, min_periods=window).mean()
    std = x.rolling(window=window, min_periods=window).std()
    denom = (std_mult * std).abs() + eps
    bbp = (x - ma) / denom

    out = out_col or f"bb_pos_{window}"
    res[out] = bbp.ffill().fillna(0.0)
    return res