"""
EMA skill
---------
Додає експоненціальне ковзне середнє (EMA) для колонки `column`.

Параметри:
- window: int = 20 — період EMA (>1)
- column: str = "close" — вхідна колонка
- adjust: bool = False — як у pandas.ewm
- out_col: Optional[str] = None — назва вихідної колонки; якщо None → f"ema_{window}"

Повертає: df з доданою колонкою EMA. NaN на старті — заповнюються forward-fill, далі 0.
"""
from typing import Optional
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame, window: int = 20, column: str = "close",
              adjust: bool = False, out_col: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not isinstance(window, int) or window <= 1:
        raise ValueError("window must be int > 1")

    out = out_col or f"ema_{window}"
    res = df.copy()
    ema = res[column].ewm(span=window, adjust=adjust, min_periods=window).mean()
    res[out] = ema
    res[out] = res[out].ffill().fillna(0.0)
    return res