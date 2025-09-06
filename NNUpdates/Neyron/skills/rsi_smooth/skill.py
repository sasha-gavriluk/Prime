"""
rsi_smooth skill
----------------
RSI з классичним Wilder EMA та додатковим EMA-згладжуванням; опційно нормалізуємо до [-1,1].

Параметри:
- window: int = 14
- ema: int = 5  # 0 → без додаткового згладжування
- column: str = "close"
- normalize: bool = True  # якщо True, (rsi-50)/50 → [-1,1]
- out_col: Optional[str] = None
- eps: float = 1e-12

Повертає: df з колонкою rsi_smooth.
"""
from typing import Optional
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame, window: int = 14, ema: int = 5, column: str = "close",
              normalize: bool = True, out_col: Optional[str] = None, eps: float = 1e-12) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if window <= 1:
        raise ValueError("window must be > 1")

    res = df.copy()
    c = res[column].astype(float)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder smoothing (EWMA з alpha=1/window)
    gain = up.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = down.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = gain / (loss + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi_s = rsi if ema <= 0 else rsi.ewm(span=ema, adjust=False, min_periods=max(ema, 1)).mean()
    if normalize:
        rsi_s = (rsi_s - 50.0) / 50.0  # ~[-1,1]

    out = out_col or (f"rsi_smooth_{window}_{ema}" if ema > 0 else f"rsi_{window}")
    res[out] = rsi_s.ffill().fillna(0.0)
    return res