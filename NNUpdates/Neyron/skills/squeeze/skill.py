"""
squeeze skill
-------------
Bollinger-width percentile режим: коли ширина смуг Боллінджера мала (нижні перцентилі), вважаємо, що ринок у "squeeze".

Параметри:
- bb_win: int = 20
- bb_std: float = 2.0
- atr_win: int = 14  # необов'язково, для довідки/можливих розширень
- lookback: int = 252  # вікно для перцентиля ширини
- threshold_pct: float = 10.0  # поріг перцентиля (0..100)
- column: str = "close"
- ma_type: str = "sma"  # або "ema"
- adjust: bool = False
- eps: float = 1e-12
- out_col: Optional[str] = None

Повертає: df з колонкою squeeze (0/1) та допоміжною bb_width.
"""
from typing import Optional
import numpy as np
import pandas as pd


def _pct_rank_last(arr: np.ndarray) -> float:
    n = arr.size
    if n <= 1 or not np.isfinite(arr[-1]):
        return np.nan
    x = arr[-1]
    le = np.sum(arr <= x) - 1
    return le / max(n - 1, 1)


def transform(df: pd.DataFrame, bb_win: int = 20, bb_std: float = 2.0, atr_win: int = 14,
              lookback: int = 252, threshold_pct: float = 10.0,
              column: str = "close", ma_type: str = "sma", adjust: bool = False,
              eps: float = 1e-12, out_col: Optional[str] = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if bb_win <= 1 or lookback <= 1:
        raise ValueError("bb_win and lookback must be > 1")

    res = df.copy()
    x = res[column].astype(float)

    if ma_type.lower() == "ema":
        ma = x.ewm(span=bb_win, adjust=adjust, min_periods=bb_win).mean()
    else:
        ma = x.rolling(window=bb_win, min_periods=bb_win).mean()
    std = x.rolling(window=bb_win, min_periods=bb_win).std()

    upper = ma + bb_std * std
    lower = ma - bb_std * std
    width = (upper - lower) / (ma.abs() + eps)  # відносна ширина

    width_pct = pd.Series(width).rolling(window=lookback, min_periods=lookback).apply(_pct_rank_last, raw=True)
    thr = (threshold_pct / 100.0)
    squeeze = (width_pct <= thr).astype(float)

    res["bb_width"] = pd.Series(width).ffill().fillna(0.0)
    out = out_col or f"squeeze_{bb_win}_{int(threshold_pct)}"
    res[out] = squeeze.ffill().fillna(0.0)
    return res