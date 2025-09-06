import numpy as np
import pandas as pd

def transform(df: pd.DataFrame, tf: str = "", out_col: str = "engulfing") -> pd.DataFrame:
    out = df.copy()
    open_col, close_col, high_col, low_col = 'open', 'close', 'high', 'low'
    prev_open = out[open_col].shift(1)
    prev_close = out[close_col].shift(1)
    bullish = (prev_close < prev_open) & (out[close_col] > out[open_col]) & (out[close_col] >= prev_open) & (out[open_col] <= prev_close)
    bearish = (prev_close > prev_open) & (out[close_col] < out[open_col]) & (out[close_col] <= prev_open) & (out[open_col] >= prev_close)
    out[out_col] = np.where(bullish, 1, np.where(bearish, -1, 0))
    return out