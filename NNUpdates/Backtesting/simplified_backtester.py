# Файл: NNUpdates/Backtesting/simplified_backtester.py

import numpy as np
import pandas as pd


class SimplifiedBacktester:
    """
    Надшвидкий, спрощений бектестер для RL-середовища.
    Працює в пам'яті з уже обробленими даними (мінімум: колонка 'close').
    Ідея: на кожному кроці робимо 1 агреговану позицію на слайсі,
    розмір якої визначається тіньовими налаштуваннями (risk, leverage, ваги).
    """

    def __init__(self, processed_data: pd.DataFrame, fee_pct: float = 0.0005, slip_pct: float = 0.0002):
        """
        :param processed_data: DataFrame з мінімумом колонок: ['close'] (інше опціонально).
        :param fee_pct: комісія за одну сторону (entry або exit), у частках (0.0005 = 5 bps).
        :param slip_pct: проскальзування в частках на транзакцію.
        """
        if 'close' not in processed_data.columns:
            raise ValueError("processed_data must contain 'close' column.")
        self.data = processed_data.reset_index(drop=True)
        self.fee_pct = float(fee_pct)
        self.slip_pct = float(slip_pct)

    def run_slice(self, start_idx: int, end_idx: int, shadow_settings: dict, equity: float) -> dict:
        """
        Проганяє симуляцію на відрізку даних [start_idx:end_idx).

        :param start_idx: Початковий індекс слайсу (включно).
        :param end_idx: Кінцевий індекс слайсу (невключно).
        :param shadow_settings: Тіньові налаштування агента (risk_pct, leverage, ваги).
        :param equity: Поточний капітал портфеля (для розміру позиції/комісій).
        :return: dict: pnl, gross_pnl, fees, slippage, num_trades, signal_confidence, market_state
        """
        # Захист від виходу за межі
        end_idx = min(end_idx, len(self.data))
        if end_idx - start_idx < 2:
            return {
                "pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "slippage": 0.0,
                "num_trades": 0, "signal_confidence": 0.0, "market_state": 0.0
            }

        df = self.data.iloc[start_idx:end_idx]
        close_start = float(df['close'].iloc[0])
        close_end = float(df['close'].iloc[-1])

        # Базові ряди для оцінок на слайсі
        rets = df['close'].pct_change().dropna()
        vol = float(rets.std()) if len(rets) else 0.0
        drift = float(rets.mean()) if len(rets) else 0.0

        # Узагальнені оцінки (без «підглядання далі слайсу»)
        eps = 1e-12
        sharpe_like = drift / (vol + eps)
        signal_confidence = float(np.tanh(sharpe_like))  # у [-1, 1]

        # Оцінка «стану ринку» ~ рівень волатильності, теж у [-1,1]
        vol_series = rets.rolling(max(2, len(rets)//3)).std().dropna()
        if len(vol_series) >= 5:
            q1, q3 = np.quantile(vol_series, [0.25, 0.75])
            iqr = max(q3 - q1, eps)
            market_state = float(np.clip((vol - (q1 + q3) / 2) / (iqr / 1.349 + eps), -3, 3) / 3.0)
        else:
            market_state = float(np.tanh(vol * 50.0))

        # Параметри позиції від налаштувань
        risk_pct = float(shadow_settings.get('default_risk_per_trade_pct', 1.0)) / 100.0  # очікуємо у %
        leverage = int(shadow_settings.get('leverage', 1))
        leverage = max(1, min(leverage, 50))

        # Нотіонал і напрямок позиції
        position_notional = equity * risk_pct * leverage
        direction = 1.0 if signal_confidence >= 0 else -1.0

        # Прибуток до витрат
        gross_return = ((close_end - close_start) / (close_start + eps)) * direction
        gross_pnl = position_notional * gross_return

        # Витрати: 1 вхід + 1 вихід
        trades = 1
        fees = trades * 2 * self.fee_pct * position_notional
        slippage = trades * 2 * self.slip_pct * position_notional

        pnl = gross_pnl - fees - slippage

        return {
            "pnl": float(pnl),
            "gross_pnl": float(gross_pnl),
            "fees": float(fees),
            "slippage": float(slippage),
            "num_trades": int(trades),
            "signal_confidence": float(signal_confidence),
            "market_state": float(market_state),
        }
