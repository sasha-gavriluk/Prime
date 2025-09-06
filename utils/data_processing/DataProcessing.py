import numpy as np
import pandas as pd

from scipy.signal import find_peaks

class IndicatorProcessor:
    def __init__(self, data: pd.DataFrame, processed_data: pd.DataFrame, indicators_params=None):
        self.data = data
        self.processed_data = processed_data
        self.indicators_params = indicators_params if indicators_params is not None else []

    def _get_unique_column_name(self, base_name: str) -> str:
        """Return a unique column name based on base_name.

        If a column with base_name already exists in processed_data, a numeric
        suffix is appended. This allows the same indicator to be added multiple
        times even with identical parameters.
        """
        if base_name not in self.processed_data.columns:
            return base_name
        counter = 1
        new_name = f"{base_name}_{counter}"
        while new_name in self.processed_data.columns:
            counter += 1
            new_name = f"{base_name}_{counter}"
        return new_name

    # Окремі методи для кожного індикатора
    def add_sma(self, period=20):
        column = self._get_unique_column_name(f'SMA_{period}')
        self.processed_data[column] = self.data['close'].rolling(window=period).mean()

    def add_ema(self, period=20):
        column = self._get_unique_column_name(f'EMA_{period}')
        self.processed_data[column] = self.data['close'].ewm(span=period, adjust=False).mean()

    def add_sma_cross(self, period_short=10, period_long=50, column='close'):
        sma_short = self.data[column].rolling(window=period_short).mean()
        sma_long = self.data[column].rolling(window=period_long).mean()
        cross = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)
        cross_column = self._get_unique_column_name(f"SMA_Cross_{period_short}_{period_long}")
        self.processed_data[cross_column] = cross

    def add_ema_cross(self, period_short=10, period_long=50, column='close'):
        ema_short = self.data[column].ewm(span=period_short, adjust=False).mean()
        ema_long = self.data[column].ewm(span=period_long, adjust=False).mean()
        cross = (ema_short > ema_long).astype(int) - (ema_short < ema_long).astype(int)
        cross_column = self._get_unique_column_name(f"EMA_Cross_{period_short}_{period_long}")
        self.processed_data[cross_column] = cross

    def add_rsi(self, period=14, column='close'):
        """
        Adds the RSI indicator to the data.

        Parameters:
        - period: int, default 14
        - column: str, default 'close'
        """
        delta = self.data[column].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        column = self._get_unique_column_name(f'RSI_{period}')
        self.processed_data[column] = rsi

    def add_macd(self, fast_period=12, slow_period=26, signal_period=9, column='close'):
        """
        Adds the MACD indicator to the data.

        Parameters:
        - fast_period: int, default 12
        - slow_period: int, default 26
        - signal_period: int, default 9
        - column: str, default 'close'
        """
        short_ema = self.data[column].ewm(span=fast_period, adjust=False).mean()
        long_ema = self.data[column].ewm(span=slow_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - signal

        # Save the MACD components with parameter-specific column names
        macd_column = self._get_unique_column_name(f'MACD_{fast_period}_{slow_period}_{signal_period}')
        signal_column = self._get_unique_column_name(f'MACD_Signal_{fast_period}_{slow_period}_{signal_period}')
        hist_column = self._get_unique_column_name(f'MACD_Hist_{fast_period}_{slow_period}_{signal_period}')

        self.processed_data[macd_column] = macd
        self.processed_data[signal_column] = signal
        self.processed_data[hist_column] = macd_hist

    def add_bollinger_bands(self, period=20, std_multiplier=2, column='close'):
        sma = self.data[column].rolling(window=period).mean()
        std = self.data[column].rolling(window=period).std()
        upper_band = sma + (std_multiplier * std)
        lower_band = sma - (std_multiplier * std)

        # Замінюємо крапку на підкреслення в std_multiplier
        std_multiplier_str = str(std_multiplier).replace('.', '_')

        upper_band_column = self._get_unique_column_name(f'Bollinger_Upper_{period}_{std_multiplier_str}')
        lower_band_column = self._get_unique_column_name(f'Bollinger_Lower_{period}_{std_multiplier_str}')
        middle_band_column = self._get_unique_column_name(f'Bollinger_Middle_{period}')

        self.processed_data[upper_band_column] = upper_band
        self.processed_data[lower_band_column] = lower_band
        self.processed_data[middle_band_column] = sma

    def add_stochastic(self, k_period=14, d_period=3, column='close'):
        """
        Додає індикатор Stochastic Oscillator до даних.

        Parameters:
        - k_period: int, період для %K лінії
        - d_period: int, період для %D лінії (сигнальна лінія)
        - column: str, колонка для розрахунку (зазвичай 'close')
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data[column]

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        stochastic_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        stochastic_d = stochastic_k.rolling(window=d_period).mean()

        k_column = self._get_unique_column_name(f'Stochastic_K_{k_period}')
        d_column = self._get_unique_column_name(f'Stochastic_D_{d_period}')

        self.processed_data[k_column] = stochastic_k
        self.processed_data[d_column] = stochastic_d

    def add_adx(self, period=14):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period).mean()

        column = self._get_unique_column_name(f'ADX_{period}')
        self.processed_data[column] = adx
        # print(f"ADX_{period} - Min: {adx.min()}, Mean: {adx.mean()}, Max: {adx.max()}")

    def add_atr(self, period=14):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        column = self._get_unique_column_name(f'ATR_{period}')
        self.processed_data[column] = atr  

    def add_williamsr(self, period=14):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        column = self._get_unique_column_name(f'WilliamsR_{period}')
        self.processed_data[column] = williams_r

    def add_cci(self, period=20):
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        column = self._get_unique_column_name(f'CCI_{period}')
        self.processed_data[column] = cci

    def add_keltner_channel(self, period=20, multiplier=2):
        """
        Обчислює Keltner Channel.
        Обчислює Keltner Channel без використання ATR.
        """
        typical_price = (self.processed_data['high'] + self.processed_data['low'] + self.processed_data['close']) / 3
        middle_band = typical_price.ewm(span=period, adjust=False).mean()

        # Обчислення середнього діапазону між High і Low
        high_low_range = self.processed_data['high'] - self.processed_data['low']
        average_range = high_low_range.rolling(window=period).mean()

        upper_band = middle_band + (multiplier * average_range)
        lower_band = middle_band - (multiplier * average_range)

        middle_column = self._get_unique_column_name(f'Keltner_Middle_{period}')
        upper_column = self._get_unique_column_name(f'Keltner_Upper_{period}')
        lower_column = self._get_unique_column_name(f'Keltner_Lower_{period}')

        self.processed_data[middle_column] = middle_band
        self.processed_data[upper_column] = upper_band
        self.processed_data[lower_column] = lower_band

    def add_volume_avg(self, period=20):
        column = self._get_unique_column_name(f'Volume_Avg_{period}')
        self.processed_data[column] = self.data['volume'].rolling(window=period).mean()

    def process_data(self):
        """Головна функція для запуску обробки індикаторів."""

        # Мапимо імена функцій на методи
        indicator_methods = {
            'SMA': self.add_sma,
            'EMA': self.add_ema,
            'SMA_Cross': self.add_sma_cross,
            'EMA_Cross': self.add_ema_cross,
            'RSI': self.add_rsi,
            'MACD': self.add_macd,
            'Bollinger_Bands': self.add_bollinger_bands,
            'Stochastic': self.add_stochastic,
            'WilliamsR': self.add_williamsr,
            'CCI': self.add_cci,
            'ADX': self.add_adx,
            'ATR': self.add_atr,
            'Keltner_Channel': self.add_keltner_channel,
            'Volume_Avg': self.add_volume_avg,
        }

        if self.indicators_params:
            # Якщо параметри передані
            for indicator in self.indicators_params:
                name = indicator.get('name')
                params = indicator.get('parameters', {})

                if name in indicator_methods:
                    indicator_methods[name](**params)
                else:
                    print(f"Індикатор '{name}' не підтримується.")
        else:
            # Якщо параметри не передані — запускаємо всі індикатори зі стандартними значеннями
            for method in indicator_methods.values():
                method()

        return self.processed_data

class PatternDetector:
    def __init__(self, data: pd.DataFrame, processed_data: pd.DataFrame , pattern_params=None):
        self.data = data
        self.processed_data = processed_data
        self.pattern_params = pattern_params if pattern_params is not None else []

    # Окремі методи для свічкових патернів
    def detect_hammer(self):
        body = np.abs(self.data['close'] - self.data['open'])
        shadow_lower = self.data['low'] - np.minimum(self.data['close'], self.data['open'])
        shadow_upper = self.data['high'] - np.maximum(self.data['close'], self.data['open'])
        condition = (shadow_lower >= 2 * body) & (shadow_upper <= body)
        self.processed_data['Hammer'] = condition.astype(int)

    def detect_inverted_hammer(self):
        """
        Виявляє патерн Inverted Hammer.
        """
        open_ = self.data['open']
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        body = abs(close - open_)
        upper_shadow = high - np.maximum(close, open_)
        lower_shadow = np.minimum(close, open_) - low

        condition = (
            (upper_shadow > 2 * body) &           # Довга верхня тінь
            (lower_shadow < body * 0.5) &         # Коротка або відсутня нижня тінь
            (close < open_)                       # Ведмежа свічка
        )

        self.processed_data['Inverted_Hammer'] = condition.astype(bool)
        # print(f"Патерн 'Inverted Hammer' виявлено: {condition.sum()} разів")

    def detect_shooting_star(self):
        body = np.abs(self.data['close'] - self.data['open'])
        shadow_upper = self.data['high'] - np.maximum(self.data['close'], self.data['open'])
        shadow_lower = np.minimum(self.data['close'], self.data['open']) - self.data['low']
        condition = (shadow_upper >= 2 * body) & (shadow_lower <= body)
        self.processed_data['Shooting_Star'] = condition.astype(int)

    def detect_engulfing(self):
        prev_close = self.data['close'].shift(1)
        prev_open = self.data['open'].shift(1)
        current_close = self.data['close']
        current_open = self.data['open']

        bullish = ((current_close > current_open) & (prev_close < prev_open) &
                   (current_close >= prev_open) & (current_open <= prev_close))
        bearish = ((current_close < current_open) & (prev_close > prev_open) &
                   (current_close <= prev_open) & (current_open >= prev_close))

        self.processed_data['Engulfing'] = np.where(bullish, 1, np.where(bearish, -1, 0))

    def detect_morning_star(self):
        prev2_close = self.data['close'].shift(2)
        prev2_open = self.data['open'].shift(2)
        prev1_close = self.data['close'].shift(1)
        prev1_open = self.data['open'].shift(1)
        current_close = self.data['close']
        current_open = self.data['open']

        condition = ((prev2_close < prev2_open) &
                     (prev1_close < prev1_open) &
                     (current_close > current_open) &
                     (current_close > prev2_open))

        self.processed_data['Morning_Star'] = condition.astype(int)

    def detect_evening_star(self):
        prev2_close = self.data['close'].shift(2)
        prev2_open = self.data['open'].shift(2)
        prev1_close = self.data['close'].shift(1)
        prev1_open = self.data['open'].shift(1)
        current_close = self.data['close']
        current_open = self.data['open']

        condition = ((prev2_close > prev2_open) &
                     (prev1_close > prev1_open) &
                     (current_close < current_open) &
                     (current_close < prev2_open))

        self.processed_data['Evening_Star'] = condition.astype(int)

    def detect_piercing_pattern(self):
        prev_close = self.data['close'].shift(1)
        prev_open = self.data['open'].shift(1)
        current_close = self.data['close']
        current_open = self.data['open']

        condition = ((prev_close < prev_open) &
                     (current_close > current_open) &
                     (current_close > (prev_close + prev_open)/2) &
                     (current_open < prev_close))

        self.processed_data['Piercing_Pattern'] = condition.astype(int)

    def detect_dark_cloud_cover(self):
        prev_close = self.data['close'].shift(1)
        prev_open = self.data['open'].shift(1)
        current_close = self.data['close']
        current_open = self.data['open']

        condition = ((prev_close > prev_open) &
                     (current_close < current_open) &
                     (current_close < (prev_close + prev_open)/2) &
                     (current_open > prev_close))

        self.processed_data['Dark_Cloud_Cover'] = condition.astype(int)

    def detect_three_white_soldiers(self):
        close = self.data['close']
        open_ = self.data['open']

        condition = ((close > open_) &
                     (close.shift(1) > open_.shift(1)) &
                     (close.shift(2) > open_.shift(2)) &
                     (close > close.shift(1)) &
                     (close.shift(1) > close.shift(2)))

        self.processed_data['Three_White_Soldiers'] = condition.astype(int)
        # print(f"Патерн 'Three_White_Soldiers' виявлено: {condition.sum()} разів")

    def detect_three_black_crows(self):
        close = self.data['close']
        open_ = self.data['open']

        condition = ((close < open_) &
                     (close.shift(1) < open_.shift(1)) &
                     (close.shift(2) < open_.shift(2)) &
                     (close < close.shift(1)) &
                     (close.shift(1) < close.shift(2)))

        self.processed_data['Three_Black_Crows'] = condition.astype(int)
        # print(f"Патерн 'Three_Black_Crows' виявлено: {condition.sum()} разів")

    def detect_hanging_man(self):
        body = np.abs(self.data['close'] - self.data['open'])
        shadow_lower = self.data['low'] - np.minimum(self.data['close'], self.data['open'])
        shadow_upper = self.data['high'] - np.maximum(self.data['close'], self.data['open'])

        condition = (shadow_lower >= 2 * body) & (shadow_upper <= body) & (body / (self.data['high'] - self.data['low']) >= 0.3)

        self.processed_data['Hanging_Man'] = condition.astype(int)

    def process_data(self):
        pattern_methods = {
            'Hammer': self.detect_hammer,
            'Inverted_Hammer': self.detect_inverted_hammer,
            'Shooting_Star': self.detect_shooting_star,
            'Engulfing': self.detect_engulfing,
            'Morning_Star': self.detect_morning_star,
            'Evening_Star': self.detect_evening_star,
            'Piercing_Pattern': self.detect_piercing_pattern,
            'Dark_Cloud_Cover': self.detect_dark_cloud_cover,
            'Three_White_Soldiers': self.detect_three_white_soldiers,
            'Three_Black_Crows': self.detect_three_black_crows,
            'Hanging_Man': self.detect_hanging_man,
        }

        if self.pattern_params:
            for pattern in self.pattern_params:
                name = pattern if isinstance(pattern, str) else pattern.get('name')
                if name in pattern_methods:
                    pattern_methods[name]()
                else:
                    print(f"Патерн '{name}' не підтримується.")
        else:
            for method in pattern_methods.values():
                method()

        return self.processed_data

class AlgorithmProcessor:
    def __init__(self, data: pd.DataFrame, processed_data: pd.DataFrame, algorithm_params=None):
        self.data = data
        self.processed_data = processed_data
        self.algorithm_params = algorithm_params if algorithm_params is not None else []

    def calculate_levels(self):
        resistance_levels_list = []
        support_levels_list = []

        # Метод 1: Піки та западини
        res_peaks, sup_peaks = self.find_peaks_levels(prominence=0.5, distance=1)
        if not res_peaks.empty:
            resistance_levels_list.append(res_peaks)
        if not sup_peaks.empty:
            support_levels_list.append(sup_peaks)

        # Метод 2: Фрактали
        res_fractals, sup_fractals = self.find_fractal_levels()
        if not res_fractals.empty:
            resistance_levels_list.append(res_fractals)
        if not sup_fractals.empty:
            support_levels_list.append(sup_fractals)

        # Метод 3: Pivot Points
        res_pivots, sup_pivots = self.calculate_pivot_points()
        if not res_pivots.empty:
            resistance_levels_list.append(res_pivots)
        if not sup_pivots.empty:
            support_levels_list.append(sup_pivots)

        # Метод 4: Фібоначчі
        res_fibo, sup_fibo = self.calculate_fibonacci_levels()
        if not res_fibo.empty:
            resistance_levels_list.append(res_fibo)
        if not sup_fibo.empty:
            support_levels_list.append(sup_fibo)

        # Перевірка, чи є рівні для комбінування
        if resistance_levels_list and support_levels_list:
            # Комбінування рівнів
            res_clusters, sup_clusters = self.combine_levels(resistance_levels_list, support_levels_list, clustering_tolerance=0.005)

            # Знаходимо значущі рівні (підтверджені як мінімум одним методом)
            self.significant_resistances, self.significant_supports = self.find_significant_levels(res_clusters, sup_clusters, methods_count=1)

            # Додаємо логічні колонки, які показують, чи ціна близька до рівнів
            self.processed_data['Near_Resistance'] = self.processed_data['close'].apply(
                lambda price: self.is_near_level(price, self.significant_resistances)
            )

            self.processed_data['Near_Support'] = self.processed_data['close'].apply(
                lambda price: self.is_near_level(price, self.significant_supports)
            )
        else:
            print("No levels found to combine.")
            self.processed_data['Near_Resistance'] = False
            self.processed_data['Near_Support'] = False

    def is_near_level(self, price, levels, tolerance=0.005):
        return any(abs((price - level) / level) <= tolerance for level in levels)

    def find_peaks_levels(self, prominence=1, distance=5):
        # Опір (максимуми)
        peaks, _ = find_peaks(self.data['high'], prominence=prominence, distance=distance)
        resistance_levels = self.data['high'].iloc[peaks]

        # Підтримка (мінімуми)
        troughs, _ = find_peaks(-self.data['low'], prominence=prominence, distance=distance)
        support_levels = self.data['low'].iloc[troughs]

        return resistance_levels, support_levels

    def find_fractal_levels(self):
        highs = self.data['high']
        lows = self.data['low']

        # Верхні фрактали
        upper_fractals = (highs.shift(2) < highs.shift(1)) & (highs.shift(1) < highs) & \
                         (highs > highs.shift(-1)) & (highs.shift(-1) > highs.shift(-2))
        resistance_levels = highs[upper_fractals]

        # Нижні фрактали
        lower_fractals = (lows.shift(2) > lows.shift(1)) & (lows.shift(1) > lows) & \
                         (lows < lows.shift(-1)) & (lows.shift(-1) < lows.shift(-2))
        support_levels = lows[lower_fractals]

        return resistance_levels, support_levels

    def calculate_pivot_points(self):
        high = self.data['high'].shift(1)
        low = self.data['low'].shift(1)
        close = self.data['close'].shift(1)

        pivot = (high + low + close) / 3
        resistance1 = (2 * pivot) - low
        support1 = (2 * pivot) - high
        resistance2 = pivot + (high - low)
        support2 = pivot - (high - low)

        # Об'єднуємо всі рівні опору та підтримки
        resistance_levels = pd.concat([resistance1, resistance2]).dropna()
        support_levels = pd.concat([support1, support2]).dropna()

        return resistance_levels, support_levels

    def calculate_fibonacci_levels(self):
        # Знайдемо останній максимум і мінімум за певний період
        lookback = min(100, len(self.data))  # кількість свічок для аналізу
        recent_high = self.data['high'].rolling(window=lookback).max().iloc[-1]
        recent_low = self.data['low'].rolling(window=lookback).min().iloc[-1]

        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        diff = recent_high - recent_low

        resistance_levels = [recent_high - diff * level for level in levels]
        support_levels = [recent_low + diff * level for level in levels]

        resistance_levels = pd.Series(resistance_levels)
        support_levels = pd.Series(support_levels)

        return resistance_levels, support_levels

    def combine_levels(self, resistance_levels_list, support_levels_list, clustering_tolerance=0.005):
        all_resistances = pd.concat(resistance_levels_list)
        all_supports = pd.concat(support_levels_list)

        # Кластеризація рівнів
        resistance_clusters = self.cluster_levels(all_resistances, clustering_tolerance)
        support_clusters = self.cluster_levels(all_supports, clustering_tolerance)

        return resistance_clusters, support_clusters

    def cluster_levels(self, levels, tolerance):
        levels = levels.dropna().sort_values().reset_index(drop=True)
        clustered_levels = []

        while not levels.empty:
            level = levels.iloc[0]
            if level == 0:
                # Уникаємо ділення на нуль
                close_levels = levels[np.abs(levels - level) <= tolerance]
            else:
                close_levels = levels[np.abs((levels - level) / level) <= tolerance]
            clustered_level = close_levels.mean()
            clustered_levels.append(clustered_level)
            levels = levels.drop(close_levels.index).reset_index(drop=True)

        return pd.Series(clustered_levels)

    def find_significant_levels(self, resistance_clusters, support_clusters, methods_count):
        # Визначаємо кількість підтверджень для кожного рівня
        resistance_levels = resistance_clusters.value_counts()
        support_levels = support_clusters.value_counts()

        # Фільтруємо рівні, які мають підтвердження від достатньої кількості методів
        significant_resistances = resistance_levels[resistance_levels >= methods_count].index
        significant_supports = support_levels[support_levels >= methods_count].index

        return significant_resistances, significant_supports
    
    def detect_market_structure(self, swing_window=3):
        """
        Визначення локальних максимумів/мінімумів і побудова структури ринку (HH, HL, LH, LL).

        :param swing_window: кількість свічок для виявлення локальних swing-high/swing-low.
        """

        highs = self.data['high']
        lows = self.data['low']

        structure = []

        for i in range(len(self.data)):
            if i < swing_window or i > len(self.data) - swing_window - 1:
                structure.append(None)
                continue

            local_high = highs[i] > highs[i - swing_window:i].max() and highs[i] > highs[i+1:i+swing_window+1].max()
            local_low = lows[i] < lows[i - swing_window:i].min() and lows[i] < lows[i+1:i+swing_window+1].min()

            if local_high:
                structure.append('swing_high')
            elif local_low:
                structure.append('swing_low')
            else:
                structure.append(None)

        self.processed_data['Market_Structure_Point'] = structure

        # Тепер визначаємо тип структури (HH, HL, LH, LL)
        last_swing_price = None
        last_swing_type = None
        structure_type = []

        for i in range(len(self.processed_data)):
            current_point = self.processed_data['Market_Structure_Point'].iloc[i]
            current_price = self.data['close'].iloc[i]

            if current_point is not None:
                if last_swing_price is None:
                    structure_type.append(None)
                else:
                    if current_point == 'swing_high':
                        if current_price > last_swing_price:
                            structure_type.append('HH')  # Higher High
                        else:
                            structure_type.append('LH')  # Lower High
                    elif current_point == 'swing_low':
                        if current_price > last_swing_price:
                            structure_type.append('HL')  # Higher Low
                        else:
                            structure_type.append('LL')  # Lower Low
                last_swing_price = current_price
                last_swing_type = current_point
            else:
                structure_type.append(None)

        self.processed_data['Market_Structure_Type'] = structure_type

    def detect_bos_choch(self):
        """
        Визначає Break of Structure (BOS) та Change of Character (CHoCH) на основі структури ринку.
        """

        structure = self.processed_data['Market_Structure_Type']
        close = self.data['close']

        last_hh = None
        last_hl = None
        last_ll = None
        last_lh = None
        trend_direction = None  # 'uptrend' або 'downtrend'

        bos = []
        choch = []

        for i in range(len(structure)):
            point = structure.iloc[i]
            price = close.iloc[i]

            bos_signal = False
            choch_signal = False

            if point == 'HH':
                if last_hh is None or price > last_hh:
                    last_hh = price
                if trend_direction == 'downtrend' and last_lh is not None and price > last_lh:
                    choch_signal = True
                    trend_direction = 'uptrend'
                elif trend_direction == 'uptrend' and last_hh is not None and price > last_hh:
                    bos_signal = True
                elif trend_direction is None:
                    trend_direction = 'uptrend'

            elif point == 'HL':
                if last_hl is None or price > last_hl:
                    last_hl = price

            elif point == 'LL':
                if last_ll is None or price < last_ll:
                    last_ll = price
                if trend_direction == 'uptrend' and last_hl is not None and price < last_hl:
                    choch_signal = True
                    trend_direction = 'downtrend'
                elif trend_direction == 'downtrend' and last_ll is not None and price < last_ll:
                    bos_signal = True
                elif trend_direction is None:
                    trend_direction = 'downtrend'

            elif point == 'LH':
                if last_lh is None or price < last_lh:
                    last_lh = price

            bos.append(bos_signal)
            choch.append(choch_signal)

        self.processed_data['BOS'] = bos
        self.processed_data['CHoCH'] = choch

    def detect_liquidity_sweep(self, swing_window=3, tolerance=0.0005):
        """
        Виявляє зачистку ліквідності (Liquidity Sweep) навколо swing-high/swing-low.

        :param swing_window: вікно для виявлення локальних swing.
        :param tolerance: допустиме відхилення для перевірки пробиття.
        """

        highs = self.data['high']
        lows = self.data['low']
        close = self.data['close']

        sweep_high = []
        sweep_low = []

        for i in range(len(self.data)):
            if i < swing_window or i > len(self.data) - swing_window - 1:
                sweep_high.append(False)
                sweep_low.append(False)
                continue

            local_high = highs[i - swing_window:i+swing_window+1].max()
            local_low = lows[i - swing_window:i+swing_window+1].min()

            # Sweep high: коли пробили попередній swing-high
            if highs[i] > local_high * (1 + tolerance):
                sweep_high.append(True)
            else:
                sweep_high.append(False)

            # Sweep low: коли пробили попередній swing-low
            if lows[i] < local_low * (1 - tolerance):
                sweep_low.append(True)
            else:
                sweep_low.append(False)

        self.processed_data['Sweep_High'] = sweep_high
        self.processed_data['Sweep_Low'] = sweep_low

    def detect_order_blocks(self, body_threshold=0.5, min_body_size=0.0001):
        """
        Виявляє Order Blocks (області великих угод перед імпульсом).

        :param body_threshold: співвідношення тіла свічки до загального діапазону для виявлення великих свічок.
        :param min_body_size: мінімальний розмір тіла для відбору OB.
        """

        open_ = self.data['open']
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']

        body = abs(close - open_)
        range_ = high - low

        bullish_ob = []
        bearish_ob = []

        for i in range(len(self.data)):
            if range_[i] == 0:
                bullish_ob.append(False)
                bearish_ob.append(False)
                continue

            body_ratio = body[i] / range_[i]

            if body_ratio > body_threshold and body[i] > min_body_size:
                if close[i] > open_[i]:
                    bullish_ob.append(True)
                    bearish_ob.append(False)
                else:
                    bullish_ob.append(False)
                    bearish_ob.append(True)
            else:
                bullish_ob.append(False)
                bearish_ob.append(False)

        self.processed_data['Bullish_OB'] = bullish_ob
        self.processed_data['Bearish_OB'] = bearish_ob

    def detect_fair_value_gaps(self, min_gap_ratio=0.0003):
        """
        Виявляє Fair Value Gaps (імбаланси) між трьома свічками.

        :param min_gap_ratio: мінімальний розмір FVG як частка ціни (0.0003 = 0.03%)
        """

        high = self.data['high']
        low = self.data['low']

        fvg_up = []
        fvg_down = []

        for i in range(2, len(self.data)):
            prev_high = high.iloc[i-2]
            prev_low = low.iloc[i-2]

            curr_high = high.iloc[i]
            curr_low = low.iloc[i]

            gap_up = curr_low > prev_high
            gap_down = curr_high < prev_low

            # Додатково перевіряємо мінімальний розмір FVG
            price = self.data['close'].iloc[i]
            if gap_up and (curr_low - prev_high) / price > min_gap_ratio:
                fvg_up.append(True)
                fvg_down.append(False)
            elif gap_down and (prev_low - curr_high) / price > min_gap_ratio:
                fvg_up.append(False)
                fvg_down.append(True)
            else:
                fvg_up.append(False)
                fvg_down.append(False)

        # Додаємо NaN для перших двох свічок (немає повного контексту)
        fvg_up = [False, False] + fvg_up
        fvg_down = [False, False] + fvg_down

        self.processed_data['FVG_Up'] = fvg_up
        self.processed_data['FVG_Down'] = fvg_down

    def process_data(self):
        algo_methods = {
            'Levels': self.calculate_levels,
            'Market_Structure': self.detect_market_structure,
            'BOS_CHoCH': self.detect_bos_choch,
            'Liquidity_Sweep': self.detect_liquidity_sweep,
            'Order_Blocks': self.detect_order_blocks,
            'Fair_Value_Gaps': self.detect_fair_value_gaps,
        }

        if self.algorithm_params:
            for name in self.algorithm_params:
                key = name if isinstance(name, str) else name.get("name")
                if key in algo_methods:
                    algo_methods[key]()
                else:
                    print(f"Алгоритмічна функція '{key}' не підтримується.")
        else:
            for method in algo_methods.values():
                method()

        return self.processed_data
    
class BacktestAlgorithmProcessor(AlgorithmProcessor):
    """
    Спеціалізований клас для обробки алгоритмічних фіч під час бектестування.
    Перевизначає методи для усунення "заглядання в майбутнє".
    Цей клас гарантує, що всі розрахунки базуються лише на історичних даних,
    доступних на момент поточної свічки.
    """
    def __init__(self, data: pd.DataFrame, processed_data: pd.DataFrame, algorithm_params=None, fractal_window=2):
        super().__init__(data, processed_data, algorithm_params)
        self.fractal_window = fractal_window # Window size for lookahead-free fractals
        # Store last confirmed swing points to build market structure incrementally
        self._last_swing_high = {'index': -1, 'value': -np.inf}
        self._last_swing_low = {'index': -1, 'value': np.inf}

    def find_fractal_levels(self):
        """
        Lookahead-free fractal detection.
        A fractal high is the highest point in a trailing window.
        A fractal low is the lowest point in a trailing window.
        """
        # Initialize columns if they don't exist
        if 'Fractal_High' not in self.processed_data.columns:
            self.processed_data['Fractal_High'] = False
        if 'Fractal_Low' not in self.processed_data.columns:
            self.processed_data['Fractal_Low'] = False

        if len(self.processed_data) < self.fractal_window + 1:
            return # Not enough data for initial fractal calculation

        # Get the current candle's index (last index in the slice)
        current_idx_in_slice = len(self.processed_data) - 1
        current_candle_df_index = self.processed_data.index[current_idx_in_slice]

        # Check for Fractal High at the current candle
        # The current high must be the maximum in the window ending at the current candle
        if current_idx_in_slice >= self.fractal_window:
            window_highs = self.processed_data['high'].iloc[current_idx_in_slice - self.fractal_window : current_idx_in_slice + 1]
            if self.processed_data['high'].iloc[current_idx_in_slice] == window_highs.max():
                self.processed_data.loc[current_candle_df_index, 'Fractal_High'] = True

            # Check for Fractal Low at the current candle
            window_lows = self.processed_data['low'].iloc[current_idx_in_slice - self.fractal_window : current_idx_in_slice + 1]
            if self.processed_data['low'].iloc[current_idx_in_slice] == window_lows.min():
                self.processed_data.loc[current_candle_df_index, 'Fractal_Low'] = True

    def find_peaks_levels(self, prominence=1, distance=5):
        """
        Lookahead-free find_peaks_levels.
        This version processes peaks/troughs only up to the current candle.
        """
        current_idx_in_slice = len(self.processed_data) - 1
        current_candle_df_index = self.processed_data.index[current_idx_in_slice]

        # To avoid lookahead, we can only confirm a peak/trough after `distance` candles have passed.
        # For a truly lookahead-free backtest, often peaks are confirmed retrospectively.
        # For simplicity here, we'll check the current candle relative to its *past* window.
        # This is a simplification, as `find_peaks` is inherently designed for full series.
        # A more robust lookahead-free peak detection would involve iterative confirmation.

        # For now, we'll mark the current candle as a potential peak/trough
        # if it's the highest/lowest in a trailing window.
        
        # Initialize columns if they don't exist
        if 'Peak_High' not in self.processed_data.columns:
            self.processed_data['Peak_High'] = False
        if 'Peak_Low' not in self.processed_data.columns:
            self.processed_data['Peak_Low'] = False

        if current_idx_in_slice >= distance:
            # Check for Peak High: current high is max in trailing window
            window_highs = self.processed_data['high'].iloc[current_idx_in_slice - distance : current_idx_in_slice + 1]
            if self.processed_data['high'].iloc[current_idx_in_slice] == window_highs.max():
                self.processed_data.loc[current_candle_df_index, 'Peak_High'] = True

            # Check for Peak Low: current low is min in trailing window
            window_lows = self.processed_data['low'].iloc[current_idx_in_slice - distance : current_idx_in_slice + 1]
            if self.processed_data['low'].iloc[current_idx_in_slice] == window_lows.min():
                self.processed_data.loc[current_candle_df_index, 'Peak_Low'] = True
        
        # This method will not return series, but update processed_data directly.
        # The `calculate_levels` method will then read from these columns.

    def detect_fair_value_gaps(self, min_gap_ratio=0.0003):
        """
        Lookahead-free Fair Value Gaps (FVG) detection for the current candle.
        Bullish FVG: Low[current] > High[current-2]
        Bearish FVG: High[current] < Low[current-2]
        """
        # Initialize columns if they don't exist
        if 'FVG_Up' not in self.processed_data.columns:
            self.processed_data['FVG_Up'] = False
        if 'FVG_Down' not in self.processed_data.columns:
            self.processed_data['FVG_Down'] = False
        if 'FVG_Size' not in self.processed_data.columns:
            self.processed_data['FVG_Size'] = np.nan

        current_idx_in_slice = len(self.processed_data) - 1
        current_candle_df_index = self.processed_data.index[current_idx_in_slice]

        if current_idx_in_slice < 2: # Need at least 3 candles (i, i-1, i-2) for FVG
            return

        curr_high = self.processed_data['high'].iloc[current_idx_in_slice]
        curr_low = self.processed_data['low'].iloc[current_idx_in_slice]
        prev2_high = self.processed_data['high'].iloc[current_idx_in_slice - 2]
        prev2_low = self.processed_data['low'].iloc[current_idx_in_slice - 2]
        current_price = self.processed_data['close'].iloc[current_idx_in_slice]

        # Bullish FVG
        if curr_low > prev2_high:
            gap_size = curr_low - prev2_high
            if current_price > 0 and (gap_size / current_price) > min_gap_ratio:
                self.processed_data.loc[current_candle_df_index, 'FVG_Up'] = True
                self.processed_data.loc[current_candle_df_index, 'FVG_Size'] = gap_size

        # Bearish FVG
        if curr_high < prev2_low:
            gap_size = prev2_low - curr_high
            if current_price > 0 and (gap_size / current_price) > min_gap_ratio:
                self.processed_data.loc[current_candle_df_index, 'FVG_Down'] = True
                self.processed_data.loc[current_candle_df_index, 'FVG_Size'] = gap_size

    def detect_market_structure(self):
        """
        Lookahead-free market structure detection based on confirmed fractals.
        This method updates the market structure for the current candle based on
        the most recent confirmed swing high/low.
        """
        # Initialize columns if they don't exist
        if 'Market_Structure_Point' not in self.processed_data.columns:
            self.processed_data['Market_Structure_Point'] = None
        if 'Market_Structure_Type' not in self.processed_data.columns:
            self.processed_data['Market_Structure_Type'] = None
        if 'Highs_Lows' not in self.processed_data.columns:
            self.processed_data['Highs_Lows'] = 0

        current_idx_in_slice = len(self.processed_data) - 1
        current_candle_df_index = self.processed_data.index[current_idx_in_slice]
        
        current_high = self.processed_data['high'].iloc[current_idx_in_slice]
        current_low = self.processed_data['low'].iloc[current_idx_in_slice]

        # Update last swing high/low if a new fractal is detected at the current candle
        if self.processed_data.loc[current_candle_df_index, 'Fractal_High']:
            self.processed_data.loc[current_candle_df_index, 'Market_Structure_Point'] = 'swing_high'
            if current_high > self._last_swing_high['value']:
                self.processed_data.loc[current_candle_df_index, 'Market_Structure_Type'] = 'HH'
                self.processed_data.loc[current_candle_df_index, 'Highs_Lows'] = 1
            else:
                self.processed_data.loc[current_candle_df_index, 'Market_Structure_Type'] = 'LH'
                self.processed_data.loc[current_candle_df_index, 'Highs_Lows'] = -1
            self._last_swing_high = {'index': current_candle_df_index, 'value': current_high}
            # If a new high, reset last low for HH/HL comparison
            self._last_swing_low = {'index': -1, 'value': np.inf} # Reset to allow new HL detection

        elif self.processed_data.loc[current_candle_df_index, 'Fractal_Low']:
            self.processed_data.loc[current_candle_df_index, 'Market_Structure_Point'] = 'swing_low'
            if current_low < self._last_swing_low['value']:
                self.processed_data.loc[current_candle_df_index, 'Market_Structure_Type'] = 'LL'
                self.processed_data.loc[current_candle_df_index, 'Highs_Lows'] = -1
            else:
                self.processed_data.loc[current_candle_df_index, 'Market_Structure_Type'] = 'HL'
                self.processed_data.loc[current_candle_df_index, 'Highs_Lows'] = 1
            self._last_swing_low = {'index': current_candle_df_index, 'value': current_low}
            # If a new low, reset last high for LL/LH comparison
            self._last_swing_high = {'index': -1, 'value': -np.inf} # Reset to allow new LH detection

        # If no new swing point, carry forward the last known structure type or None
        if current_idx_in_slice > 0 and self.processed_data.loc[current_candle_df_index, 'Market_Structure_Type'] is None:
            self.processed_data.loc[current_candle_df_index, 'Market_Structure_Type'] = \
                self.processed_data['Market_Structure_Type'].iloc[current_idx_in_slice - 1]
            self.processed_data.loc[current_candle_df_index, 'Highs_Lows'] = \
                self.processed_data['Highs_Lows'].iloc[current_idx_in_slice - 1]

    def calculate_levels(self):
        """
        Lookahead-free calculate_levels method.
        This method is now part of BacktestAlgorithmProcessor and will use
        the lookahead-free versions of `find_fractal_levels` and `find_peaks_levels`.
        """
        # Ensure fractal and peak levels are calculated first, as they modify processed_data in place
        self.find_fractal_levels() # This updates 'Fractal_High' and 'Fractal_Low'
        self.find_peaks_levels()   # This updates 'Peak_High' and 'Peak_Low'

        resistance_levels_list = []
        support_levels_list = []

        current_idx_in_slice = len(self.processed_data) - 1
        current_candle_df_index = self.processed_data.index[current_idx_in_slice]
        
        # Метод 1: Піки та западини (з lookahead-free Peak_High/Low)
        if 'Peak_High' in self.processed_data.columns and self.processed_data.loc[current_candle_df_index, 'Peak_High']:
            resistance_levels_list.append(pd.Series([self.processed_data['high'].iloc[current_idx_in_slice]]))
        if 'Peak_Low' in self.processed_data.columns and self.processed_data.loc[current_candle_df_index, 'Peak_Low']:
            support_levels_list.append(pd.Series([self.processed_data['low'].iloc[current_idx_in_slice]]))

        # Метод 2: Фрактали (з lookahead-free Fractal_High/Low)
        if 'Fractal_High' in self.processed_data.columns and self.processed_data.loc[current_candle_df_index, 'Fractal_High']:
            resistance_levels_list.append(pd.Series([self.processed_data['high'].iloc[current_idx_in_slice]]))
        if 'Fractal_Low' in self.processed_data.columns and self.processed_data.loc[current_candle_df_index, 'Fractal_Low']:
            support_levels_list.append(pd.Series([self.processed_data['low'].iloc[current_idx_in_slice]]))

        # Метод 3: Pivot Points (використовуємо базовий метод, він вже lookahead-free для поточної свічки)
        res_pivots, sup_pivots = super().calculate_pivot_points()
        if not res_pivots.empty:
            resistance_levels_list.append(res_pivots)
        if not sup_pivots.empty:
            support_levels_list.append(sup_pivots)

        # Метод 4: Фібоначчі (використовуємо базовий метод, він вже lookahead-free для поточної свічки)
        res_fibo, sup_fibo = super().calculate_fibonacci_levels()
        if not res_fibo.empty:
            resistance_levels_list.append(res_fibo)
        if not sup_fibo.empty:
            support_levels_list.append(sup_fibo)

        # Ініціалізуємо колонки, якщо вони ще не існують
        if 'Near_Resistance' not in self.processed_data.columns:
            self.processed_data['Near_Resistance'] = False
        if 'Near_Support' not in self.processed_data.columns:
            self.processed_data['Near_Support'] = False

        # Перевірка, чи є рівні для комбінування
        if resistance_levels_list or support_levels_list:
            all_resistances = pd.concat(resistance_levels_list).dropna() if resistance_levels_list else pd.Series()
            all_supports = pd.concat(support_levels_list).dropna() if support_levels_list else pd.Series()

            # Кластеризація рівнів
            # Викликаємо cluster_levels з правильним ім'ям аргументу 'tolerance'
            res_clusters = self.cluster_levels(all_resistances, tolerance=0.005) 
            sup_clusters = self.cluster_levels(all_supports, tolerance=0.005) 

            # Знаходимо значущі рівні (підтверджені як мінімум одним методом)
            # Примітка: find_significant_levels також є методом батьківського класу.
            # Оскільки він працює з Series, його можна використовувати без змін.
            self.significant_resistances, self.significant_supports = self.find_significant_levels(res_clusters, sup_clusters, methods_count=1)

            # Додаємо логічні колонки, які показують, чи ціна близька до рівнів
            current_price = self.processed_data['close'].iloc[current_idx_in_slice]
            
            self.processed_data.loc[current_candle_df_index, 'Near_Resistance'] = self.is_near_level(current_price, self.significant_resistances)
            self.processed_data.loc[current_candle_df_index, 'Near_Support'] = self.is_near_level(current_price, self.significant_supports)
        else:
            # Якщо рівнів не знайдено, встановлюємо False для поточної свічки
            self.processed_data.loc[current_candle_df_index, 'Near_Resistance'] = False
            self.processed_data.loc[current_candle_df_index, 'Near_Support'] = False
            # print("No levels found to combine for the current candle.")


    def process_data(self):
        """Main function to run algorithmic processing based on provided parameters."""
        algo_methods = {
            'Levels': self.calculate_levels, # Now calls the overridden version
            'Market_Structure': self.detect_market_structure,
            'BOS_CHoCH': self.detect_bos_choch,
            'Liquidity_Sweep': self.detect_liquidity_sweep,
            'Order_Blocks': self.detect_order_blocks,
            'Fair_Value_Gaps': self.detect_fair_value_gaps,
        }

        if self.algorithm_params:
            for name in self.algorithm_params:
                key = name if isinstance(name, str) else name.get("name")
                if key in algo_methods:
                    algo_methods[key]()
                else:
                    print(f"Алгоритмічна функція '{key}' не підтримується.")
        else:
            # If no parameters, run all algorithms with default values
            for method in algo_methods.values():
                method()

        return self.processed_data
    
class DataProcessingManager:
    def __init__(self, data: pd.DataFrame, indicators_params=None, pattern_params=None, algorithm_params=None, algorithm_processor_class=AlgorithmProcessor):
        self.data = data
        self.processed_data = data.copy()

        self.indicator_processor = IndicatorProcessor(
            self.data, self.processed_data, indicators_params
        )
        self.pattern_detector = PatternDetector(
            self.data, self.processed_data, pattern_params
        )
        self.algorithm_processor = AlgorithmProcessor(
            self.data, self.processed_data, algorithm_params
        )
        # Use the specified algorithm_processor_class
        self.algorithm_processor = algorithm_processor_class(
            self.data, self.processed_data, algorithm_params
        )

    def process_all(self):
        """Головний процес обробки: індикатори -> патерни -> алгоритми."""
        self.indicator_processor.process_data()
        self.pattern_detector.process_data()
        self.algorithm_processor.process_data()
        return self.processed_data
