# utils/strategies/LogicRegistry.py
import pandas as pd
import numpy as np

# --- БАЗОВІ КЛАСИ (Ваш існуючий код) ---

class LogicRegistry:
    def __init__(self):
        self._logic_map = {}

    def register(self, name: str, logic_class):
        self._logic_map[name] = logic_class
        print(f"Логіку '{name}' зареєстровано.")

    def get(self, name: str):
        logic_class = self._logic_map.get(name)
        if not logic_class:
            raise ValueError(f"Логіка з назвою '{name}' не зареєстрована.")
        return logic_class

    def list_all(self):
        return list(self._logic_map.keys())

class StrategyInterpreter:
    def __init__(self, config, registry, data):
        self.data = data
        self.registry = registry
        self.entry_rules = []
        self.exit_rules = []

        # Створення екземплярів класів логіки для умов входу
        for rule_config in config.get("entry_conditions", []):
            logic_class = registry.get(rule_config["type"])
            self.entry_rules.append(logic_class(rule_config["params"]))

        # Створення екземплярів класів логіки для умов виходу
        for rule_config in config.get("exit_conditions", []):
            logic_class = registry.get(rule_config["type"])
            self.exit_rules.append(logic_class(rule_config["params"]))

    def run(self) -> pd.Series:
        signals = pd.Series("hold", index=self.data.index)

        # Логіка входу: всі умови мають виконуватися одночасно (AND)
        if self.entry_rules:
            entry_mask = pd.Series(True, index=self.data.index)
            for rule in self.entry_rules:
                entry_mask &= rule.evaluate_series(self.data)
            signals[entry_mask] = "buy"

        # Логіка виходу: будь-яка умова призводить до виходу (OR)
        if self.exit_rules:
            exit_mask = pd.Series(False, index=self.data.index)
            for rule in self.exit_rules:
                exit_mask |= rule.evaluate_series(self.data)
            signals[exit_mask] = "sell"
            
        return signals

# --- ІСНУЮЧІ КЛАСИ ЛОГІКИ (Ваш код) ---

class CrossoverLogic:
    # Ваш код CrossoverLogic без змін
    def __init__(self, params):
        self.src1_col = params["col_1"]
        self.src2_col = params["col_2"]
        self.direction = params.get("direction", "above")

    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.src1_col not in df.columns or self.src2_col not in df.columns:
            raise ValueError(f"Колонки '{self.src1_col}' або '{self.src2_col}' не знайдено")
        a, b = df[self.src1_col], df[self.src2_col]
        if self.direction == "above":
            return (a.shift(1) < b.shift(1)) & (a > b)
        else:
            return (a.shift(1) > b.shift(1)) & (a < b)

class ThresholdLogic:
    # Ваш код ThresholdLogic без змін
    def __init__(self, params):
        self.source = params["source"]
        self.threshold = params["threshold"]
        self.operator = params["operator"]

    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.source not in df.columns:
            raise ValueError(f"Колонка '{self.source}' не знайдено")
        series = df[self.source]
        ops = {'>': series > self.threshold, '<': series < self.threshold, '>=': series >= self.threshold, '<=': series <= self.threshold}
        if self.operator not in ops:
            raise ValueError(f"Невідомий оператор: {self.operator}")
        return ops[self.operator]

class PatternLogic:
    # Ваш код PatternLogic без змін
    def __init__(self, params):
        self.pattern_name = params["pattern_name"]
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.pattern_name not in df.columns:
            raise ValueError(f"Колонка патерну '{self.pattern_name}' не знайдено")
        return df[self.pattern_name].astype(bool)


# --- НОВІ КЛАСИ ЛОГІКИ (15 правил) ---

# --- Тренд та моментум ---

class PriceVsMALogic:
    """Порівнює ціновий ряд з ковзною середньою."""
    def __init__(self, params):
        self.series_col = params.get("series", "close")
        self.ma_col = params["ma_col"] # e.g., "SMA_50"
        self.is_above = params.get("is_above", True)
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_above:
            return df[self.series_col] > df[self.ma_col]
        else:
            return df[self.series_col] < df[self.ma_col]

class MACDZeroCrossLogic:
    """Перетин лінією MACD нульового рівня."""
    def __init__(self, params):
        self.macd_line_col = params["macd_line"]
        self.cross_above = params.get("cross_above", True)
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        macd = df[self.macd_line_col]
        if self.cross_above:
            return (macd > 0) & (macd.shift(1) <= 0)
        else:
            return (macd < 0) & (macd.shift(1) >= 0)

class ADXTrendLogic:
    """Фільтр сили тренду за ADX."""
    def __init__(self, params):
        self.adx_col = params["adx_line"]
        self.threshold = params["threshold"]
        self.is_strong = params.get("is_strong", True)
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_strong:
            return df[self.adx_col] > self.threshold
        else:
            return df[self.adx_col] <= self.threshold

# --- Волатильність ---

class BollingerBreakoutLogic:
    """Пробій ціною стрічки Боллінджера."""
    def __init__(self, params):
        self.series_col = params.get("series", "close")
        self.band_col = params["band_col"] # e.g., "BBU_20_2.0" or "BBL_20_2.0"
        self.is_above = params["is_above"]
        
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_above:
            return df[self.series_col] > df[self.band_col]
        else:
            return df[self.series_col] < df[self.band_col]

class BollingerSqueezeLogic:
    """'Стиснення' стрічок Боллінджера."""
    def __init__(self, params):
        self.bandwidth_col = params["bandwidth_col"] # e.g., "BBB_20_2.0"
        self.threshold = params["threshold"]
        self.is_squeezing = params.get("is_squeezing", True)
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_squeezing:
            return df[self.bandwidth_col] < self.threshold
        else:
            return df[self.bandwidth_col] > self.threshold

class ATRVolatilityLogic:
    """Фільтр за рівнем волатильності ATR."""
    def __init__(self, params):
        self.atr_col = params["atr_col"]
        self.threshold = params["threshold"]
        self.is_high_vol = params.get("is_high_vol", True)
        
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_high_vol:
            return df[self.atr_col] > self.threshold
        else:
            return df[self.atr_col] < self.threshold

# --- Осцилятори ---

class StochasticLevelLogic:
    """Перебування стохастика в зоні перекупленості/перепроданості."""
    def __init__(self, params):
        self.k_line_col = params["k_line"] # e.g., "STOCHk_14_3_3"
        self.level = params["level"]
        self.is_above = params["is_above"]
        
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_above:
            return df[self.k_line_col] > self.level
        else:
            return df[self.k_line_col] < self.level

# --- Об'єм ---

class VolumeSpikeLogic:
    """Аномальний сплеск об'єму."""
    def __init__(self, params):
        self.volume_col = params.get("volume_col", "volume")
        self.ma_col = params["ma_col"] # e.g., "volume_sma_20"
        self.factor = params.get("factor", 2.0)
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        return df[self.volume_col] > (df[self.ma_col] * self.factor)

class OBVTrendLogic:
    """Тренд за індикатором On-Balance Volume."""
    def __init__(self, params):
        self.obv_col = params["obv_col"]
        self.ma_col = params["ma_col"] # e.g., "OBV_SMA_20"
        self.is_above = params.get("is_above", True)
        
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        if self.is_above:
            return df[self.obv_col] > df[self.ma_col]
        else:
            return df[self.obv_col] < df[self.ma_col]

# --- Патерни та ціна ---

class HigherHighLogic:
    """Поточний максимум є найвищим за N періодів."""
    def __init__(self, params):
        self.series_col = params.get("series", "high")
        self.period = params["period"]
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        return df[self.series_col] == df[self.series_col].rolling(window=self.period).max()

class LowerLowLogic:
    """Поточний мінімум є найнижчим за N періодів."""
    def __init__(self, params):
        self.series_col = params.get("series", "low")
        self.period = params["period"]
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        return df[self.series_col] == df[self.series_col].rolling(window=self.period).min()

class CandleBodySizeLogic:
    """Перевірка розміру тіла свічки."""
    def __init__(self, params):
        self.min_size_factor = params["min_size_factor"]
        self.relative_to_wick = params.get("relative_to_wick", True)
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        body_size = abs(df['close'] - df['open'])
        if self.relative_to_wick:
            total_range = df['high'] - df['low']
            return (body_size / total_range.replace(0, np.nan)) >= self.min_size_factor
        else:
            atr_col = 'ATR_14' # Потребує ATR
            if atr_col not in df.columns:
                raise ValueError(f"Для розрахунку CandleBodySize потрібна колонка {atr_col}")
            return body_size >= (df[atr_col] * self.min_size_factor)

# --- Час ---

class TimeOfDayLogic:
    """Торгівля лише у визначені години."""
    def __init__(self, params):
        self.start_hour = params["start_hour"]
        self.end_hour = params["end_hour"]
    
    def evaluate_series(self, df: pd.DataFrame) -> pd.Series:
        return (df.index.hour >= self.start_hour) & (df.index.hour < self.end_hour)


# --- ГЛОБАЛЬНИЙ РЕЄСТР ---
global_logic_registry = LogicRegistry()

# Реєстрація існуючих правил
global_logic_registry.register("Crossover", CrossoverLogic)
global_logic_registry.register("Threshold", ThresholdLogic)
global_logic_registry.register("Pattern", PatternLogic)

# Реєстрація 15 нових правил
print("\n--- Реєстрація розширеного набору правил ---")
global_logic_registry.register("PriceVsMA", PriceVsMALogic)
global_logic_registry.register("MACDZeroCross", MACDZeroCrossLogic)
global_logic_registry.register("ADXTrend", ADXTrendLogic)
global_logic_registry.register("BollingerBreakout", BollingerBreakoutLogic)
global_logic_registry.register("BollingerSqueeze", BollingerSqueezeLogic)
global_logic_registry.register("ATRVolatility", ATRVolatilityLogic)
global_logic_registry.register("StochasticLevel", StochasticLevelLogic)
# Зауважте: Стохастичний кросовер є підтипом загального кросовера.
# Тому його можна реалізувати через CrossoverLogic.
# global_logic_registry.register("StochasticCross", StochasticCrossLogic) 
global_logic_registry.register("VolumeSpike", VolumeSpikeLogic)
global_logic_registry.register("OBVTrend", OBVTrendLogic)
global_logic_registry.register("HigherHigh", HigherHighLogic)
global_logic_registry.register("LowerLow", LowerLowLogic)
global_logic_registry.register("CandleBodySize", CandleBodySizeLogic)
global_logic_registry.register("TimeOfDay", TimeOfDayLogic)
# Зауважте: MACDCross також можна реалізувати через CrossoverLogic.