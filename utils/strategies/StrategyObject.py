# base.py
import pandas as pd
from typing import Dict, List, Optional, Tuple

from utils.common.SettingsLoader import SettingsLoaderStrategies
from utils.common.FileStructureManager import FileStructureManager
from utils.data_processing.DataHandler import DataHandler
from utils.data_processing.DataProcessing import DataProcessingManager
from utils.common.other_utils import tf_to_minutes
from utils.strategies.LogicRegistry import StrategyInterpreter, global_logic_registry

class StrategyObject:
    def __init__(self, strategy_key: str, raw_ohlcv_data: pd.DataFrame):
        self.strategy_key = strategy_key
        self.raw_data = raw_ohlcv_data # Зберігаємо сирі дані

        # Завантаження налаштувань стратегії (це залишається без змін)
        self.settings = SettingsLoaderStrategies(
            module_name="Strategies",
            name_strategy=strategy_key
        )
        self.config = self.settings.settings or {}
        if not self.config:
            raise ValueError(f"Стратегія '{strategy_key}' не знайдена.")

        # ---- НОВА ЛОГІКА: Незалежна обробка даних ----
        # 1. Отримуємо список потрібних індикаторів з конфігурації
        required_indicators = self.config.get("required_indicators", [])
        if not required_indicators:
            print(f"⚠️ Для стратегії '{self.strategy_key}' не вказано 'required_indicators'. Аналіз може бути неповним.")

        # 2. Створюємо власний, тимчасовий DataProcessingManager
        #    Передаємо йому сирі дані та список потрібних індикаторів.
        dpm = DataProcessingManager(
            data=self.raw_data,
            indicators_params=required_indicators,
            pattern_params=[], # Стратегії поки не використовують патерни, але можна додати
            algorithm_params=[] # аналогічно
        )
        # 3. Обробляємо дані. Тепер self.data містить тільки те, що потрібно стратегії.
        self.data = dpm.process_all()
        # ---------------------------------------------

        # Інша частина конструктора залишається без змін,
        # але тепер вона працює з self.data, який ми щойно створили.
        self.builder = CustomStrategyBuilder(
            name=self.config["name"],
            processed_data=self.data, # <--- Передаємо сюди вже оброблені дані
            entry_conditions=self.config.get("entry_conditions", []),
            exit_conditions=self.config.get("exit_conditions", []),
            filters=self.config.get("filters", []),
            goal=self.config.get("goal", ""),
            risk_params=self.config.get("risk_params", {})
        )


    def generate_signals(self) -> pd.Series:
        return self.builder.generate_signals()

    def describe(self):
        return self.builder.describe()

    def save_or_update(self):
        # Записуємо опис стратегії в свій файл та зберігаємо
        self.settings.settings = self.builder.describe()
        self.settings.save_data()

class CustomStrategyBuilder:
    def __init__(self, name, processed_data, entry_conditions, exit_conditions, filters=None, goal="", risk_params=None):
        self.name = name
        self.data = processed_data
        self.entry_conditions = entry_conditions or []
        self.exit_conditions = exit_conditions or []
        self.filters = filters or []
        self.goal = goal
        self.risk_params = risk_params or {}

    def generate_signals(self) -> pd.Series:
        interpreter = StrategyInterpreter(
            config={
                "entry_conditions": self.entry_conditions,
                "exit_conditions": self.exit_conditions
            },
            registry=global_logic_registry,
            data=self.data
        )
        return interpreter.run()

    def describe(self):
        return {
            "name": self.name,
            "goal": self.goal,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "filters": self.filters,
            "risk_params": self.risk_params
        }

    @staticmethod
    def generate_default_strategy(template_key: str, processed_data: pd.DataFrame):
        # Можеш прописати шаблони під ключі (наприклад, RSI)
        if template_key == "rsi_basic":
            return CustomStrategyBuilder(
                name="RSI Basic",
                processed_data=processed_data,
                entry_conditions=["RSI_14 < 30"],
                exit_conditions=["RSI_14 > 70"],
                filters=["Volume_Avg_20 > 10000"],
                goal="Buy oversold, sell overbought",
                risk_params={"stop_loss": 0.02, "take_profit": 0.04}
            )
        else:
            raise ValueError(f"Шаблон '{template_key}' не підтримується.")
        
    def save(self, strategy_key: str):
        """
        Допоміжний метод, щоб зберегти цю стратегію під ключем strategy_key
        в data/Strategies/<strategy_key>_strategy.json
        """
        loader = SettingsLoaderStrategies(
            module_name="Strategies",
            name_strategy=strategy_key
        )
        loader.settings = self.describe()
        loader.save_data()

class MultiStrategyObject:
    @classmethod
    def from_raw_files(
        cls,
        tf_map: Dict[str, Tuple[str, str]],  # {tf: (raw_file_path, strategy_key)}
        limit_rows: Optional[int] = None,
        indicators: List = None,
        patterns: List = None,
        algorithms: List = None
    ) -> "MultiStrategyObject":
        """
        Ініціалізація з "сирих" CSV-файлів і одночасне генерування оброблених даних через DataHandler.
        Передаємо кастомні інструменти (indicators, patterns, algorithms).
        """
        dh = DataHandler()
        dh.set_custom_parameters(indicators, patterns, algorithms)

        data_by_tf: Dict[str, pd.DataFrame] = {}
        strategy_map: Dict[str, str] = {}

        for tf, (raw_path, strat_key) in tf_map.items():
            # Використовуємо єдиний метод generate_strategy_data для обробки та сигналів
            processed_df = dh.process_and_return_df(
                raw_file_path=raw_path,
                limit_rows=limit_rows,
                indicators=indicators,
                patterns=patterns,
                algorithms=algorithms
            )
            data_by_tf[tf] = processed_df
            strategy_map[tf] = strat_key

        return cls(data_by_timeframe=data_by_tf, strategy_map=strategy_map)
    

    def __init__(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        strategy_map: Dict[str, str]
    ):
        self.data_by_timeframe = data_by_timeframe
        self.strategy_map = strategy_map
        # Створюємо StrategyObject для кожного TF
        self.strategies = {
            tf: StrategyObject(strat_key, df)
            for tf, (df, strat_key) in zip(data_by_timeframe.keys(), [(df, sk) for df, sk in zip(data_by_timeframe.values(), strategy_map.values())])
        }

    def generate_signals(self) -> Dict[str, pd.Series]:
        return {tf: strat.generate_signals() for tf, strat in self.strategies.items()}

    def describe(self) -> Dict[str, dict]:
        return {tf: strat.describe() for tf, strat in self.strategies.items()}

    def save_or_update(self):
        for strat in self.strategies.values():
            strat.save_or_update()

    def combine_signals(
        self,
        method: str = "weighted",
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.0,
        priority_order: Optional[List[str]] = None
    ) -> pd.Series:
        sigs = self.generate_signals()
        idx = next(iter(sigs.values())).index
        num = {tf: sigs[tf].map({"buy": 1, "sell": -1, "hold": 0}) for tf in sigs}

        if method == "sum":
            total = sum(num.values())

        elif method == "weighted":
            if weights is None:
                weights = {tf: tf_to_minutes(tf) for tf in num}
            norm = sum(weights.values())
            total = sum(num[tf] * (weights.get(tf, 0) / norm) for tf in num)

        elif method == "priority":
            if not priority_order:
                raise ValueError("Для priority потрібен priority_order")
            result = pd.Series("hold", index=idx)
            for tf in priority_order:
                if tf not in num:
                    continue
                mask_buy = (num[tf] == 1) & (result == "hold")
                mask_sell = (num[tf] == -1) & (result == "hold")
                result[mask_buy] = "buy"
                result[mask_sell] = "sell"
            return result

        else:
            raise ValueError(f"Невідомий метод '{method}'")

        final = pd.Series("hold", index=idx)
        final[total > threshold] = "buy"
        final[total < -threshold] = "sell"
        return final
