import os
import re
import pandas as pd
import asyncio

from utils.common.FileStructureManager import FileStructureManager
from utils.common.SettingsLoader import ProcessingSettingsBuilder  # Замініть шлях на свій
from utils.data_processing.DataProcessing import DataProcessingManager

class DataHandler:
    def __init__(self):
        self.fsm = FileStructureManager()
        self.settings_builder = ProcessingSettingsBuilder()
        self.settings_builder.ensure_defaults_exist()
        self.custom_indicators = None
        self.custom_patterns = None
        self.custom_algorithms = None
        self.limit_rows = 1000
        print("✅ Налаштування завантажені або створені.")

    def set_custom_parameters(self, indicators=None, patterns=None, algorithms=None):
        self.custom_indicators = indicators
        self.custom_patterns = patterns
        self.custom_algorithms = algorithms

    def set_files_by_symbols(self, symbols, timeframe_key, extension=".csv", limit=1000):
        self.limit_rows = limit
        directory_path = self.fsm.get_path(timeframe_key, is_file=False)

        if not directory_path:
            print(f"❌ Не знайдено директорію за ключем: {timeframe_key}")
            return

        selected_files = []

        for symbol in symbols:
            subfolder_path = os.path.join(directory_path, symbol)
            if not os.path.isdir(subfolder_path):
                print(f"⚠️ Пропущено: папка {symbol} не існує")
                continue

            for file in os.listdir(subfolder_path):
                if not file.endswith(extension):
                    continue

                name, ext = os.path.splitext(file)
                if re.search(r'_processing($|[_\.])', name):
                    print(f"⚠️ Пропущено оброблений файл: {file}")
                    continue

                if name == symbol:
                    selected_files.append(os.path.join(subfolder_path, file))
                    break  # Знайдено головний файл, далі не шукати

        self.files_list = selected_files
        print(f"📦 Вибрано {len(self.files_list)} файлів для обробки.")

    def get_processing_params(self):
        indicators = self.custom_indicators or self.settings_builder.get_indicator_settings()
        patterns = self.custom_patterns or self.settings_builder.get_pattern_settings()
        algorithms = self.custom_algorithms or self.settings_builder.get_algorithm_settings()
        return indicators, patterns, algorithms

    async def process_file_full(self, file_path, limit_rows=None, output_suffix: str = "_processing"):
        try:
            print(f"▶️ Початок обробки файлу: {file_path}")

            df = pd.read_csv(file_path)
            if 'close_time' in df.columns:
                df['close_time'] = pd.to_datetime(df['close_time'])

            row_limit = limit_rows or self.limit_rows
            if row_limit:
                df = df.tail(row_limit).reset_index(drop=True)

            indicators_params, pattern_params, algorithm_params = self.get_processing_params()

            manager = DataProcessingManager(
                df,
                indicators_params=indicators_params,
                pattern_params=pattern_params,
                algorithm_params=algorithm_params
            )
            processed_df = manager.process_all()

            custom_indicators = self.settings_builder.get_settings_from_class("DataProcessingManager", "indicators_custom")
            if custom_indicators:
                print(f"⚙️ Додаємо кастомні індикатори: {custom_indicators}")
                manager_custom = DataProcessingManager(
                    processed_df,
                    indicators_params=custom_indicators
                )
                processed_df = manager_custom.process_all()

            dir_name, file_name = os.path.split(file_path)
            name, ext = os.path.splitext(file_name)
            dir_name, file_name = os.path.split(file_path)
            name, ext = os.path.splitext(file_name)
            processed_name = f"{name}{output_suffix}{ext}"
            processed_path = os.path.join(dir_name, processed_name)

            processed_df.to_csv(processed_path, index=False)
            print(f"✅ Файл оброблено і збережено: {processed_path}")

        except Exception as e:
            print(f"❌ Помилка під час обробки {file_path}: {e}")

    async def process_all_files_async(self, limit_rows=None):
        if not hasattr(self, "files_list") or not self.files_list:
            print("⚠️ Список файлів порожній. Виклич load_files_list() або set_files_by_symbols() перед обробкою.")
            return

        tasks = [
            self.process_file_full(file_path, limit_rows=limit_rows)
            for file_path in self.files_list
        ]
        await asyncio.gather(*tasks)

    def process_all_files(self, limit_rows=None):
        asyncio.run(self.process_all_files_async(limit_rows=limit_rows))

    def generate_strategy_data(
        self,
        raw_file_path: str,
        strategy_key: str,
        limit_rows: int = None,
        indicators: list = None,
        patterns: list = None,
        algorithms: list = None,
        output_suffix: str = "_processing"
    ) -> pd.Series:
        """
        1) Читає сирий CSV із data/Strategies
        2) Обрізає до останніх limit_rows рядків (якщо вказано)
        3) Використовує DataProcessingManager з вашими (чи кастом) indicators/patterns/algorithms
        4) Зберігає оброблений файл поруч з сирим, додаючи output_suffix до імені
        5) Інстанціює StrategyObject для strategy_key і повертає сигнали

        :param raw_file_path: шлях до data/Strategies/<symbol>.csv
        :param strategy_key: ключ стратегії (назва JSON-файлу без розширення)
        :param limit_rows: останні N рядків (необов’язково)
        :param indicators: кастомні індикатори (необов’язково)
        :param patterns: кастомні патерни (необов’язково)
        :param algorithms: кастомні алгоритми (необов’язково)
        :param output_suffix: суфікс для обробленого файлу (за замовчуванням "_processing")
        :return: pd.Series зі значеннями "buy"/"sell"/"hold"
        """
        # --- 1. Завантажуємо сирі дані ---
        df = pd.read_csv(raw_file_path)
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'])

        # --- 2. Обрізаємо, якщо потрібно ---
        if limit_rows:
            df = df.tail(limit_rows).reset_index(drop=True)

        # --- 3. Підготуємо налаштування обробки ---
        settings_builder = ProcessingSettingsBuilder()
        settings_builder.ensure_defaults_exist()
        indicators_params = indicators or settings_builder.get_indicator_settings()
        pattern_params    = patterns   or settings_builder.get_pattern_settings()
        algorithm_params  = algorithms or settings_builder.get_algorithm_settings()

        # --- 4. Обробляємо через ваш DataProcessingManager ---
        manager = DataProcessingManager(
            data=df,
            indicators_params=indicators_params,
            pattern_params=pattern_params,
            algorithm_params=algorithm_params
        )
        processed_df = manager.process_all()

        # --- 5. Зберігаємо оброблений CSV поруч із сирим ---
        dir_name, file_name = os.path.split(raw_file_path)
        name, ext = os.path.splitext(file_name)
        processed_name = f"{name}{output_suffix}{ext}"
        processed_path = os.path.join(dir_name, processed_name)
        processed_df.to_csv(processed_path, index=False)
        print(f"✅ Оброблений файл збережено: {processed_path}")

        # --- 6. Генеруємо сигнали по вашій стратегії ---
        # StrategyObject всередині використовує SettingsLoaderStrategies,
        # тому в data/Strategies/ має лежати <strategy_key>_strategy.json
        from utils.strategies.StrategyObject import StrategyObject
        strat = StrategyObject(strategy_key=strategy_key, processed_data=processed_df)
        signals = strat.generate_signals()

        return signals

    def process_and_return_df(self, raw_file_path, limit_rows, indicators, patterns, algorithms):
        # --- 1. Завантажуємо сирі дані ---
        df = pd.read_csv(raw_file_path)
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'])

        # --- 2. Обрізаємо, якщо потрібно ---
        if limit_rows:
            df = df.tail(limit_rows).reset_index(drop=True)

        # --- 3. Підготуємо налаштування обробки ---
        settings_builder = ProcessingSettingsBuilder()
        settings_builder.ensure_defaults_exist()
        indicators_params = indicators or settings_builder.get_indicator_settings()
        pattern_params    = patterns   or settings_builder.get_pattern_settings()
        algorithm_params  = algorithms or settings_builder.get_algorithm_settings()

        # --- 4. Обробляємо через ваш DataProcessingManager ---
        manager = DataProcessingManager(
            data=df,
            indicators_params=indicators_params,
            pattern_params=pattern_params,
            algorithm_params=algorithm_params
        )
        processed_df = manager.process_all()
        return processed_df