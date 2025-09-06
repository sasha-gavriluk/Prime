# utils/common/Installer.py

import os
import json
from .other_utils import get_project_root, ensure_directory_exists
from .SettingsLoader import (
    DecisionSettingsManager,
    FinancialSettingsManager,
    MarketStateSettingsManager,
    ProcessingSettingsBuilder
)
from .HistoryDB import HistoryDB
from .FileStructureManager import FileStructureManager

class Installer:
    """
    Клас, що відповідає за перевірку та створення необхідних
    директорій і файлів конфігурації для першого запуску програми.
    """
    def __init__(self):
        self.root_dir = get_project_root()
        print("🚀 Запуск інсталятора для перевірки структури проєкту...")

    def run_setup(self):
        """
        Головний метод, який послідовно виконує всі кроки
        встановлення та перевірки.
        """
        print("\n--- 1. Перевірка та створення директорій ---")
        self._create_directories()

        print("\n--- 2. Перевірка та створення файлів конфігурації ---")
        self._create_api_keys_template()
        self._create_default_strategy_template()

        print("\n--- 3. Ініціалізація налаштувань модулів ---")
        self._initialize_settings_modules()

        print("\n--- 4. Ініціалізація бази даних історії ---")
        self._initialize_database()

        print("\n--- 5. Оновлення карти файлової структури ---")
        self._refresh_file_structure_map()

        print("\n✅ Встановлення та перевірка успішно завершені!")

    def _create_directories(self):
        """Створює базові папки, якщо вони відсутні."""
        # Список обов'язкових директорій
        required_dirs = [
            'data',
            'data/Settings',
            'data/Strategies',
            'data/Backups',
            'data/Binance',
            'data/Bybit',
            'logs',
            'logs/backtests'
        ]
        for dir_name in required_dirs:
            dir_path = os.path.join(self.root_dir, dir_name)
            if not os.path.exists(dir_path):
                ensure_directory_exists(dir_path) #
                print(f"   Створено директорію: {dir_path}")
            else:
                print(f"   Директорія існує: {dir_path}")

    def _create_api_keys_template(self):
        """Створює шаблон для ключів API, якщо файл відсутній."""
        keys_path = os.path.join(self.root_dir, 'data', 'Settings', 'Settings_keys.json')
        if not os.path.exists(keys_path):
            api_template = {
              "binance": {
                "api_key": "YOUR_BINANCE_API_KEY",
                "api_secret": "YOUR_BINANCE_API_SECRET"
              },
              "bybit": {
                "api_key": "YOUR_BYBIT_API_KEY",
                "api_secret": "YOUR_BYBIT_API_SECRET"
              }
            }
            with open(keys_path, 'w', encoding='utf-8') as f:
                json.dump(api_template, f, indent=4)
            print(f"   Створено шаблон для ключів API: {keys_path}")
            print("   ⚠️ Важливо: заповніть цей файл вашими реальними ключами API.")
        else:
            print(f"   Файл ключів API існує: {keys_path}")

    def _create_default_strategy_template(self):
        """Створює приклад файлу стратегії, щоб показати користувачу формат."""
        strategy_path = os.path.join(self.root_dir, 'data', 'Strategies', 'example_strategy_strategy.json') #
        if not os.path.exists(strategy_path):
            strategy_template = {
                "name": "Example RSI Crossover",
                "goal": "Buy when RSI is oversold, Sell when RSI is overbought.",
                "timeframe_id": "1h",
                "required_indicators": [
                    {"name": "RSI", "parameters": {"period": 14}}
                ],
                "entry_conditions": [
                    {
                        "type": "Threshold",
                        "params": {"source": "RSI_14", "threshold": 30, "operator": "<"}
                    }
                ],
                "exit_conditions": [
                    {
                        "type": "Threshold",
                        "params": {"source": "RSI_14", "threshold": 70, "operator": ">"}
                    }
                ],
                "filters": [],
                "risk_params": {"stop_loss_pct": 2.0, "take_profit_ratio": 1.5}
            }
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump(strategy_template, f, indent=4)
            print(f"   Створено приклад стратегії: {strategy_path}")
        else:
            print(f"   Приклад стратегії існує: {strategy_path}")

    def _initialize_settings_modules(self):
        """
        Ініціалізує класи налаштувань, щоб вони створили свої файли
        з дефолтними значеннями.
        """
        print("   Ініціалізація налаштувань для:")
        ProcessingSettingsBuilder().ensure_defaults_exist() #
        print("     - DataProcessingManager")
        DecisionSettingsManager().ensure_defaults_exist() #
        print("     - DecisionEngine")
        MarketStateSettingsManager().ensure_defaults_exist() #
        print("     - MarketStateDetector")
        FinancialSettingsManager().ensure_defaults_exist() #
        print("     - FinancialAdvisor")

    def _initialize_database(self):
        """Ініціалізує клас для роботи з історією, який сам створить БД."""
        HistoryDB() #
        print("   База даних історії угод перевірена/створена.")

    def _refresh_file_structure_map(self):
        """Оновлює карту файлової структури після створення всіх файлів."""
        fsm = FileStructureManager() #
        fsm.refresh_structure() #
        print("   Карту файлової структури оновлено.")