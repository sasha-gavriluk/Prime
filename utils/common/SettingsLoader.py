import json
import os

class SettingsLoader:
    def __init__(self, module_name):
        """
        Ініціалізація класу налаштувань для конкретного модуля.

        :param module_name: Назва модуля, для якого створюється файл налаштувань.
        """
        self.module_name = module_name
        self.base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', "..", 'data', 'Settings'))
        self.settings_file = os.path.join(self.base_dir, f"{module_name}_settings.json")

        # Завантажуємо або створюємо файл налаштувань
        self.settings = self.load_data()

    def load_data(self):
        """
        Завантажує дані з файлу налаштувань або створює новий файл, якщо його немає.

        :return: Дані з файлу у вигляді словника.
        """
        if not os.path.exists(self.settings_file):
            print(f"Файл {self.settings_file} не знайдено. Створюється новий файл.")
            with open(self.settings_file, 'w', encoding="utf-8") as file:
                json.dump({}, file)  # Створюємо порожній файл JSON
            return {}

        try:
            with open(self.settings_file, 'r', encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Помилка завантаження даних: {e}")
            return {}
        
    def reload(self):
        self.settings = self.load_data()

    def save_data(self):
        """
        Зберігає поточні дані в файл налаштувань.
        """
        try:
            with open(self.settings_file, 'w') as file:
                json.dump(self.settings, file, indent=4)
        except Exception as e:
            print(f"Помилка збереження даних: {e}")

    def get_nested_setting(self, keys, default=None):
        """
        Отримує значення налаштувань з багаторівневої структури за допомогою списку ключів.

        :param keys: Список ключів, які вказують на вкладену структуру.
        :param default: Значення за замовчуванням, якщо ключі не знайдено.
        :return: Значення налаштувань або значення за замовчуванням.
        """
        value = self.settings
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def update_nested_setting(self, keys, value):
        """
        Оновлює значення у багаторівневих налаштуваннях за допомогою списку ключів.

        :param keys: Список ключів для оновлення.
        :param value: Нове значення.
        """
        d = self.settings
        try:
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value
            self.save_data()
        except Exception as e:
            print(f"Помилка оновлення налаштувань: {e}")

    def add_settings_to_class(self, class_name, key_name, new_settings):
        """
        Додає або оновлює налаштування для вказаного класу з вказаною назвою ключа.

        :param class_name: Назва класу, до якого додаються налаштування.
        :param key_name: Назва ключа, під яким будуть додані нові налаштування.
        :param new_settings: Нові налаштування у вигляді словника або списку.
        """
        if class_name not in self.settings:
            self.settings[class_name] = {}  # Створюємо пустий словник для класу, якщо його ще немає

        if key_name not in self.settings[class_name]:
            # Створюємо пустий словник або список для ключа, якщо його ще немає
            if isinstance(new_settings, dict):
                self.settings[class_name][key_name] = {}
            elif isinstance(new_settings, list):
                self.settings[class_name][key_name] = []

        existing_settings = self.settings[class_name][key_name]

        # Якщо нові налаштування є словником
        if isinstance(new_settings, dict):
            for new_key, new_value in new_settings.items():
                if new_key in existing_settings:
                    if existing_settings[new_key] != new_value:
                        print(f"Оновлюємо параметри для {new_key}.")
                        existing_settings[new_key] = new_value
                else:
                    print(f"Додаємо новий індикатор/паттерн {new_key}.")
                    existing_settings[new_key] = new_value

        # Якщо нові налаштування є списком
        elif isinstance(new_settings, list):
            for item in new_settings:
                if item not in existing_settings:
                    existing_settings.append(item)
                else:
                    print(f"Елемент {item} вже існує в списку.")

        # Зберігаємо оновлені налаштування
        self.save_data()

    def get_settings_from_class(self, class_name, key_name):
        """
        Отримує налаштування для вказаного класу та ключа.

        :param class_name: Назва класу, з якого витягуються налаштування.
        :param key_name: Назва ключа, під яким зберігаються налаштування.
        :return: Налаштування у вигляді словника або списку, або None, якщо не знайдено.
        """
        if class_name in self.settings and key_name in self.settings[class_name]:
            return self.settings[class_name][key_name]
        else:
            print(f"Налаштування для {class_name} або {key_name} не знайдено.")
            return None
        
    def delete_nested_setting(self, keys):
        """
        Видаляє ключ у багаторівневих налаштуваннях за списком ключів.
        :param keys: список ключів (для вкладеності)
        """
        d = self.settings
        try:
            for key in keys[:-1]:
                d = d[key]
            if keys[-1] in d:
                del d[keys[-1]]
                self.save_data()
                print(f"Ключ {'.'.join(keys)} видалено.")
            else:
                print(f"Ключ {'.'.join(keys)} не знайдено для видалення.")
        except (KeyError, TypeError) as e:
            print(f"Помилка при видаленні {'.'.join(keys)}: {e}")


class ProcessingSettingsBuilder(SettingsLoader):
    def __init__(self, module_name="DataProcessingManager"):
        super().__init__(module_name)

        # Перевіряємо та ініціалізуємо дефолтні налаштування, якщо їх нема
    def ensure_defaults_exist(self):
        defaults = {
            "indicators": [
                {"name": "SMA", "parameters": {"period": 20}},
                {"name": "EMA", "parameters": {"period": 50}},
                {"name": "EMA", "parameters": {"period": 20}},
                {"name": "RSI", "parameters": {"period": 14}},
                {"name": "MACD", "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}},
                {"name": "ATR", "parameters": {"period": 14}},
                {"name": "Bollinger_Bands", "parameters": {"period": 20, "std_multiplier": 2}},
                {"name": "ADX", "parameters": {"period": 14}},
                {"name": "Stochastic", "parameters": {"k_period": 14, "d_period": 3}},
                {"name": "CCI", "parameters": {"period": 20}},
                {"name": "Volume_Avg", "parameters": {"period": 20}},
                {"name": "Keltner_Channel", "parameters": {"period": 20, "multiplier": 2}},
                {"name": "SMA_Cross", "parameters": {"period_short": 10, "period_long": 50}},
                {"name": "EMA_Cross", "parameters": {"period_short": 10, "period_long": 50}},
                {"name": "WilliamsR", "parameters": {"period": 14}}
            ],
            "patterns": [
                "Hammer",
                "Inverted_Hammer",
                "Shooting_Star",
                "Engulfing",
                "Morning_Star",
                "Evening_Star",
                "Piercing_Pattern",
                "Dark_Cloud_Cover",
                "Three_White_Soldiers",
                "Three_Black_Crows",
                "Hanging_Man"
            ],
            "algorithms": [
                "Levels",
                "Market_Structure",
                "BOS_CHoCH",
                "Liquidity_Sweep",
                "Order_Blocks",
                "Fair_Value_Gaps"
            ]
        }

        if "DataProcessingManager" not in self.settings:
            self.settings["DataProcessingManager"] = {}

        dpm = self.settings["DataProcessingManager"]
        updated = False

        for key, default_value in defaults.items():
            if key not in dpm:
                dpm[key] = default_value
                updated = True

        if updated:
            self.save_data()

    def get_indicator_settings(self):
        return self.settings["DataProcessingManager"].get("indicators", [])

    def get_pattern_settings(self):
        return self.settings["DataProcessingManager"].get("patterns", [])

    def get_algorithm_settings(self):
        return self.settings["DataProcessingManager"].get("algorithms", [])

    def add_indicator_setting(self, name: str, parameters: dict):
        new_setting = {"name": name, "parameters": parameters}
        self.add_settings_to_class("DataProcessingManager", "indicators_custom", [new_setting])

    def update_indicators(self, new_list: list):
        self.update_nested_setting(["DataProcessingManager", "indicators"], new_list)

    def update_patterns(self, new_list: list):
        self.update_nested_setting(["DataProcessingManager", "patterns"], new_list)

    def update_algorithms(self, new_list: list):
        self.update_nested_setting(["DataProcessingManager", "algorithms"], new_list)

class DecisionSettingsManager(SettingsLoader):
    def __init__(self, module_name="DecisionEngine"):
        super().__init__(module_name)
        self.ensure_defaults_exist()

    def ensure_defaults_exist(self):
        defaults = {
            "timeframe_weights": {
                "1m": 0.5, "3m": 0.7, "5m": 1.0, "15m": 1.2, "30m": 1.3,
                "1h": 1.5, "2h": 1.6, "4h": 1.8, "1d": 2.0
            },
            "metric_weights": {
                "smc_confidence": 1.5,
                "retracement_quality": 1.0,
                "fvg_quality": 0.8,
                "pattern_buy_score": 1.0,
                "pattern_sell_score": 1.0,
                "news_impact": 1.2  # <-- ДОДАНО ВАГУ ДЛЯ НОВИН
            },
             "agent_configurations": {
                "default": { # Для повільних ТФ: 1h, 4h, 1d
                    "trend": {"column": "EMA_50"},
                    "volatility": {"column": "ATR_14", "threshold": 0.3}
                    
                },
                "fast": { # Для швидких ТФ: 1m, 5m, 15m
                    "trend": {"column": "EMA_20"},
                    "volatility": {"column": "ATR_14", "threshold": 0.5}
                }
            }
        }

        updated = False
        for key, default_value in defaults.items():
            if key not in self.settings:
                self.settings[key] = default_value
                updated = True
            elif isinstance(default_value, dict):
                # Якщо ключ існує і є словником, перевіряємо вкладені ключі
                for sub_key, sub_default_value in default_value.items():
                    if sub_key not in self.settings.get(key, {}):
                        self.settings[key][sub_key] = sub_default_value
                        updated = True
        
        if updated:
            self.save_data()

    def get_timeframe_weights(self):
        return self.settings.get("timeframe_weights", {})

    def get_metric_weights(self):
        return self.settings.get("metric_weights", {})

    def get_agent_configurations(self):
        return self.settings.get("agent_configurations", {})

    def update_timeframe_weights(self, new_weights: dict):
        self.update_nested_setting(["timeframe_weights"], new_weights)

    def update_metric_weights(self, new_weights: dict):
        self.update_nested_setting(["metric_weights"], new_weights)

class MarketStateSettingsManager(SettingsLoader):
    def __init__(self, module_name="MarketStateDetector"):
        super().__init__(module_name)
        self.ensure_defaults_exist()

    def ensure_defaults_exist(self):
        defaults = {
            "states": [
                {"name": "trend", "params": {"adx_period": 14, "adx_threshold": 25, "adx_strong_threshold": 60}},
                {"name": "range", "params": {"adx_period": 14, "adx_threshold": 20}}
            ]
        }
        
        # Перевіряємо, чи існує ключ "MarketStateDetector" у налаштуваннях
        if "MarketStateDetector" not in self.settings:
            self.settings["MarketStateDetector"] = {}

        # Перевіряємо та додаємо дефолтні налаштування, якщо їх немає
        dpm_settings = self.settings["MarketStateDetector"]
        updated = False
        for key, default_value in defaults.items():
            if key not in dpm_settings:
                dpm_settings[key] = default_value
                updated = True
            # Якщо ключ існує, але це словник, перевіряємо вкладені ключі (для майбутніх розширень)
            elif isinstance(default_value, dict) and isinstance(dpm_settings[key], dict):
                for sub_key, sub_default_value in default_value.items():
                    if sub_key not in dpm_settings[key]:
                        dpm_settings[key][sub_key] = sub_default_value
                        updated = True
            # Якщо це список (наприклад, "states"), можна перевірити наявність конкретних елементів
            elif isinstance(default_value, list) and isinstance(dpm_settings[key], list):
                # Простий приклад: переконатися, що обидва стани "trend" і "range" існують
                existing_state_names = {s.get('name') for s in dpm_settings[key]}
                for default_state in default_value:
                    if default_state.get('name') not in existing_state_names:
                        dpm_settings[key].append(default_state)
                        updated = True

        if updated:
            self.save_data()

    def get_state_configurations(self):
        # Повертаємо конфігурації станів
        return self.settings.get("MarketStateDetector", {}).get("states", [])

    def update_state_configurations(self, new_configs: list):
        self.update_nested_setting(["MarketStateDetector", "states"], new_configs)


class SettingsLoaderStrategies(SettingsLoader):

    def __init__(self, module_name, name_strategy):
        self.module_name = module_name
        self.name_strategy = name_strategy
        self.base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Strategies'))
        self.settings_file = os.path.join(self.base_dir, f"{self.name_strategy}_strategy.json")

        # Завантажити або створити файл налаштувань
        self.settings = self.load_data()

class FinancialSettingsManager(SettingsLoader):
    def __init__(self, module_name="FinancialAdvisor"):
        """Ініціалізує менеджер для фінансових налаштувань."""
        super().__init__(module_name)
        self.ensure_defaults_exist()

    def ensure_defaults_exist(self):
        """Створює дефолтні фінансові параметри, якщо файл порожній."""
        defaults = {
            "total_capital": 1000.0,
            "default_risk_per_trade_pct": 1.0,
            "leverage": 20,
            "trade_mode": "futures",
            "risk_reward_ratio": 1.5,
            "atr_multiplier": 2.0,
            "default_stop_loss_pct": 2.0
        }
        
        updated = False
        for key, default_value in defaults.items():
            if key not in self.settings:
                self.settings[key] = default_value
                updated = True
        
        if updated:
            self.save_data()

    def get_financial_settings(self):
        """Повертає словник з фінансовими налаштуваннями."""
        return self.settings
