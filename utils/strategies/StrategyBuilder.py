import os
import json
from typing import List, Dict, Any

# Використовуємо існуючий SettingsLoader для операцій з файлами
from utils.common.SettingsLoader import SettingsLoaderStrategies

class Strategy:
    """Клас даних для представлення торгової стратегії."""
    def __init__(self, name: str, goal: str = "", entry_conditions: List[Dict] = None, 
                 exit_conditions: List[Dict] = None, filters: List[Dict] = None, 
                 risk_params: Dict = None):
        self.name = name
        self.goal = goal
        self.entry_conditions = entry_conditions or []
        self.exit_conditions = exit_conditions or []
        self.filters = filters or []  # Фільтри поки не реалізовані в інтерпретаторі, але корисно їх мати
        self.risk_params = risk_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Серіалізує стратегію в словник."""
        return {
            "name": self.name,
            "goal": self.goal,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "filters": self.filters,
            "risk_params": self.risk_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        """Десеріалізує словник в об'єкт Strategy."""
        return cls(
            name=data.get("name", "Unnamed Strategy"),
            goal=data.get("goal", ""),
            entry_conditions=data.get("entry_conditions", []),
            exit_conditions=data.get("exit_conditions", []),
            filters=data.get("filters", []),
            risk_params=data.get("risk_params", {}),
        )

class StrategyBuilder:
    """
    Керує створенням, зміною та зберіганням торгових стратегій.
    Цей клас виступає як backend-логіка для конструктора стратегій.
    """
    def __init__(self, strategy_key: str = None):
        self.strategy: Strategy = None
        self.strategy_key = strategy_key
        self.settings_loader = None
        if strategy_key:
            self.load_strategy(strategy_key)

    def create_new(self, strategy_key: str, name: str, goal: str = ""):
        """Створює нову, порожню стратегію."""
        self.strategy_key = strategy_key
        self.strategy = Strategy(name=name, goal=goal)
        self.settings_loader = SettingsLoaderStrategies(
            module_name="Strategies",
            name_strategy=self.strategy_key
        )
        print(f"Нова стратегія '{name}' створена. Ключ: {strategy_key}")

    def load_strategy(self, strategy_key: str):
        """Завантажує існуючу стратегію з файлу."""
        self.strategy_key = strategy_key
        self.settings_loader = SettingsLoaderStrategies(
            module_name="Strategies",
            name_strategy=self.strategy_key
        )
        config = self.settings_loader.settings
        if not config:
            raise FileNotFoundError(f"Файл конфігурації для стратегії '{strategy_key}' не знайдено.")
        
        self.strategy = Strategy.from_dict(config)
        print(f"Стратегію '{self.strategy.name}' завантажено з файлу.")

    def add_rule(self, condition_type: str, rule_type: str, params: Dict):
        """
        Додає нове правило до стратегії.
        :param condition_type: 'entry' або 'exit'.
        :param rule_type: Тип логіки, напр., 'Crossover', 'Threshold'.
        :param params: Параметри для правила.
        """
        if not self.strategy:
            raise ValueError("Спочатку створіть або завантажте стратегію.")
            
        rule = {"type": rule_type, "params": params}
        
        if condition_type == 'entry':
            self.strategy.entry_conditions.append(rule)
            print(f"Додано умову входу: {rule}")
        elif condition_type == 'exit':
            self.strategy.exit_conditions.append(rule)
            print(f"Додано умову виходу: {rule}")
        else:
            raise ValueError("condition_type має бути 'entry' або 'exit'.")

    def set_risk_parameters(self, stop_loss: float = None, take_profit: float = None, **kwargs):
        """Встановлює параметри ризик-менеджменту."""
        if not self.strategy:
            raise ValueError("Спочатку створіть або завантажте стратегію.")
            
        if stop_loss is not None:
            self.strategy.risk_params['stop_loss'] = stop_loss
        if take_profit is not None:
            self.strategy.risk_params['take_profit'] = take_profit
        
        self.strategy.risk_params.update(kwargs)
        print(f"Оновлено параметри ризику: {self.strategy.risk_params}")

    def save(self):
        """Зберігає поточну конфігурацію стратегії в JSON-файл."""
        if not self.strategy or not self.settings_loader:
            raise ValueError("Немає активної стратегії для збереження.")
            
        self.settings_loader.settings = self.strategy.to_dict()
        self.settings_loader.save_data()
        print(f"Стратегію '{self.strategy.name}' збережено у файл: {self.settings_loader.settings_file}")
