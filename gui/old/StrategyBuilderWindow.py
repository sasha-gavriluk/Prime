# gui/StrategyBuilderWindow.py

import json
from kivy.uix.screenmanager import Screen

from gui.UIManager import UIManager
from utils.common.FileStructureManager import FileStructureManager
from utils.strategies.StrategyBuilder import StrategyBuilder

class StrategyBuilderWindow(Screen):
    """
    Вікно для візуального створення та редагування торгових стратегій.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fsm = FileStructureManager()
        # Backend-логіка для роботи зі стратегіями
        self.builder = StrategyBuilder()
        
        self.ui = UIManager(canvas_size=(800, 600))
        self.build_ui()
        self.add_widget(self.ui.build())

    def build_ui(self):
        # --- Верхня панель ---
        self.ui.add("btn_back", type="Button", text="<-- Назад до роботи", style_class="button_secondary")
        self.ui.set_bounds("btn_back", 10, 540, 200, 590)
        self.ui.set_action("btn_back", "on_press", lambda x: setattr(self.manager, 'current', 'job'))

        self.ui.add("btn_save", type="Button", text="Зберегти стратегію", style_class="button_primary")
        self.ui.set_bounds("btn_save", 650, 540, 790, 590)
        self.ui.set_action("btn_save", "on_press", self.save_strategy)

        # --- Ліва панель: Список існуючих стратегій ---
        self.ui.add("lbl_strategies", type="Label", text="Існуючі стратегії", style_class="label_header")
        self.ui.set_bounds("lbl_strategies", 10, 10, 250, 40)
        self.ui.add("scroll_strategies", type="ScrollView")
        self.ui.set_bounds("scroll_strategies", 10, 50, 250, 480)
        self.ui.add("grid_strategies", type="GridLayout", parent="scroll_strategies", cols=1, size_hint_y=None, spacing=5)
        self.ui.set_size("grid_strategies", height=0) # Висота буде динамічною

        self.ui.add("btn_new_strategy", type="Button", text="Нова стратегія", style_class="button_secondary")
        self.ui.set_bounds("btn_new_strategy", 10, 490, 250, 530)
        self.ui.set_action("btn_new_strategy", "on_press", self.create_new_strategy)

        # --- Центральна панель: Редактор ---
        self.ui.add("lbl_editor", type="Label", text="Редактор", style_class="label_header")
        self.ui.set_bounds("lbl_editor", 260, 10, 790, 40)
        
        self.ui.add("lbl_strategy_key", type="Label", text="Ключ (ім'я файлу):")
        self.ui.set_bounds("lbl_strategy_key", 270, 490, 420, 520)
        self.ui.add("input_strategy_key", type="TextInput", hint_text="Наприклад, my_rsi_strategy", multiline=False)
        self.ui.set_bounds("input_strategy_key", 430, 490, 780, 520)

        self.ui.add("lbl_config", type="Label", text="Конфігурація (JSON):")
        self.ui.set_bounds("lbl_config", 270, 450, 780, 480)
        self.ui.add("input_config_json", type="TextInput", text="{}", style_class="info_box")
        self.ui.set_bounds("input_config_json", 270, 50, 780, 440)

    def on_enter(self, *args):
        """Викликається при вході на екран, оновлює список стратегій."""
        self.populate_strategy_list()

    def populate_strategy_list(self):
        """Завантажує імена існуючих стратегій та створює для них кнопки."""
        grid = self.ui.registry["grid_strategies"]
        grid.clear_widgets()
        
        strategy_names = self.fsm.get_all_strategy_names()
        for name in strategy_names:
            btn = self.ui.dynamic.create(
                "Button", 
                text=name, 
                size_hint_y=None, 
                height=40, 
                style_class="button_secondary"
            )
            btn.bind(on_press=lambda instance, s_name=name: self.load_strategy_for_edit(s_name))
            grid.add_widget(btn)
        
        grid.height = len(strategy_names) * 45 # Динамічна висота

    def load_strategy_for_edit(self, strategy_key: str):
        """Завантажує дані стратегії в редактор."""
        try:
            self.builder.load_strategy(strategy_key)
            self.ui.registry['input_strategy_key'].text = strategy_key
            
            # Конвертуємо словник в гарно відформатований JSON для відображення
            strategy_json = json.dumps(self.builder.strategy.to_dict(), indent=4, ensure_ascii=False)
            self.ui.registry['input_config_json'].text = strategy_json
            
        except Exception as e:
            print(f"Помилка завантаження стратегії: {e}")
            self.ui.registry['input_config_json'].text = f"Помилка: {e}"

    def create_new_strategy(self, instance):
        """Очищує поля для створення нової стратегії."""
        self.ui.registry['input_strategy_key'].text = ""
        # Шаблон нової стратегії
        template = {
            "name": "New Strategy Name",
            "goal": "Description of the goal.",
            "timeframe_id": "1h",
            "required_indicators": [
                {"name": "RSI", "parameters": {"period": 14}}
            ],
            "entry_conditions": [
                {"type": "Threshold", "params": {"source": "RSI_14", "threshold": 30, "operator": "<"}}
            ],
            "exit_conditions": [
                {"type": "Threshold", "params": {"source": "RSI_14", "threshold": 70, "operator": ">"}}
            ]
        }
        self.ui.registry['input_config_json'].text = json.dumps(template, indent=4, ensure_ascii=False)
        self.builder = StrategyBuilder() # Скидаємо поточний білдер

    def save_strategy(self, instance):
        """Зберігає поточну конфігурацію з редактора у файл."""
        strategy_key = self.ui.registry['input_strategy_key'].text.strip()
        if not strategy_key:
            print("Помилка: Ключ стратегії (ім'я файлу) не може бути порожнім.")
            return

        try:
            # Парсимо JSON з поля вводу
            strategy_data = json.loads(self.ui.registry['input_config_json'].text)
            
            # Ініціалізуємо білдер
            self.builder.create_new(strategy_key=strategy_key, name=strategy_data.get("name", "Unnamed"))
            
            # Заповнюємо даними з JSON
            self.builder.strategy.goal = strategy_data.get("goal", "")
            self.builder.strategy.entry_conditions = strategy_data.get("entry_conditions", [])
            self.builder.strategy.exit_conditions = strategy_data.get("exit_conditions", [])
            
            # Додаємо додаткові поля, які важливі для роботи StrategyObject
            self.builder.strategy.to_dict()['timeframe_id'] = strategy_data.get('timeframe_id', '1h')
            self.builder.strategy.to_dict()['required_indicators'] = strategy_data.get('required_indicators', [])

            self.builder.save()
            print(f"Стратегія '{strategy_key}' успішно збережена.")
            
            # Оновлюємо список стратегій, щоб побачити нову
            self.populate_strategy_list()
            
        except json.JSONDecodeError:
            print("Помилка: Некоректний формат JSON в полі конфігурації.")
        except Exception as e:
            print(f"Невідома помилка при збереженні: {e}")