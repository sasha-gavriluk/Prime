from kivy.uix.screenmanager import Screen
from kivy.clock import Clock

from utils.common.SettingsLoader import (
    DecisionSettingsManager,
    SettingsLoader,
    FinancialSettingsManager,
)
from gui.UIManager import UIManager
from utils.common.SettingsLoader import DecisionSettingsManager, SettingsLoader, FinancialSettingsManager
from utils.common.FileStructureManager import FileStructureManager
from utils.nn.NeuralNetworkManager import get_controlled_parameters

NN_PARAMS = set(get_controlled_parameters())
NN_BLOCKED_TFS = {p.split("_")[1] for p in NN_PARAMS if p.startswith("tf_")}
NN_BLOCKED_METRICS = NN_PARAMS & {"smc_confidence", "pattern_score", "state_strength"}

class SettingsWindowPlus(Screen):
    def __init__(self, connection_controller, **kwargs):
        super().__init__(**kwargs)
        # Завантажувач для налаштувань, які змінює користувач
        self.settings_loader = SettingsLoader("GUI")
        self.connection_controller = connection_controller
        self.ui = UIManager(canvas_size=(800, 600))

        # Завантажувачі для отримання оригінальних (дефолтних) значень
        self.decision_settings = DecisionSettingsManager()
        self.news_fetcher_settings = SettingsLoader("NewsFetcher")
        self.news_analyzer_settings = SettingsLoader("NewsAnalyzer")
        self.fsm = FileStructureManager()
        self.financial_settings_manager = FinancialSettingsManager()

        # Списки для зберігання віджетів
        self.bot_weight_rows = []
        self.metric_weight_rows = []
        self.news_source_widgets = []
        self.news_multiplier_widgets = []
        self.strategy_widgets = []

        self.build_ui()
        root = self.ui.build()
        self.add_widget(root)

    def build_ui(self):
        # --- Верхня панель ---
        # Координати (y=540) вказують на верхню частину екрану, оскільки (0,0) - це верхній лівий кут.
        self.ui.add("btn_back", type="Button", text="<-- Назад", style_class="button_secondary")
        self.ui.set_bounds("btn_back", 10, 540, 105, 590)
        self.ui.set_action("btn_back", "on_press", self.go_back_to_settings)

        self.ui.add("btn_financial", type="Button", text="Фінанси", style_class="button_primary")
        self.ui.set_bounds("btn_financial", 115, 540, 250, 590)
        self.ui.set_action("btn_financial", "on_press", self.open_financial_settings_popup)

        # --- Перший ряд налаштувань ---
        self.ui.add("lbl_bot_weights", type="Label", text="Ваги таймфреймів", style_class="label_header")
        self.ui.set_bounds("lbl_bot_weights", 10, 270, 260, 300)
        self.ui.add("scroll_bot_weights", type="ScrollView")
        self.ui.set_bounds("scroll_bot_weights", 10, 300, 260, 530)
        self.ui.add("grid_bot_weights", type="GridLayout", parent="scroll_bot_weights", cols=1, size_hint_y=None)
        self.ui.set_size("grid_bot_weights", width=240, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_bot_weights"].bind(minimum_height=self.ui.registry["grid_bot_weights"].setter("height")))

        self.ui.add("lbl_metric_weights", type="Label", text="Ваги метрик", style_class="label_header")
        self.ui.set_bounds("lbl_metric_weights", 270, 270, 520, 300)
        self.ui.add("scroll_metric_weights", type="ScrollView")
        self.ui.set_bounds("scroll_metric_weights", 270, 300, 520, 530)
        self.ui.add("grid_metric_weights", type="GridLayout", parent="scroll_metric_weights", cols=1, size_hint_y=None)
        self.ui.set_size("grid_metric_weights", width=240, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_metric_weights"].bind(minimum_height=self.ui.registry["grid_metric_weights"].setter("height")))

        # --- Другий ряд налаштувань (для новин) ---
        # Координати (y=10) вказують на нижню частину екрану.
        self.ui.add("lbl_news_sources", type="Label", text="Джерела новин (NewsFetcher)", style_class="label_header")
        self.ui.set_bounds("lbl_news_sources", 10, 10, 380, 40)
        self.ui.add("scroll_news_sources", type="ScrollView")
        self.ui.set_bounds("scroll_news_sources", 10, 40, 380, 270)
        self.ui.add("grid_news_sources", type="GridLayout", parent="scroll_news_sources", cols=1, size_hint_y=None)
        self.ui.set_size("grid_news_sources", width=360, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_news_sources"].bind(minimum_height=self.ui.registry["grid_news_sources"].setter("height")))

        self.ui.add("lbl_news_multipliers", type="Label", text="Вплив новин (NewsAnalyzer)", style_class="label_header")
        self.ui.set_bounds("lbl_news_multipliers", 390, 10, 780, 40)
        self.ui.add("scroll_news_multipliers", type="ScrollView")
        self.ui.set_bounds("scroll_news_multipliers", 390, 40, 780, 270)
        self.ui.add("grid_news_multipliers", type="GridLayout", parent="scroll_news_multipliers", cols=1, size_hint_y=None)
        self.ui.set_size("grid_news_multipliers", width=380, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_news_multipliers"].bind(minimum_height=self.ui.registry["grid_news_multipliers"].setter("height")))

        self.ui.add("lbl_strategies", type="Label", text="Активні стратегії та їх ваги", style_class="label_header")
        self.ui.set_bounds("lbl_strategies", 530, 270, 780, 300)
        self.ui.add("scroll_strategies", type="ScrollView")
        self.ui.set_bounds("scroll_strategies", 530, 300, 780, 530)
        self.ui.add("grid_strategies", type="GridLayout", parent="scroll_strategies", cols=1, size_hint_y=None)
        self.ui.set_size("grid_strategies", width=240, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_strategies"].bind(minimum_height=self.ui.registry["grid_strategies"].setter("height")))

    def on_enter(self, *args):
        """Викликається при вході на екран."""
        self.settings_loader.reload()
        self.decision_settings.reload()
        self.news_fetcher_settings.reload()
        self.news_analyzer_settings.reload()

        self.nn_enabled = self.settings_loader.get_nested_setting(
            ["user_selections", "neural_network", "enabled"], False
        )
        
        self._synchronize_settings()
        self._synchronize_financial_settings()

        self.populate_bot_weight_inputs()
        self.restore_saved_weights()
        
        self.populate_metric_weight_inputs()
        self.restore_saved_metric_weights()

        self.populate_news_source_settings()
        self.restore_news_source_settings()

        self.populate_news_multiplier_settings()
        self.restore_news_multiplier_settings()

        self.populate_strategy_widgets()
        self.restore_strategy_settings()

    def _synchronize_financial_settings(self):
        """Копіює дефолтні фінансові налаштування в GUI.json, якщо їх там немає."""
        if self.settings_loader.get_nested_setting(['user_selections', 'financial_settings']) is None:
            default_financials = self.financial_settings_manager.get_financial_settings()
            self.settings_loader.update_nested_setting(['user_selections', 'financial_settings'], default_financials)
            print("INFO: Додано дефолтні фінансові налаштування у GUI.")

    def _synchronize_settings(self):
        """
        Перевіряє, чи існують налаштування у файлі GUI.
        Якщо ні, копіює їх з дефолтних файлів налаштувань.
        """
        # --- Синхронізація ваг таймфреймів ---
        if self.settings_loader.get_nested_setting(['user_selections', 'selected_timeframe_weights']) is None:
            default_weights = self.decision_settings.get_timeframe_weights()
            formatted_weights = {tf: {'weight': w} for tf, w in default_weights.items()}
            self.settings_loader.update_nested_setting(['user_selections', 'selected_timeframe_weights'], formatted_weights)
            print("INFO: Додано дефолтні ваги таймфреймів у GUI налаштування.")

        # --- Синхронізація ваг метрик ---
        if self.settings_loader.get_nested_setting(['user_selections', 'selected_metric_weights']) is None:
            default_metrics = self.decision_settings.get_metric_weights()
            formatted_metrics = {metric: {'weight': w} for metric, w in default_metrics.items()}
            self.settings_loader.update_nested_setting(['user_selections', 'selected_metric_weights'], formatted_metrics)
            print("INFO: Додано дефолтні ваги метрик у GUI налаштування.")
        
        # --- Синхронізація джерел новин ---
        if self.settings_loader.get_nested_setting(['user_selections', 'news_sources']) is None:
            default_sources = self.news_fetcher_settings.settings.get("sources", [])
            self.settings_loader.update_nested_setting(['user_selections', 'news_sources'], default_sources)
            print("INFO: Додано дефолтні джерела новин у GUI налаштування.")

        # --- Синхронізація коефіцієнтів впливу новин ---
        if self.settings_loader.get_nested_setting(['user_selections', 'news_multipliers']) is None:
            default_multipliers = self.news_analyzer_settings.settings.get("impact_multipliers", {})
            self.settings_loader.update_nested_setting(['user_selections', 'news_multipliers'], default_multipliers)
            print("INFO: Додано дефолтні коефіцієнти впливу новин у GUI налаштування.")

        # Перевіряємо, чи є налаштування для стратегій в GUI.json
        if self.settings_loader.get_nested_setting(['user_selections', 'selected_strategies']) is None:
            # Якщо немає, знаходимо всі доступні стратегії
            available_strategies = self.fsm.get_all_strategy_names()
            # І створюємо для них дефолтні налаштування (вимкнені, вага 1.0)
            default_strategies = {
                name: {'enabled': False, 'weight': 1.0} for name in available_strategies
            }
            self.settings_loader.update_nested_setting(['user_selections', 'selected_strategies'], default_strategies)
            print("INFO: Додано дефолтні налаштування для знайдених стратегій у GUI.")

    def go_back_to_settings(self, instance):
        """Зберігає всі налаштування та повертається на попередній екран."""
        self.save_all_settings()
        self.manager.current = 'settings'

    def save_all_settings(self):
        """Централізований виклик збереження для всіх секцій."""
        self.save_weights_settings()
        self.save_news_source_settings()
        self.save_news_multiplier_settings()
        self.save_strategy_settings()
        print("✅ Усі розширені налаштування збережено.")

    # --- Методи для ваг бота та метрик ---
    def populate_bot_weight_inputs(self):
        grid = self.ui.registry["grid_bot_weights"]
        grid.clear_widgets()
        self.bot_weight_rows = []
        default_weights = self.decision_settings.get_timeframe_weights()
        for tf in default_weights:
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, spacing=5)
            cb = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=30)
            label = self.ui.dynamic.create("Label", text=tf, size_hint_x=None, width=80)
            input_field = self.ui.dynamic.create("TextInput", text="0.0", multiline=False, size_hint_x=None, width=100, disabled=True, style_class="text_input_main")
            if self.nn_enabled and tf in NN_BLOCKED_TFS:
                cb.disabled = True
                input_field.disabled = True
            row.add_widget(cb)
            row.add_widget(label)
            row.add_widget(input_field)
            grid.add_widget(row)
            self.bot_weight_rows.append((cb, tf, input_field))
        grid.bind(minimum_height=grid.setter("height"))

    def populate_metric_weight_inputs(self):
        grid = self.ui.registry["grid_metric_weights"]
        grid.clear_widgets()
        self.metric_weight_rows = []
        default_weights = self.decision_settings.get_metric_weights()
        for metric in default_weights:
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, spacing=5)
            cb = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=30)
            label = self.ui.dynamic.create("Label", text=metric, size_hint_x=None, width=120)
            input_field = self.ui.dynamic.create("TextInput", text="0.0", multiline=False, size_hint_x=None, width=100, disabled=True, style_class="text_input_main")
            if self.nn_enabled and metric in NN_BLOCKED_METRICS:
                cb.disabled = True
                input_field.disabled = True
            row.add_widget(cb)
            row.add_widget(label)
            row.add_widget(input_field)
            grid.add_widget(row)
            self.metric_weight_rows.append((cb, metric, input_field))
        grid.bind(minimum_height=grid.setter("height"))

    def restore_saved_weights(self):
        saved_weights = self.settings_loader.get_nested_setting(['user_selections', 'selected_timeframe_weights'], {})
        for cb, tf, input_field in self.bot_weight_rows:
            saved_entry = saved_weights.get(tf)
            is_active = saved_entry is not None
            weight_value = saved_entry.get('weight') if is_active else 0.0
            
            cb.active = is_active
            cb.disabled = self.nn_enabled and tf in NN_BLOCKED_TFS
            input_field.text = str(weight_value)
            input_field.disabled = (not is_active) or (self.nn_enabled and tf in NN_BLOCKED_TFS)
            
            def on_active(instance, value, tf=tf, input_field=input_field, saved_weights=saved_weights):
                default_weight = self.decision_settings.get_timeframe_weights().get(tf, 0.0)
                if value:
                    # При активації беремо збережене значення, якщо воно є, інакше дефолтне
                    current_value = saved_weights.get(tf, {'weight': default_weight}).get('weight')
                    input_field.text = str(current_value)
                    input_field.disabled = False
                else:
                    input_field.text = "0.0"
                    input_field.disabled = True
            cb.unbind(active=on_active)
            if not (self.nn_enabled and tf in NN_BLOCKED_TFS):
                cb.bind(active=on_active)

    def restore_saved_metric_weights(self):
        saved = self.settings_loader.get_nested_setting(['user_selections', 'selected_metric_weights'], {})
        for cb, metric, input_field in self.metric_weight_rows:
            saved_entry = saved.get(metric)
            is_active = saved_entry is not None
            weight_value = saved_entry.get('weight') if is_active else 0.0

            cb.active = is_active
            cb.disabled = self.nn_enabled and metric in NN_BLOCKED_METRICS
            input_field.text = str(weight_value)
            input_field.disabled = (not cb.active) or (self.nn_enabled and metric in NN_BLOCKED_METRICS)
            
            def on_active(inst, val, metric=metric, inp=input_field, saved=saved):
                default_weight = self.decision_settings.get_metric_weights().get(metric, 0.0)
                if val:
                    current_value = saved.get(metric, {'weight': default_weight}).get('weight')
                    inp.text = str(current_value)
                    inp.disabled = False
                else:
                    inp.text = "0.0"
                    inp.disabled = True
            cb.unbind(active=on_active)
            if not (self.nn_enabled and metric in NN_BLOCKED_METRICS):
                cb.bind(active=on_active)

    def save_weights_settings(self):
        tf_selected = {}
        for cb, tf, input_field in self.bot_weight_rows:
            if not cb.active: continue
            try:
                tf_selected[tf] = {"weight": float(input_field.text.strip())}
            except ValueError: pass
        self.settings_loader.update_nested_setting(['user_selections', 'selected_timeframe_weights'], tf_selected)

        metric_selected = {}
        for cb, metric, input_field in self.metric_weight_rows:
            if not cb.active: continue
            try:
                metric_selected[metric] = {"weight": float(input_field.text.strip())}
            except ValueError: pass
        self.settings_loader.update_nested_setting(['user_selections', 'selected_metric_weights'], metric_selected)
    
    # --- Методи для налаштувань новин ---
    def populate_news_source_settings(self):
        grid = self.ui.registry["grid_news_sources"]
        grid.clear_widgets()
        self.news_source_widgets = []
        
        # Список віджетів будуємо на основі дефолтних налаштувань, щоб бачити всі можливі джерела
        sources = self.news_fetcher_settings.settings.get("sources", [])
        for source in sources:
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, style_class="text_input_main")
            cb = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=40)
            label = self.ui.dynamic.create("Label", text=source.get("name", "Unknown Source"))
            row.add_widget(cb)
            row.add_widget(label)
            grid.add_widget(row)
            # Зберігаємо оригінальну конфігурацію джерела разом з чекбоксом
            self.news_source_widgets.append((cb, source))
        grid.bind(minimum_height=grid.setter("height"))

    def restore_news_source_settings(self):
        # Стан чекбоксів відновлюємо з GUI.json
        gui_sources = self.settings_loader.get_nested_setting(['user_selections', 'news_sources'], [])
        gui_sources_dict = {src['name']: src.get('enabled', False) for src in gui_sources}

        for cb, source_config in self.news_source_widgets:
            source_name = source_config.get("name")
            cb.active = gui_sources_dict.get(source_name, False)

    def save_news_source_settings(self):
        # Зберігаємо стан чекбоксів у GUI.json
        gui_sources = self.settings_loader.get_nested_setting(['user_selections', 'news_sources'], [])
        gui_sources_dict = {src['name']: src for src in gui_sources}

        for cb, source_config in self.news_source_widgets:
            source_name = source_config.get("name")
            if source_name in gui_sources_dict:
                gui_sources_dict[source_name]['enabled'] = cb.active
        
        # Оновлюємо список об'єктів у налаштуваннях GUI
        self.settings_loader.update_nested_setting(['user_selections', 'news_sources'], list(gui_sources_dict.values()))

    def populate_news_multiplier_settings(self):
        grid = self.ui.registry["grid_news_multipliers"]
        grid.clear_widgets()
        self.news_multiplier_widgets = []
        
        # Будуємо UI на основі дефолтних ключів
        multipliers = self.news_analyzer_settings.settings.get("impact_multipliers", {})
        for key, value in multipliers.items():
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, spacing=10)
            label = self.ui.dynamic.create("Label", text=key, size_hint_x=0.6)
            input_field = self.ui.dynamic.create("TextInput", text=str(value), multiline=False, size_hint_x=0.4, disabled=False, style_class="text_input_main")
            row.add_widget(label)
            row.add_widget(input_field)
            grid.add_widget(row)
            self.news_multiplier_widgets.append((key, input_field))
        grid.bind(minimum_height=grid.setter("height"))

    def restore_news_multiplier_settings(self):
        # Відновлюємо значення з GUI.json
        gui_multipliers = self.settings_loader.get_nested_setting(['user_selections', 'news_multipliers'], {})
        for key, input_field in self.news_multiplier_widgets:
            input_field.text = str(gui_multipliers.get(key, 1.0))

    def save_news_multiplier_settings(self):
        # Зберігаємо значення у GUI.json
        gui_multipliers = {}
        for key, input_field in self.news_multiplier_widgets:
            try:
                gui_multipliers[key] = float(input_field.text.strip())
            except (ValueError, TypeError):
                # Якщо значення некоректне, беремо дефолтне
                default_val = self.news_analyzer_settings.settings.get("impact_multipliers", {}).get(key, 1.0)
                gui_multipliers[key] = default_val
                print(f"⚠️ Некоректне значення для коефіцієнта '{key}'. Встановлено дефолтне значення.")
        
        self.settings_loader.update_nested_setting(['user_selections', 'news_multipliers'], gui_multipliers)

    def populate_strategy_widgets(self):
        grid = self.ui.registry["grid_strategies"]
        grid.clear_widgets()
        self.strategy_widgets.clear()

        available_strategies = self.fsm.get_all_strategy_names()
        if not available_strategies:
            grid.add_widget(self.ui.dynamic.create("Label", text="Не знайдено файлів стратегій\nв 'data/Strategies/'", size_hint_y=None, height=40))
            return
            
        for name in available_strategies:
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, spacing=5)
            cb = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=30)
            label = self.ui.dynamic.create("Label", text=name, size_hint_x=0.6)
            input_field = self.ui.dynamic.create("TextInput", text="1.0", multiline=False, size_hint_x=0.3, disabled=not cb.active, style_class="text_input_main")
            
            def toggle_input(checkbox, value, text_input=input_field):
                text_input.disabled = not value
            cb.bind(active=toggle_input)

            row.add_widget(cb)
            row.add_widget(label)
            row.add_widget(input_field)
            grid.add_widget(row)
            self.strategy_widgets.append({'name': name, 'checkbox': cb, 'input': input_field})
            
        grid.bind(minimum_height=grid.setter("height"))

    def restore_strategy_settings(self):
        saved_strategies = self.settings_loader.get_nested_setting(['user_selections', 'selected_strategies'], {})
        for widget_info in self.strategy_widgets:
            strat_name = widget_info['name']
            config = saved_strategies.get(strat_name, {'enabled': False, 'weight': 1.0})
            
            widget_info['checkbox'].active = config.get('enabled', False)
            widget_info['input'].text = str(config.get('weight', 1.0))
            widget_info['input'].disabled = not config.get('enabled', False)

    def save_strategy_settings(self):
        strategies_to_save = {}
        for widget_info in self.strategy_widgets:
            name = widget_info['name']
            is_enabled = widget_info['checkbox'].active
            try:
                weight = float(widget_info['input'].text.strip())
            except ValueError:
                weight = 1.0
            
            strategies_to_save[name] = {'enabled': is_enabled, 'weight': weight}
        
        self.settings_loader.update_nested_setting(['user_selections', 'selected_strategies'], strategies_to_save)

    def open_financial_settings_popup(self, instance):
        """Створює та показує спливаюче вікно для налаштування ризик-менеджменту."""
        nn_enabled = self.settings_loader.get_nested_setting(
            ["user_selections", "neural_network", "enabled"], False
        )

        content = self.ui.dynamic.create("BoxLayout", orientation='vertical', spacing=10, padding=10)
        grid = self.ui.dynamic.create("GridLayout", cols=2, spacing=10, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))

        current_settings = self.settings_loader.get_nested_setting(['user_selections', 'financial_settings'], {})
        
        param_inputs = {}
        setting_keys = [
            "total_capital", "default_risk_per_trade_pct", "leverage", "risk_reward_ratio", 
            "atr_multiplier", "default_stop_loss_pct", "trade_mode"
        ]
        blocked_keys = {"default_risk_per_trade_pct", "leverage", "risk_reward_ratio"}

        for key in setting_keys:
            # --- ОСНОВНЕ ВИПРАВЛЕННЯ ТУТ ---
            # Встановлюємо size_hint_y=None та фіксовану висоту для кожного елемента
            label = self.ui.dynamic.create("Label", text=key.replace('_', ' ').capitalize(), size_hint_y=None, height=30)
            text_input = self.ui.dynamic.create(
                "TextInput",
                text=str(current_settings.get(key, '')),
                multiline=False,
                size_hint_y=None,
                height=30,
                style_class="text_input_main",
                disabled=nn_enabled and key in blocked_keys,
            )

            grid.add_widget(label)
            grid.add_widget(text_input)
            param_inputs[key] = text_input

        content.add_widget(grid)

        button_layout = self.ui.dynamic.create("BoxLayout", size_hint_y=None, height=40, spacing=10)
        btn_save = self.ui.dynamic.create("Button", text="Зберегти")
        btn_cancel = self.ui.dynamic.create("Button", text="Скасувати")
        button_layout.add_widget(btn_save)
        button_layout.add_widget(btn_cancel)
        content.add_widget(button_layout)

        popup = self.ui.dynamic.create(
            "Popup",
            title="Налаштування ризик-менеджменту",
            content=content,
            size_hint=(None, None),
            size=(500, 400),
            auto_dismiss=False,
        )

        def save_and_close(instance):
            new_settings = {}
            try:
                for key, input_widget in param_inputs.items():
                    value = input_widget.text.strip()
                    if key != 'trade_mode':
                        new_settings[key] = float(value)
                    else:
                        new_settings[key] = value.lower()
            except ValueError:
                print("ПОМИЛКА: Некоректне значення. Введіть число для числових полів.")
                return

            self.settings_loader.update_nested_setting(['user_selections', 'financial_settings'], new_settings)
            print("✅ Фінансові налаштування збережено.")
            popup.dismiss()

        btn_save.bind(on_press=save_and_close)
        btn_cancel.bind(on_press=popup.dismiss)
        popup.open()
