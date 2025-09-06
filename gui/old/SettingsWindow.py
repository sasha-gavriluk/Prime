# gui/SettingsWindow.py

from kivy.uix.screenmanager import Screen
from kivy.clock import Clock

from gui.UIManager import UIManager
from utils.common.SettingsLoader import SettingsLoader, ProcessingSettingsBuilder

class SettingsWindow(Screen):
    def __init__(self, connection_controller, **kwargs):
        super().__init__(**kwargs)
        self.connection_controller = connection_controller
        self.settings_loader = SettingsLoader("GUI")
        self.ui = UIManager(canvas_size=(800, 600))
        
        self.pair_widgets = []
        self.tf_widgets = []
        self.indicator_widgets = []  # (box_widget, config_dict)
        self.current_indicator_configs = []
        self.pattern_checkboxes = []
        self.algorithm_checkboxes = []
        self.active_tf_for_config = None

        self.processing_settings = ProcessingSettingsBuilder()
        self.available_indicators = {}
        for item in self.processing_settings.get_indicator_settings():
            if item["name"] not in self.available_indicators:
                self.available_indicators[item["name"]] = item.get("parameters", {})

        self.build_ui()
        root = self.ui.build()
        self.add_widget(root)
        
        search_input = self.ui.registry["search_input"]
        search_input.bind(text=self.on_pair_search)

    def build_ui(self):
        # === КНОПКИ ===
        self.ui.add("btn_back", type="Button", text="<-- Назад", style_class="button_secondary")
        self.ui.set_bounds("btn_back", 10, 540, 105, 590)
        self.ui.set_action("btn_back", "on_press", self.go_back)

        self.ui.add("btn_reset", type="Button", text="Скинути налаштування", style_class="button_secondary")
        self.ui.set_bounds("btn_reset", 260, 540, 540, 590)
        self.ui.set_action("btn_reset", "on_press", self.reset_settings)

        self.ui.add("btn_advanced", type="Button", text="Розширені -->", style_class="button_primary")
        self.ui.set_bounds("btn_advanced", 670, 540, 790, 590)
        self.ui.set_action("btn_advanced", "on_press", self.go_to_advanced_settings)

        # === ВАЛЮТНІ ПАРИ ===
        self.ui.add("lbl_pairs", type="Label", text="Валютні пари", style_class="label_header")
        self.ui.set_bounds("lbl_pairs", 520, 10, 780, 40)
        self.ui.add("scroll_pairs", type="ScrollView")
        self.ui.set_bounds("scroll_pairs", 520, 50, 780, 240)
        self.ui.add("pairs_grid", type="GridLayout", parent="scroll_pairs", cols=1, size_hint_y=None)
        self.ui.set_size("pairs_grid", width=260, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["pairs_grid"].bind(minimum_height=self.ui.registry["pairs_grid"].setter("height")))

        self.ui.add("search_input", type="TextInput", hint_text="Пошук пари...", multiline=False)
        self.ui.set_bounds("search_input", 520, 250, 780, 280)

        # === ТАЙМФРЕЙМИ ===
        self.ui.add("lbl_tf", type="Label", text="Таймфрейми", style_class="label_header")
        self.ui.set_bounds("lbl_tf", 520, 310, 780, 340)
        self.ui.add("scroll_tf", type="ScrollView")
        self.ui.set_bounds("scroll_tf", 520, 350, 780, 540)
        self.ui.add("tf_grid", type="GridLayout", parent="scroll_tf", cols=1, size_hint_y=None)
        self.ui.set_size("tf_grid", width=260, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["tf_grid"].bind(minimum_height=self.ui.registry["tf_grid"].setter("height")))

        # === ІНДИКАТОРИ, ПАТЕРНИ, АЛГОРИТМИ ===
        self.ui.add("lbl_indicators", type="Label", text="Індикатори", style_class="label_header")
        self.ui.set_bounds("lbl_indicators", 10, 310, 400, 340)

        self.ui.add("btn_add_indicator", type="Button", text="Додати", style_class="button_primary")
        self.ui.set_bounds("btn_add_indicator", 420, 310, 510, 340)
        self.ui.set_action("btn_add_indicator", "on_press", self.open_add_indicator_popup)

        self.ui.add("scroll_indicators", type="ScrollView")
        self.ui.set_bounds("scroll_indicators", 10, 350, 510, 530)
        self.ui.add("grid_indicators", type="GridLayout", parent="scroll_indicators", spacing=10, cols=1, size_hint_y=None)
        self.ui.set_size("grid_indicators", width=500, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_indicators"].bind(minimum_height=self.ui.registry["grid_indicators"].setter("height")))

        self.ui.add("lbl_patterns", type="Label", text="Патерни", style_class="label_header")
        self.ui.set_bounds("lbl_patterns", 270, 10, 510, 40)
        self.ui.add("scroll_patterns", type="ScrollView")
        self.ui.set_bounds("scroll_patterns", 270, 50, 510, 300)
        self.ui.add("grid_patterns", type="GridLayout", parent="scroll_patterns", cols=1, size_hint_y=None)
        self.ui.set_size("grid_patterns", width=240, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_patterns"].bind(minimum_height=self.ui.registry["grid_patterns"].setter("height")))

        self.ui.add("lbl_algorithms", type="Label", text="Алгоритми", style_class="label_header")
        self.ui.set_bounds("lbl_algorithms", 10, 10, 250, 40)
        self.ui.add("scroll_algorithms", type="ScrollView")
        self.ui.set_bounds("scroll_algorithms", 10, 50, 250, 300)
        self.ui.add("grid_algorithms", type="GridLayout", parent="scroll_algorithms", cols=1, size_hint_y=None)
        self.ui.set_size("grid_algorithms", width=240, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_algorithms"].bind(minimum_height=self.ui.registry["grid_algorithms"].setter("height")))

    def go_to_advanced_settings(self, instance):
        self.save_settings(instance)
        self.manager.current = 'settings_plus'

    def on_enter(self, *args):
        self.settings_loader.reload()
        if not self.tf_widgets:
            self.add_timeframe_checkboxes()
            self.populate_symbols(self.settings_loader.get_nested_setting(['user_selections', 'pairs_top_ustd'], []))
            self.populate_pattern_checkboxes()
            self.populate_algorithm_checkboxes()

        self.restore_global_selections()
        self.force_show_all_pairs()

        active_tfs = self.settings_loader.get_nested_setting(['user_selections', 'selected_timeframes'], [])
        if active_tfs:
            first_active_tf = active_tfs[0]
            for tf, _, config_cb in self.tf_widgets:
                if tf == first_active_tf:
                    config_cb.active = True
                    break
        else:
            self.active_tf_for_config = None
            self._update_ui_for_tf(None)

    def restore_global_selections(self):
        saved_pairs = self.settings_loader.get_nested_setting(['user_selections', 'selected_pairs'], [])
        for _, checkbox, label in self.pair_widgets:
            checkbox.active = label.text in saved_pairs
        active_tfs = self.settings_loader.get_nested_setting(['user_selections', 'selected_timeframes'], [])
        for tf, activation_cb, config_cb in self.tf_widgets:
            activation_cb.active = tf in active_tfs

    def _populate_list(self, grid_id, data_source, widget_list, item_builder_func):
        widget_list.clear()
        grid = self.ui.registry[grid_id]
        grid.clear_widgets()
        for item in data_source:
            item_builder_func(grid, item, grid_id)

    def refresh_indicator_grid(self):
        grid = self.ui.registry["grid_indicators"]
        grid.clear_widgets()
        self.indicator_widgets.clear()
        for config in self.current_indicator_configs:
            self._add_indicator_row(config)

    def _add_indicator_row(self, config):
        grid = self.ui.registry["grid_indicators"]
        box = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, spacing=5)
        lbl = self.ui.dynamic.create("Label", text=config["name"])
        btn_config = self.ui.dynamic.create("Button", text="Налаштувати", size_hint_x=None, width=120, style_class="button_secondary")
        btn_remove = self.ui.dynamic.create("Button", text="Видалити", size_hint_x=None, width=80, style_class="button_secondary")
        btn_config.bind(on_press=lambda inst, c=config: self.open_indicator_config(c))
        btn_remove.bind(on_press=lambda inst, c=config, b=box: self.remove_indicator(c, b))
        box.add_widget(lbl); box.add_widget(btn_config); box.add_widget(btn_remove)
        grid.add_widget(box)
        self.indicator_widgets.append((box, config))

    def open_add_indicator_popup(self, instance):
        if not self.active_tf_for_config:
            return
        grid = self.ui.dynamic.create(
            "GridLayout", cols=1, spacing=10, padding=5, size_hint_y=None
        )
        grid.bind(minimum_height=grid.setter("height"))
        for name, params in self.available_indicators.items():
            row = self.ui.dynamic.create(
                "BoxLayout", orientation='horizontal', size_hint_y=None, height=40, spacing=10
            )
            lbl = self.ui.dynamic.create("Label", text=name)
            btn_add = self.ui.dynamic.create(
                "Button", text="Додати", size_hint_x=None, width=80, style_class="button_primary"
            )
            btn_add.bind(on_press=lambda inst, n=name: self.add_indicator(n))
            row.add_widget(lbl); row.add_widget(btn_add)
            grid.add_widget(row)
        scroll = self.ui.dynamic.create("ScrollView")
        scroll.add_widget(grid)
        popup = self.ui.dynamic.create(
            "Popup", title="Доступні індикатори", content=scroll, size_hint=(None, None), size=(300, 400)
        )
        popup.open()
        self._add_indicator_popup = popup

    def add_indicator(self, name):
        params = self.available_indicators.get(name, {}).copy()
        new_entry = {"name": name, "parameters": params}
        self.current_indicator_configs.append(new_entry)
        self._add_indicator_row(new_entry)
        self._save_single_tf_config(self.active_tf_for_config)
        if hasattr(self, '_add_indicator_popup'):
            self._add_indicator_popup.dismiss()

    def remove_indicator(self, config, box):
        if config in self.current_indicator_configs:
            self.current_indicator_configs.remove(config)
        grid = self.ui.registry["grid_indicators"]
        if box in grid.children:
            grid.remove_widget(box)
        self.indicator_widgets = [(b, c) for b, c in self.indicator_widgets if c != config]
        self._save_single_tf_config(self.active_tf_for_config)
        
    def populate_pattern_checkboxes(self):
        self._populate_list("grid_patterns", self.processing_settings.get_pattern_settings(), self.pattern_checkboxes, self._build_simple_row)

    def populate_algorithm_checkboxes(self):
        self._populate_list("grid_algorithms", self.processing_settings.get_algorithm_settings(), self.algorithm_checkboxes, self._build_simple_row)

    def _build_simple_row(self, grid, name, grid_id):
        box = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30)
        cb = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=40)
        cb.bind(active=lambda instance, value: self._save_single_tf_config(self.active_tf_for_config))
        lbl = self.ui.dynamic.create("Label", text=name)
        box.add_widget(cb); box.add_widget(lbl)
        grid.add_widget(box)
        if grid_id == "grid_patterns": self.pattern_checkboxes.append((cb, name))
        else: self.algorithm_checkboxes.append((cb, name))

    def populate_symbols(self, symbols):
        grid = self.ui.registry["pairs_grid"]
        grid.clear_widgets()
        self.pair_widgets.clear()
        for pair in symbols:
            pair_box = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30)
            checkbox = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=40)
            label = self.ui.dynamic.create("Label", text=pair)
            pair_box.add_widget(checkbox); pair_box.add_widget(label)
            grid.add_widget(pair_box)
            self.pair_widgets.append((pair_box, checkbox, label))
        grid.height = len(symbols) * 30

    def add_timeframe_checkboxes(self):
        grid = self.ui.registry["tf_grid"]
        grid.clear_widgets()
        self.tf_widgets.clear()
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        for tf in timeframes:
            tf_box = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30, spacing=5)
            activation_checkbox = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=40)
            label = self.ui.dynamic.create("Label", text=tf)
            config_checkbox = self.ui.dynamic.create("CheckBox", size_hint_x=None, width=40, group='tf_config_selection')

            activation_checkbox.bind(active=lambda i, v, t=tf, c=activation_checkbox: self._handle_tf_activation(t, c, v))
            config_checkbox.bind(active=lambda i, v, t=tf, c=config_checkbox: self._handle_tf_config_selection(t, c, v))

            tf_box.add_widget(activation_checkbox); tf_box.add_widget(label); tf_box.add_widget(config_checkbox)
            grid.add_widget(tf_box)
            self.tf_widgets.append((tf, activation_checkbox, config_checkbox))

    def on_pair_search(self, instance, value):
        search_value = value.strip().lower()
        visible_count = 0
        for pair_box, checkbox, label in self.pair_widgets:
            is_match = search_value in label.text.lower()
            pair_box.height, pair_box.opacity = (30, 1) if is_match else (0, 0)
            if is_match: visible_count += 1
        self.ui.registry["pairs_grid"].height = visible_count * 30

    def force_show_all_pairs(self): self.on_pair_search(None, "")
    
    def save_settings(self, instance):
        selected_pairs = [label.text for _, checkbox, label in self.pair_widgets if checkbox.active]
        self.settings_loader.update_nested_setting(['user_selections', 'selected_pairs'], selected_pairs)
        print("✅ Усі налаштування користувача збережено.")

    def reset_settings(self, instance=None):
        for _, checkbox, _ in self.pair_widgets: checkbox.active = False
        for _, act_cb, cfg_cb in self.tf_widgets: act_cb.active = False; cfg_cb.active = False
        self.current_indicator_configs.clear()
        self.refresh_indicator_grid()
        for cb, _ in self.pattern_checkboxes: cb.active = False
        for cb, _ in self.algorithm_checkboxes: cb.active = False
        for key in ['selected_pairs', 'selected_timeframes', 'selected_indicators', 'selected_patterns', 'selected_algorithms', 'tf_configurations']:
            self.settings_loader.delete_nested_setting(['user_selections', key])
        print("🔄 Усі вибрані налаштування скинуто.")
        self.force_show_all_pairs()

    def go_back(self, instance): self.manager.current = 'job'

    def open_indicator_config(self, indicator_config):
        if not self.active_tf_for_config:
            return
        
        name = indicator_config["name"]
        current_params = indicator_config.get("parameters", {})
        
        param_inputs = {}
        layout = self.ui.dynamic.create("BoxLayout", orientation='vertical', spacing=10, padding=10)
        for param, value in current_params.items():
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=30)
            label = self.ui.dynamic.create("Label", text=param, size_hint_x=0.4)
            input_field = self.ui.dynamic.create("TextInput", text=str(value), multiline=False)
            param_inputs[param] = input_field
            row.add_widget(label); row.add_widget(input_field)
            layout.add_widget(row)

        btn_row = self.ui.dynamic.create("BoxLayout", size_hint_y=None, height=40, spacing=10)
        btn_back = self.ui.dynamic.create("Button", text="Назад", style_class="button_secondary")
        btn_save = self.ui.dynamic.create("Button", text="Зберегти", style_class="button_primary")
        btn_row.add_widget(btn_back); btn_row.add_widget(btn_save)
        layout.add_widget(btn_row)

        popup = self.ui.dynamic.create("Popup", title=f"Налаштування {name} для {self.active_tf_for_config.upper()}",
                                     content=layout, size_hint=(None, None), size=(400, 400), auto_dismiss=False)
        
        def close_popup(instance):
            popup.dismiss()
        
        def save_params(instance):
            new_params = {}
            try:
                for param, input_field in param_inputs.items():
                    val = input_field.text.strip()
                    new_params[param] = float(val) if '.' in val else int(val)
            except ValueError: return
            
            indicator_config["parameters"] = new_params
            self._save_single_tf_config(self.active_tf_for_config)
            popup.dismiss()

        btn_back.bind(on_press=close_popup)
        btn_save.bind(on_press=save_params)
        popup.open()

    def on_pre_leave(self, *args):
        self.save_settings(instance=None)
        self.active_tf_for_config = None

    def _switch_active_tf_config(self, new_tf):
        self.active_tf_for_config = new_tf
        self._update_ui_for_tf(new_tf)

    def _update_ui_for_tf(self, tf):
        config = self.settings_loader.get_nested_setting(['user_selections', 'tf_configurations', tf], {}) if tf else {}
        
        self.current_indicator_configs = list(config.get('selected_indicators', []))
        self.refresh_indicator_grid()

        saved_patterns = config.get('selected_patterns', [])
        for cb, name in self.pattern_checkboxes: cb.active = name in saved_patterns
            
        saved_algorithms = config.get('selected_algorithms', [])
        for cb, name in self.algorithm_checkboxes: cb.active = name in saved_algorithms

    def _create_default_config(self):
        return {
            'selected_indicators': [],
            'selected_patterns': self.processing_settings.get_pattern_settings(),
            'selected_algorithms': self.processing_settings.get_algorithm_settings()
        }

    def _handle_tf_activation(self, tf, checkbox, is_active):
        configs = self.settings_loader.get_nested_setting(['user_selections', 'tf_configurations'], {})
        if is_active:
            if tf not in configs: configs[tf] = self._create_default_config()
        else:
            if tf in configs: del configs[tf]
            if self.active_tf_for_config == tf:
                self.active_tf_for_config = None
                self._update_ui_for_tf(None)
                checkbox.active = False
        
        self.settings_loader.update_nested_setting(['user_selections', 'tf_configurations'], configs)
        self.settings_loader.update_nested_setting(['user_selections', 'selected_timeframes'], list(configs.keys()))

    def _handle_tf_config_selection(self, tf, checkbox, is_active):
        if is_active and self.active_tf_for_config and self.active_tf_for_config != tf:
            self._save_single_tf_config(self.active_tf_for_config)
        if not is_active:
            if self.active_tf_for_config == tf: checkbox.active = True
            return

        activation_cb = next((act_cb for w_tf, act_cb, _ in self.tf_widgets if w_tf == tf), None)
        if activation_cb and not activation_cb.active: activation_cb.active = True
        self._switch_active_tf_config(tf)

    def _save_single_tf_config(self, tf_to_save):
        if not tf_to_save: return

        
        current_ui_config = {
            'selected_indicators': [dict(ind) for ind in self.current_indicator_configs],
            'selected_patterns': [name for cb, name in self.pattern_checkboxes if cb.active],
            'selected_algorithms': [name for cb, name in self.algorithm_checkboxes if cb.active]
        }
        all_tf_configs = self.settings_loader.get_nested_setting(['user_selections', 'tf_configurations'], {})
        all_tf_configs[tf_to_save] = current_ui_config
        self.settings_loader.update_nested_setting(['user_selections', 'tf_configurations'], all_tf_configs)

        # Оновлюємо глобальні списки для сумісності з іншими модулями
        all_indicators = []
        all_patterns = set()
        all_algorithms = set()
        for cfg in all_tf_configs.values():
            all_indicators.extend(cfg.get('selected_indicators', []))
            all_patterns.update(cfg.get('selected_patterns', []))
            all_algorithms.update(cfg.get('selected_algorithms', []))
        self.settings_loader.update_nested_setting(['user_selections', 'selected_indicators'], all_indicators)
        self.settings_loader.update_nested_setting(['user_selections', 'selected_patterns'], list(all_patterns))
        self.settings_loader.update_nested_setting(['user_selections', 'selected_algorithms'], list(all_algorithms))