# gui/JobWindow.py

import asyncio
import datetime
import os
import sys
import time
import traceback
import pandas as pd
import math
import threading

from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.core.window import Window

from gui.UIManager import UIManager
from utils.analysis.DecisionOrchestrator import DecisionOrchestrator
from utils.analysis.TradeMonitor import TradeMonitor
from utils.common.SettingsLoader import SettingsLoader
from utils.common.FileStructureManager import FileStructureManager
from utils.data_processing.DataHandler import DataHandler
from utils.analysis.TimeframeAgent import TimeframeAgent
from utils.financial.FinancialAdvisor import FinancialAdvisor
from utils.common.HistoryDB import HistoryDB
from utils.nn.NeuralNetworkManager import NeuralNetworkManager

# --- Допоміжні класи для логіки (без змін) ---
class SignalConfigLoader:
    def __init__(self, gui_loader: SettingsLoader):
        from utils.common.SettingsLoader import DecisionSettingsManager, FinancialSettingsManager
        self.gui_loader = gui_loader
        self.decision_settings = DecisionSettingsManager()
        self.financial_settings = FinancialSettingsManager()
        self.pairs, self.active_timeframes = [], []
        self.metric_weights, self.timeframe_weights = {}, {}
        self.strategy_configs, self.agent_configs = {}, {}
        self.user_financial_settings = {}

    def load_all(self):
        self.gui_loader.reload()
        self.pairs = self.gui_loader.get_nested_setting(['user_selections', 'selected_pairs'], [])
        self.active_timeframes = self.gui_loader.get_nested_setting(['user_selections', 'selected_timeframes'], [])
        self._load_weights()
        self._load_strategies()
        self._load_agent_configs()
        self._load_financials()

    def _extract_weight(self, raw):
        if isinstance(raw, (int, float)): return float(raw)
        if isinstance(raw, dict) and 'weight' in raw:
            try: return float(raw['weight'])
            except (ValueError, TypeError): pass
        return 1.0

    def _load_weights(self):
        raw_metrics = self.gui_loader.get_nested_setting(['user_selections', 'selected_metric_weights'], {})
        raw_tfs = self.gui_loader.get_nested_setting(['user_selections', 'selected_timeframe_weights'], {})
        self.metric_weights = {m: self._extract_weight(w) for m, w in raw_metrics.items()}
        self.timeframe_weights = {tf: self._extract_weight(w) for tf, w in raw_tfs.items()}

    def _load_strategies(self):
        self.strategy_configs = self.gui_loader.get_nested_setting(['user_selections', 'selected_strategies'], {})

    def _load_agent_configs(self):
        gui_configs = self.gui_loader.get_nested_setting(['user_selections', 'tf_configurations'], {})
        default_config = self.decision_settings.get_agent_configurations().get("default", {})
        fast_config = self.decision_settings.get_agent_configurations().get("fast", default_config)
        self.agent_configs = {"gui": gui_configs, "default": default_config, "fast": fast_config}

    def _load_financials(self):
        defaults = self.financial_settings.get_financial_settings()
        settings = self.gui_loader.get_nested_setting(['user_selections', 'financial_settings'], defaults)
        for key, value in defaults.items():
            settings.setdefault(key, value)
        settings["open_trades"] = []
        self.user_financial_settings = settings

    def apply_nn_settings(self, nn_settings: dict):
        """Apply neural-network generated parameters to current config."""
        if not nn_settings:
            return
        changes = []

        # Update financial settings
        for key in ("default_risk_per_trade_pct", "risk_reward_ratio", "leverage"):
            if key in nn_settings:
                old_val = self.user_financial_settings.get(key)
                self.user_financial_settings[key] = nn_settings[key]
                changes.append(f"{key}: {old_val} -> {nn_settings[key]}")

        # Update metric weights if provided
        for key in ("smc_confidence", "pattern_score", "state_strength"):
            if key in nn_settings:
                old_val = self.metric_weights.get(key)
                self.metric_weights[key] = nn_settings[key]
                changes.append(f"{key}: {old_val} -> {nn_settings[key]}")

        # Update timeframe weights mapping
        tf_map = {"5m": "tf_5m", "1h": "tf_1h", "4h": "tf_4h"}
        for tf, nn_key in tf_map.items():
            if nn_key in nn_settings:
                old_val = self.timeframe_weights.get(tf)
                self.timeframe_weights[tf] = nn_settings[nn_key]
                changes.append(f"{tf}: {old_val} -> {nn_settings[nn_key]}")

        if changes:
            print("[NN] Applied settings:")
            for change in changes:
                print(f"  - {change}")
        else:
            print("[NN] Model produced no applicable settings")

class SignalProcessor:
    def __init__(self, config: SignalConfigLoader):
        self.config = config

    def generate(self, exchange: str, pair: str, raw_data_by_tf: dict, processed_data_by_tf: dict):
        timeframe_agents = []
        for tf, df in processed_data_by_tf.items():
            agent_config = self.config.agent_configs["gui"].get(tf, self.config.agent_configs["default"])
            timeframe_agents.append(TimeframeAgent(tf, df, agent_config))
        if not timeframe_agents:
            raise ValueError(f"Немає даних для аналізу по парі {pair} на біржі {exchange}.")
        orchestrator = DecisionOrchestrator(
            timeframe_agents=timeframe_agents, raw_data_by_tf=raw_data_by_tf,
            timeframe_weights=self.config.timeframe_weights, metric_weights=self.config.metric_weights,
            strategy_configs=self.config.strategy_configs, user_financial_settings=self.config.user_financial_settings,
            trade_mode=self.config.user_financial_settings.get('trade_mode', 'futures'), enable_news=False
        )
        return orchestrator.run()

class ResultFormatter:
    @staticmethod
    def format(result: dict, exchange: str, pair: str) -> str:
        tech = result.get('technical_decision', {})
        fin = result.get('financial_briefing', {})
        lines = [f"\n--- 📊 РЕЗУЛЬТАТ для [{exchange}] {pair} ---",
                 f"Технічний сигнал: {tech.get('direction', 'neutral').upper()} (Впевненість: {tech.get('confidence', 0.0):.2%})"]
        if fin.get("status") == "ok":
            lines.append(f"--- 💼 Фінансовий брифінг ({fin.get('trade_mode', 'N/A').upper()}) ---")
            params = fin.get('trade_parameters', {})
            lines.extend([f"Рекомендований ТФ для розрахунків: {fin.get('recommended_tf_for_financials', 'N/A')}",
                          f"  - Ризик на угоду: ${params.get('risk_per_trade_usd', 0):.2f}",
                          f"  - Stop Loss: {params.get('stop_loss_price', 0):.4f} | Take Profit: {params.get('take_profit_price', 0):.4f}"])
            if fin.get('trade_mode') == 'futures':
                lines.extend([f"  - Розмір позиції: ${params.get('position_size_usd', 0):,.2f} ({params.get('leverage')}x)",
                              f"  - Необхідна маржа: ${params.get('margin_required_usd', 0):.2f}",
                              f"  - Ціна ліквідації (прибл.): {params.get('liquidation_price', 0):.4f}"])
            if fin.get('commentary'): lines.append(f"Коментар: {fin.get('commentary')}")
        else:
            lines.append(f"⚠️ Фінансовий аналіз не проведено: {fin.get('reason', 'Невідома помилка.')}")
        lines.append("-" * 35)
        return "\n".join(lines)

# --- Основний клас вікна ---
class JobWindow(Screen):
    class _UIOutput:
        def __init__(self, callback): self.callback = callback
        def write(self, text):
            if text and not text.isspace(): self.callback(text.strip())
        def flush(self): pass

    def __init__(self, connection_controller, nn_manager: NeuralNetworkManager = None, **kwargs):
        super().__init__(**kwargs)
        self.connection_controller = connection_controller
        self.nn_manager = nn_manager
        self.settings_loader = SettingsLoader("GUI")
        self.fsm = FileStructureManager()
        self.financial_advisor = FinancialAdvisor()
        self.trade_monitor = None
        self.ui = UIManager(canvas_size=(800, 600))
        self.build_ui()
        self.add_widget(self.ui.build())
        self.info_box = None
        self.trade_signals = []

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_key_down)

    def build_ui(self):
        # --- Верхня панель ---
        self.ui.add("btn_back", type="Button", text="<-- Назад", style_class="button_secondary")
        self.ui.set_bounds("btn_back", 10, 540, 122, 590)
        self.ui.set_action("btn_back", "on_press", self.back_button_clicked)

        self.ui.add("btn_settings", type="Button", text="Налаштування", style_class="button_settings")
        self.ui.set_bounds("btn_settings", 128, 540, 280, 590)
        self.ui.set_action("btn_settings", "on_press", self.settings_button_clicked)

        self.ui.add("btn_clear_info", type="Button", text="Очистити інформацію", style_class="button_clear_info")
        self.ui.set_bounds("btn_clear_info", 285, 540, 495, 590)
        self.ui.set_action("btn_clear_info", "on_press", self.clear_info)
        
        self.ui.add("btn_history", type="Button", text="Історія угод", style_class="button_history")
        self.ui.set_bounds("btn_history", 500, 540, 650, 590)
        self.ui.set_action("btn_history", "on_press", lambda x: setattr(self.manager, 'current', 'history'))

        # --- Ліва панель управління ---
        self.ui.add("btn_load_data", type="Button", text="Завантажити дані", style_class="button_load_data")
        self.ui.set_bounds("btn_load_data", 10, 485, 280, 535)
        self.ui.set_action("btn_load_data", "on_press", self.load_data)

        self.ui.add("btn_process_data", type="Button", text="Обробка даних", style_class="button_process_data")
        self.ui.set_bounds("btn_process_data", 10, 430, 280, 480)
        self.ui.set_action("btn_process_data", "on_press", self.process_data)

        self.ui.add("btn_get_signal", type="Button", text="Отримати сигнал", style_class="button_get_signal")
        self.ui.set_bounds("btn_get_signal", 10, 375, 280, 425)
        self.ui.set_action("btn_get_signal", "on_press", self.get_signal)



    ############################################################################################################################
    #    self.ui.add("btn_strategy_builder", type="Button", text="Конструктор", style_class="button_primary")                  #
    #    self.ui.set_bounds("btn_strategy_builder", 260, 10, 280, 535) # Розмістимо її на вільне місце                         #
    #    self.ui.set_action("btn_strategy_builder", "on_press", self.go_to_strategy_builder)                                   #
    ############################################################################################################################
        


        # --- Панель торгових сигналів ---
        self.ui.add("lbl_signals", type="Label", text="Торгові Сигнали", style_class="label_header")
        self.ui.set_bounds("lbl_signals", 10, 10, 280, 40)
        
        self.ui.add("scroll_signals", type="ScrollView")
        self.ui.set_bounds("scroll_signals", 10, 40, 280, 370)
        
        self.ui.add("grid_signals", type="GridLayout", parent="scroll_signals", cols=1, size_hint_y=None, spacing=5)
        self.ui.set_size("grid_signals", width=260, height=0)
        Clock.schedule_once(lambda dt: self.ui.registry["grid_signals"].bind(minimum_height=self.ui.registry["grid_signals"].setter("height")))

        # --- Моніторинг ---
        self.ui.add("lbl_monitor", type="Label", text="Моніторинг:")
        self.ui.set_bounds("lbl_monitor", 655, 540, 755, 590)

        self.ui.add("cb_monitor", type="CheckBox")
        self.ui.set_bounds("cb_monitor", 755, 540, 790, 590)

        # --- Поле виводу інформації ---
        self.ui.add("info_box", type="TextInput", text="⚠️ Тут буде різна інформація ⚠️", readonly=True, style_class="info_box")
        self.ui.set_bounds("info_box", 290, 10, 790, 535)
        self.ui.set_size("info_box", size_hint=(None, None), width=500, height=525)

    def on_enter(self, *args):
        self.settings_loader.reload()
        self.update_load_data_button_state()
        self.ui.registry["btn_process_data"].disabled = True
        self.ui.registry["btn_get_signal"].disabled = True
        self.info_box = self.ui.registry["info_box"]
        self.trade_signals = [] 
        self._update_trade_signals_ui()
        monitor_checkbox = self.ui.registry['cb_monitor']
        monitor_checkbox.unbind(active=self.toggle_monitoring)
        monitor_checkbox.active = self.trade_monitor is not None
        monitor_checkbox.bind(active=self.toggle_monitoring)

    #def on_pre_leave(self, *args):
    #    if self.trade_monitor:
    #        threading.Thread(target=self._stop_trade_monitor, daemon=True).start()

    def toggle_monitoring(self, checkbox, is_active):
        if is_active:
            if not self.trade_monitor:
                self.trade_monitor = TradeMonitor(self.connection_controller, HistoryDB())
                self.trade_monitor.set_logger(self.append_info)
                threading.Thread(target=self.trade_monitor.start, daemon=True).start()
        elif self.trade_monitor:
            threading.Thread(target=self._stop_trade_monitor, daemon=True).start()

    def _stop_trade_monitor(self):
        if self.trade_monitor:
            self.trade_monitor.stop()
            self.trade_monitor = None

    def clear_info(self, instance):
        self.info_box.text = ""

    def back_button_clicked(self, instance): self.manager.current = 'start'
    def settings_button_clicked(self, instance): self.manager.current = 'settings'

    def append_info(self, text): Clock.schedule_once(lambda dt: self._do_append_info(text))
    def _do_append_info(self, text):
        self.info_box.text += f"\n{text}"

    def get_signal(self, instance):
        self.append_info("🔄 Отримання сигналів...")
        threading.Thread(target=self._get_signal_thread, daemon=True).start()

    def _get_signal_thread(self):
        try:
            print("[Job] Signal generation started")
            config_loader = SignalConfigLoader(self.settings_loader)
            config_loader.load_all()
            if not config_loader.pairs or not config_loader.active_timeframes:
                return self.append_info("⚠️ Виберіть пари та таймфрейми у налаштуваннях.")
            

            nn_enabled = self.settings_loader.get_nested_setting(
                ["user_selections", "neural_network", "enabled"], False
            )
            nn_runner = None
            print(f"[NN] enabled flag: {nn_enabled}")
            if nn_enabled and self.nn_manager and self.nn_manager.is_ready():
                try:
                    nn_runner = self.nn_manager.make_live_runner()
                    print("[NN] Live runner initialized")
                except Exception as e:
                    self.append_info(f"⚠️ Не вдалося ініціалізувати модель: {e}")
                    print(f"[NN] Runner init failed: {e}")
                    nn_enabled = False
            elif nn_enabled:
                self.append_info("⚠️ Модель нейромережі не завантажена.")
                print("[NN] Enabled but manager not ready")
                nn_enabled = False

            fsm = FileStructureManager()
            processor = SignalProcessor(config_loader)
            formatter = ResultFormatter()
            self.trade_signals.clear()

            for pair in config_loader.pairs:
                for exchange in self.connection_controller.get_connected_exchanges():
                    raw_data, proc_data = {}, {}
                    for tf in config_loader.active_timeframes:
                        key = f"{exchange}_{tf}_{pair.replace('/', '_')}"
                        files = fsm.get_all_files_in_directory(directory_key=key, extensions=['.csv'])
                        if not files: continue
                        proc_file = next((f for f in files if '_processing' in f.lower()), None)
                        raw_file = next((f for f in files if '_processing' not in f.lower()), None)
                        if proc_file: proc_data[tf] = pd.read_csv(proc_file)
                        if raw_file: raw_data[tf] = pd.read_csv(raw_file)
                    
                    if not proc_data:
                        self.append_info(f"⛔ [{exchange}] Не знайдено оброблених даних для {pair}.")
                        continue

                    if nn_enabled and nn_runner and proc_data:
                        try:
                            nn_runner.reset_buffers()
                            sample_df = next(iter(proc_data.values()))
                            if 'close' in sample_df.columns and len(sample_df) >= nn_runner.step_length:
                                print(f"[NN] Running model for {pair} on {exchange}")
                                nn_result = None
                                for close in sample_df['close'].tail(nn_runner.step_length):
                                    nn_result = nn_runner.on_new_candle(float(close))
                                if nn_result and nn_result.get('settings'):
                                    print(f"[NN] Output: {nn_result['settings']}")
                                    config_loader.apply_nn_settings(nn_result['settings'])
                                else:
                                    print("[NN] Model returned no settings")
                            else:
                                print(f"[NN] Insufficient data for model on {pair} {exchange}")
                        except Exception as e:
                            self.append_info(f"⚠️ Помилка роботи нейромережі: {e}")
                            print(f"[NN] Error while processing {pair} on {exchange}: {e}")

                    try:
                        result = processor.generate(exchange, pair, raw_data, proc_data)
                        if result.get("financial_briefing", {}).get("status") == "ok":
                            self.trade_signals.append({"exchange": exchange, "pair": pair, "result": result})
                        self.append_info(formatter.format(result, exchange, pair))
                    except Exception as e:
                        self.append_info(f"❌ Помилка аналізу {pair}: {e}\n{traceback.format_exc()}")
            
            Clock.schedule_once(self._update_trade_signals_ui)
        except Exception as e:
            self.append_info(f"❌ Критична помилка: {e}\n{traceback.format_exc()}")
        finally:
            print("[Job] Signal generation finished")
            Clock.schedule_once(lambda dt: (self.append_info("🔄 Робочий цикл завершено."),
                                            self.update_load_data_button_state(),
                                            setattr(self.ui.registry["btn_process_data"], 'disabled', True),
                                            setattr(self.ui.registry["btn_get_signal"], 'disabled', True)))

    def _update_trade_signals_ui(self, dt=None):
        grid = self.ui.registry["grid_signals"]
        grid.clear_widgets()
        sorted_signals = sorted(self.trade_signals, key=lambda x: x['result']['technical_decision'].get('confidence', 0), reverse=True)
        if not sorted_signals:
            grid.add_widget(self.ui.dynamic.create("Label", text="Немає активних сигналів.", size_hint_y=None, height=30))
            return

        for signal_data in sorted_signals:
            tech = signal_data['result']['technical_decision']
            direction = tech.get('direction', 'N/A').upper()
            confidence = tech.get('confidence', 0)
            
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=40, spacing=5)
            color = (0.1, 0.8, 0.1, 1) if direction == 'BUY' else (0.8, 0.1, 0.1, 1)
            
            info_text = f"{signal_data['pair']}\n{direction} ({confidence:.1%})"
            lbl = self.ui.dynamic.create("Label", text=info_text, color=color, halign='left', valign='middle')
            lbl.bind(size=lbl.setter('text_size'))

            btn = self.ui.dynamic.create("Button", text="Торгувати", size_hint_x=0.7, style_class="button_secondary")
            btn.bind(on_press=lambda x, data=signal_data: self._open_advanced_trade_popup(data))
            
            row.add_widget(lbl)
            row.add_widget(btn)
            grid.add_widget(row)
        
        grid.height = len(sorted_signals) * 45

    def _open_advanced_trade_popup(self, signal_data):
        content = self.ui.dynamic.create("BoxLayout", orientation='vertical', spacing=10, padding=10)
        briefing = signal_data['result']['financial_briefing']
        rec_params = briefing['trade_parameters']
        
        header = self.ui.dynamic.create("Label", text=f"{signal_data['pair']} - {signal_data['result']['technical_decision']['direction'].upper()}", font_size='18sp', size_hint_y=None, height=30)
        content.add_widget(header)

        # ... (Код для створення елементів попапу через self.ui.dynamic.create) ...
        # Цей код є довгим і повторюваним, але слідує одному патерну:
        # 1. Створити віджет: `widget = self.ui.dynamic.create(...)`
        # 2. Додати його до батьківського контейнера: `parent.add_widget(widget)`

        # --- SL/TP Mode ---
        mode_box = self.ui.dynamic.create("BoxLayout", size_hint_y=None, height=30)
        mode_box.add_widget(self.ui.dynamic.create("Label", text="Режим розрахунку:", bold=True, size_hint_x=0.5))
        sl_tp_auto_cb = self.ui.dynamic.create("CheckBox", group='sl_tp_mode', active=True, size_hint_x=0.1)
        mode_box.add_widget(sl_tp_auto_cb)
        mode_box.add_widget(self.ui.dynamic.create("Label", text="Авто", size_hint_x=0.2))
        sl_tp_manual_cb = self.ui.dynamic.create("CheckBox", group='sl_tp_mode', size_hint_x=0.1)
        mode_box.add_widget(sl_tp_manual_cb)
        mode_box.add_widget(self.ui.dynamic.create("Label", text="Ручний", size_hint_x=0.2))
        content.add_widget(mode_box)

        # --- Grid для SL/TP ---
        params_grid = self.ui.dynamic.create("GridLayout", cols=2, size_hint_y=None, height=120, spacing=(10, 5))
        sl_pct_input = self.ui.dynamic.create("TextInput", hint_text="%", multiline=False, disabled=True, input_filter='float')
        params_grid.add_widget(self.ui.dynamic.create("Label", text="Відсоток від ціни (SL):"))
        params_grid.add_widget(sl_pct_input)
        sl_display_label = self.ui.dynamic.create("Label", text=f"{rec_params.get('stop_loss_price', 0):.4f}")
        params_grid.add_widget(self.ui.dynamic.create("Label", text="Ціна SL:"))
        params_grid.add_widget(sl_display_label)
        tp_pct_input = self.ui.dynamic.create("TextInput", hint_text="%", multiline=False, disabled=True, input_filter='float')
        params_grid.add_widget(self.ui.dynamic.create("Label", text="Відсоток від ціни (TP):"))
        params_grid.add_widget(tp_pct_input)
        tp_display_label = self.ui.dynamic.create("Label", text=f"{rec_params.get('take_profit_price', 0):.4f}")
        params_grid.add_widget(self.ui.dynamic.create("Label", text="Ціна TP:"))
        params_grid.add_widget(tp_display_label)
        content.add_widget(params_grid)

        # --- Amount Settings ---
        amount_box = self.ui.dynamic.create("BoxLayout", orientation='vertical', size_hint_y=None, height=90)
        amount_box.add_widget(self.ui.dynamic.create("Label", text="Кількість:", bold=True))
        amount_mode_box = self.ui.dynamic.create("BoxLayout", size_hint_y=None, height=30)
        amount_auto_cb = self.ui.dynamic.create("CheckBox", group='amount_mode', active=True, size_hint_x=0.1)
        amount_mode_box.add_widget(amount_auto_cb)
        amount_mode_box.add_widget(self.ui.dynamic.create("Label", text="Авто", size_hint_x=0.2))
        amount_manual_cb = self.ui.dynamic.create("CheckBox", group='amount_mode', size_hint_x=0.1)
        amount_mode_box.add_widget(amount_manual_cb)
        amount_pct_input = self.ui.dynamic.create("TextInput", hint_text="% балансу", multiline=False, disabled=True, input_filter='float')
        amount_mode_box.add_widget(self.ui.dynamic.create("Label", text="Ручний (%):", size_hint_x=0.4))
        amount_mode_box.add_widget(amount_pct_input)
        amount_box.add_widget(amount_mode_box)
        amount_display_label = self.ui.dynamic.create("Label", text=f"Розмір: {rec_params.get('position_size_units', 0):.5f} ({rec_params.get('position_size_usd', 0):.2f}$)")
        amount_box.add_widget(amount_display_label)
        content.add_widget(amount_box)
        
        # --- Exchange Selection ---
        exchange_grid = self.ui.dynamic.create("GridLayout", cols=2, spacing=5, size_hint_y=None, height=60)
        exchange_checkboxes = {}
        for name in self.connection_controller.get_connected_exchanges().keys():
            row = self.ui.dynamic.create("BoxLayout", size_hint_y=None, height=30)
            cb = self.ui.dynamic.create("CheckBox", active=(name == signal_data['exchange']))
            row.add_widget(cb)
            row.add_widget(self.ui.dynamic.create("Label", text=name))
            exchange_grid.add_widget(row)
            exchange_checkboxes[name] = cb
        content.add_widget(self.ui.dynamic.create("Label", text="Виконати на біржах:"))
        content.add_widget(exchange_grid)

        # --- Buttons ---
        button_box = self.ui.dynamic.create("BoxLayout", size_hint_y=None, height=40, spacing=10)
        btn_execute = self.ui.dynamic.create("Button", text="Виконати", style_class="button_primary")
        btn_cancel = self.ui.dynamic.create("Button", text="Скасувати", style_class="button_secondary")
        button_box.add_widget(btn_execute); button_box.add_widget(btn_cancel)
        content.add_widget(button_box)

        popup = self.ui.dynamic.create("Popup", title="Виконання угоди", content=content, size_hint=(None, None), size=(450, 550), auto_dismiss=False)

        # --- Логіка та колбеки для попапу (без змін) ---
        financial_settings = self.settings_loader.get_nested_setting(['user_selections', 'financial_settings'], {})
        def update_calculations():
            capital_to_use = 0.0
            financial_settings = self.settings_loader.get_nested_setting(['user_selections', 'financial_settings'], {})
            capital_mode = financial_settings.get('capital_management_mode', 'auto')
            if capital_mode == 'manual':
                capital_to_use = financial_settings.get('manual_total_capital', 0.0)
            else:
                capital_to_use = self.connection_controller.get_api(signal_data['exchange']).get_usdt_balance()
            
            sl_price = 0.0
            try:
                if sl_tp_auto_cb.active:
                    sl_price = rec_params.get('stop_loss_price', 0)
                else:
                    pct = float(sl_pct_input.text)
                    sl_price = rec_params['entry_price'] * (1 - pct / 100.0) if signal_data['result']['technical_decision']['direction'] == 'buy' else rec_params['entry_price'] * (1 + pct / 100.0)
            except (ValueError, KeyError): pass
            sl_display_label.text = f"{sl_price:.4f}"

            tp_price = 0.0
            try:
                if sl_tp_auto_cb.active:
                    risk_per_unit = abs(rec_params['entry_price'] - sl_price) if sl_price > 0 else 0
                    rr_ratio = rec_params.get('risk_reward_ratio', 1.5)
                    if risk_per_unit > 0:
                        tp_price = rec_params['entry_price'] + risk_per_unit * rr_ratio if signal_data['result']['technical_decision']['direction'] == 'buy' else rec_params['entry_price'] - risk_per_unit * rr_ratio
                else:
                    tp_pct = float(tp_pct_input.text)
                    tp_price = rec_params['entry_price'] * (1 + tp_pct / 100.0) if signal_data['result']['technical_decision']['direction'] == 'buy' else rec_params['entry_price'] * (1 - tp_pct / 100.0)
            except (ValueError, KeyError): pass
            tp_display_label.text = f"{tp_price:.4f}"

            pos_size_units, pos_size_usd = 0, 0
            if amount_auto_cb.active:
                pos_size_units, pos_size_usd = rec_params.get('position_size_units', 0), rec_params.get('position_size_usd', 0)
            else:
                try:
                    pos_info = self.financial_advisor.calculate_futures_position_size(
                        capital=capital_to_use, risk_per_trade_pct=float(amount_pct_input.text),
                        entry_price=rec_params['entry_price'], stop_loss_price=sl_price, leverage=financial_settings.get('leverage', 20)
                    )
                    pos_size_units, pos_size_usd = pos_info.get('position_size_units', 0), pos_info.get('position_size_usd', 0)
                except (ValueError, KeyError): pass
            amount_display_label.text = f"Розмір: {pos_size_units:.5f} ({pos_size_usd:.2f}$)"
            return sl_price, tp_price, pos_size_units

        def on_sl_tp_mode_change(*args):
            is_manual = sl_tp_manual_cb.active
            sl_pct_input.disabled = not is_manual
            tp_pct_input.disabled = not is_manual
            update_calculations()
        
        def on_amount_mode_change(*args):
            is_manual = amount_manual_cb.active
            amount_pct_input.disabled = not is_manual
            if is_manual and not amount_pct_input.text:
                amount_pct_input.text = str(financial_settings.get('default_risk_per_trade_pct', 1.0))
            update_calculations()

        sl_tp_auto_cb.bind(active=on_sl_tp_mode_change)
        sl_tp_manual_cb.bind(active=on_sl_tp_mode_change)
        amount_auto_cb.bind(active=on_amount_mode_change)
        amount_manual_cb.bind(active=on_amount_mode_change)
        for input_field in [sl_pct_input, tp_pct_input, amount_pct_input]:
            input_field.bind(text=lambda *args: update_calculations())

        def execute_trade(instance):
            final_sl, final_tp, final_amount = update_calculations()
            if final_amount <= 0 or final_sl <= 0 or final_tp <= 0:
                return self.append_info("❌ Помилка: Некоректні параметри SL, TP або кількості.")
            selected_exchanges = [name for name, cb in exchange_checkboxes.items() if cb.active]
            if not selected_exchanges:
                return self.append_info("❌ Помилка: Виберіть хоча б одну біржу.")
            
            threading.Thread(target=self._execute_advanced_trade_thread, args=(
                selected_exchanges, signal_data['pair'], signal_data['result']['technical_decision']['direction'], final_amount, 
                final_sl, final_tp, int(financial_settings.get('leverage', 20))
            )).start()
            popup.dismiss()

        btn_execute.bind(on_press=execute_trade)
        btn_cancel.bind(on_press=popup.dismiss)
        popup.open()
        update_calculations() # Початковий розрахунок

    def _execute_advanced_trade_thread(self, exchanges, symbol, side, amount, stop_loss, take_profit, leverage):
        self.append_info(f"\n▶️ Розміщення {side.upper()} ордера на {amount:.5f} {symbol} з плечем {leverage}x...")
        self.append_info(f"   SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
        history_db = HistoryDB()
        current_settings = self.settings_loader.settings.get('user_selections', {})
        for ex_name in exchanges:
            try:
                api = self.connection_controller.get_api(ex_name)
                result = api.create_order(symbol=symbol, order_type='market', side=side, amount=amount, stop_loss=stop_loss, take_profit=take_profit, leverage=leverage)
                self.append_info(f"  ✅ [{ex_name}] Успіх! ID ордера: {result.get('id', 'N/A')}")
                
                entry_price = result.get('price')
                if not entry_price and result.get('cost') and result.get('filled') and result['filled'] > 0:
                    entry_price = result['cost'] / result['filled']
                if not entry_price:
                    self.append_info(f"  ⚠️ Не вдалося отримати ціну з ордера, запитуємо тікер...")
                    entry_price = api.client.fetch_ticker(symbol)['last']
                
                self.append_info(f"  ℹ️ Фактична ціна входу: {entry_price:.4f}")
                trade_data = {'exchange': ex_name, 'symbol': symbol, 'direction': side, 'status': 'open',
                              'entry_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              'entry_price': entry_price, 'size': amount, 'stop_loss': stop_loss,
                              'take_profit': take_profit, 'leverage': leverage}
                history_db.record_trade(trade_data, current_settings)
            except Exception as e:
                self.append_info(f"  ❌ [{ex_name}] Помилка! {e}")

    def update_load_data_button_state(self):
        pairs = self.settings_loader.get_nested_setting(['user_selections', 'selected_pairs'], [])
        tfs = self.settings_loader.get_nested_setting(['user_selections', 'selected_timeframes'], [])
        self.ui.registry["btn_load_data"].disabled = not (pairs and tfs)

    def load_data(self, instance):
        pairs = self.settings_loader.get_nested_setting(['user_selections', 'selected_pairs'], [])
        tfs = self.settings_loader.get_nested_setting(['user_selections', 'selected_timeframes'], [])
        if not pairs or not tfs:
            return self.append_info("⚠️ Немає вибраних пар або таймфреймів!")
        self.append_info(f"▶️ Завантаження даних для: {', '.join(pairs)}")
        threading.Thread(target=self.run_data_loading, args=(pairs, tfs), daemon=True).start()

    def run_data_loading(self, pairs, timeframes):
        exchanges = self.connection_controller.get_connected_exchanges()
        if not exchanges:
            return Clock.schedule_once(lambda dt: self.append_info("✖ Немає підключених бірж!"))
        for ex_name, api in exchanges.items():
            Clock.schedule_once(lambda dt, ex=ex_name: self.append_info(f"▶️ Завантаження з {ex}"))
            api.updates_symbols_data(pairs, timeframes)
        Clock.schedule_once(lambda dt: (self.append_info("✅ Завдання на завантаження даних завершено."),
                                        setattr(self.ui.registry["btn_load_data"], 'disabled', True),
                                        setattr(self.ui.registry["btn_process_data"], 'disabled', False)))

    def process_data(self, instance):
        self.append_info("🔄 Запуск обробки даних...")
        threading.Thread(target=self._process_data_thread, daemon=True).start()

    def _process_data_thread(self):
        output = self._UIOutput(self.append_info)
        sys.stdout, sys.stderr = output, output
        try:
            keys = self.generate_data_keys()
            if not keys: return
            fsm = FileStructureManager()
            indicators, patterns, algorithms, _, _ = self.get_user_processing_settings()
            handler = DataHandler()
            handler.set_custom_parameters(indicators, patterns, algorithms)
            found_files = []
            for key in keys:
                files = fsm.get_all_files_in_directory(directory_key=key, extensions=[".csv"])
                main_file = next((f for f in files if "_processing" not in os.path.basename(f)), None)
                if main_file: found_files.append(main_file)
            if not found_files: return self.append_info("⛔ Файлів для обробки не знайдено.")
            handler.files_list = found_files
            handler.process_all_files()
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            Clock.schedule_once(lambda dt: (self.append_info("✅ Обробка завершена."),
                                            setattr(self.ui.registry["btn_process_data"], 'disabled', True),
                                            setattr(self.ui.registry["btn_get_signal"], 'disabled', False)))

    def get_user_processing_settings(self):
        get = self.settings_loader.get_nested_setting
        return (get(['user_selections', 'selected_indicators'], []),
                get(['user_selections', 'selected_patterns'], []),
                get(['user_selections', 'selected_algorithms'], []),
                get(['user_selections', 'selected_timeframes'], []),
                get(['user_selections', 'selected_pairs'], []))
    
    def generate_data_keys(self):
        exchanges = self.connection_controller.get_connected_exchanges()
        if not exchanges: self.append_info("❌ Немає підключених бірж."); return []
        _, _, _, tfs, pairs = self.get_user_processing_settings()
        if not tfs or not pairs: self.append_info("⚠️ Виберіть пари і таймфрейми."); return []
        return [f"{ex}_{tf}_{p.replace('/', '_')}" for ex in exchanges for tf in tfs for p in pairs]
    
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard = None

    def _on_key_down(self, keyboard, keycode, text, modifiers):
        # Перевіряємо, чи натиснуто Ctrl+B
        if 'ctrl' in modifiers and keycode[1] == 'b':
            self.go_to_strategy_builder(None)
            return True # Повідомляємо Kivy, що ми обробили натискання
        return False

    def go_to_strategy_builder(self, instance):
        """Перемикає екран на конструктор стратегій."""
        self.manager.current = 'strategy_builder'