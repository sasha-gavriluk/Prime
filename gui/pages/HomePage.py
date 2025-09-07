# pages/home_blocks.py
import threading
from datetime import datetime, timezone

from kivy.clock import Clock

from utils.common.SettingsLoader import DecisionSettingsManager, FinancialSettingsManager, SettingsLoader
from utils.analysis.DecisionOrchestrator import DecisionOrchestrator
from utils.analysis.TimeframeAgent import TimeframeAgent 

class SignalConfigLoader:
    def __init__(self, gui_loader: SettingsLoader):
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

class HomePage:
   TEXT_CONNECTION = "Підключено"
   TEXT_DISCONNECTED = "Відключено"

   def __init__(self, ui, connection_controller, fms, history_db):
      self.ui = ui
      self.fms = fms
      self.history_db = history_db

      self.cc = connection_controller
      self.exchanges = ["Binance", "Bybit"]

   def add_home_build(self, page_id):
      """
      ui  — ваш UIManager
      page_id — id віджета-контейнера сторінки (наприклад, "home_page"), який уже створено у вашому app_ui.py
      Повертає id кореневого контейнера блоків ("blocks").
      """

      ID_WIDGETS_BLOCKS = "blocks"
      self.ui.add(ID_WIDGETS_BLOCKS, "BoxLayout", parent=page_id, orientation="horizontal", padding=(12, 12, 12, 12), spacing=12, size_hint=(1, 1))

      # ============================================== ЛІВИЙ СТОВПЕЦЬ =================================================
      self.ui.add("left_col", "BoxLayout", parent=ID_WIDGETS_BLOCKS,orientation="vertical", spacing=12, size_hint=(0.6, 1))

      # ================ БЛОК 1 ==================== 
      self.ui.add("connection", "BoxLayout", parent="left_col", style_class="connection", orientation="horizontal", size_hint=(1, None), height=50)
      for exchange in self.exchanges:
         self.create_exchange_block(exchange)

      # ============================================

      # ================ БЛОК 2 ====================
      self.ui.add("balances", "BoxLayout", parent="left_col", style_class="", orientation="vertical", size_hint=(1, 0.35))

      self.ui.add("grid_block_balances", "GridLayout", parent="balances", cols=3, rows=1, spacing=10, size_hint=(1, None), height=40, style_class="grid_block_balances")
      self.ui.add("balances_title", "Label", parent="grid_block_balances", text="Біржа".upper(), halign="center", valign="middle", style_class="name_table_balances")
      self.ui.add("balances_value", "Label", parent="grid_block_balances", text="Баланс".upper(), halign="center", valign="middle", style_class="name_table_balances")
      self.ui.add("balances_open_orders", "Label", parent="grid_block_balances", text="Відкриті ордери".upper(), halign="center", valign="middle", style_class="name_table_balances")

      for exchange_balance in self.exchanges:
         self.create_exchange_balance_block(exchange_balance)

      # ============================================

      # ================ БЛОК 3 ====================
      self.ui.add("block_l3", "BoxLayout", parent="left_col", style_class="block", orientation="vertical", size_hint=(1, 1))

      # 1. СТВОРЮЄМО ЗАГОЛОВОК ТАБЛИЦІ ІСТОРІЇ (поза скролом)
      self.ui.add("history_header", "GridLayout", parent="block_l3", cols=4, spacing=10, size_hint=(1, None), height=40)
      
      # Додаємо назви колонок
      self.ui.add("history_symbol", "Label", "history_header", text="Символ", style_class="name_table_balances", size_hint_x=0.25)
      self.ui.add("history_type", "Label", "history_header", text="Тип", style_class="name_table_balances", size_hint_x=0.15)
      self.ui.add("history_amount", "Label", "history_header", text="Кількість", style_class="name_table_balances", size_hint_x=0.15)
      self.ui.add("history_time", "Label", "history_header", text="Час", style_class="name_table_balances", size_hint_x=0.35)

      self.ui.add("scroll_history", "ScrollView", parent="block_l3", do_scroll_y=True, do_scroll_x=False, size_hint=(1, 1))
      self.ui.set_size("scroll_history", size_hint=(1, 1))
      self.ui.add("history_grid_content","GridLayout", parent="scroll_history", cols=1, spacing=12, padding=(12, 12, 12, 12), size_hint_y=None)
      self.ui.set_size("history_grid_content", height=0)
      Clock.schedule_once(lambda dt:self.ui.registry["history_grid_content"].bind(minimum_height=self.ui.registry["history_grid_content"].setter("height")), 0)
      
      # +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++
      # +++++++ +++++++ +++++++ +++++++  ПОТРІБНО ДЛЯ ЦЬОГО БЛОКУ ДОБАВИТИ СКРОЛ ЯК І НАЛАШТУВАННЯХ +++++++ +++++++ +++++++ +++++++ +++
      # +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ 

      # ============================================
      # =================================================================================================================

      # ================================================ ПРАВИЙ СТОВПЕЦЬ ================================================
      # ================ БЛОК 4 ====================
      self.ui.add("block_right", "BoxLayout", parent=ID_WIDGETS_BLOCKS,style_class="block", orientation="vertical", size_hint=(0.4, 1))
      self.ui.add("analysis", "Button", parent="block_right", text="Аналіз", size_hint=(1, 0.1), style_class="btn_analysis")
      
      self.ui.add("view_block", "BoxLayout", parent="block_right", orientation="vertical", size_hint=(1, 0.9))
      self.ui.add("scroll_view", "ScrollView", parent="view_block", do_scroll_y=True, do_scroll_x=False, size_hint=(1, 1))
      self.ui.set_size("scroll_view", size_hint=(1, 1))
      self.ui.add("view_grid_content","GridLayout", parent="view_block", cols=1, spacing=12, size_hint_y=None)
      self.ui.set_size("view_grid_content", height=0)
      Clock.schedule_once(lambda dt:self.ui.registry["view_grid_content"].bind(minimum_height=self.ui.registry["view_grid_content"].setter("height")), 0)
      # ============================================
      # =================================================================================================================

      return ID_WIDGETS_BLOCKS
   
   # ========================================= Блок 1 =========================================

   def check_connection(self, exchange_name):
      # Створюємо фоновий потік для перевірки підключення
      def worker():
            # Виконуємо підключення
            success = self.cc.connect(exchange_name)
            Clock.schedule_once(lambda dt: self.update_status(exchange_name, success))

      self.blocked_button(exchange_name)
      threading.Thread(target=worker).start()

   def disconnect_exchange(self, exchange_name):
      def worker():
         # Припускаємо, що у вашому ConnectionController є метод disconnect
         self.cc.disconnect(exchange_name)
         # Оновлюємо статус у головному потоці Kivy
         Clock.schedule_once(lambda dt: self.update_status(exchange_name, False))
      
      self.unblocked_button()
      threading.Thread(target=worker).start()

   def toggle_connection(self, exchange_name):
      # 1. Отримати словник усіх підключених бірж
      connected_exchanges = self.cc.get_connected_exchanges()
      # 2. Перевірити, чи є потрібна біржа у цьому словнику
      if exchange_name in connected_exchanges:
         # Якщо біржа підключена, відключити її
         self.disconnect_exchange(exchange_name)
      else:
         # Якщо біржа не підключена, підключити її
         self.check_connection(exchange_name)

   def create_exchange_block(self, exchange_name):

      # ================================ Кнопка підключки Binance ====================
      self.ui.add("grid_" + exchange_name, "GridButton", parent="connection", cols=2, rows=1, spacing=10, size_hint=(1, None), height=40, style_class="exchange_block")
      self.ui.add("btn_" + exchange_name, "Button", parent="grid_" + exchange_name, text=exchange_name.upper(), style_class="exchange_disconnection", height=40, width=60, size_hint=(0.6, 1))
      self.ui.add("label_" + exchange_name, "Label", parent="grid_" + exchange_name, text=self.TEXT_DISCONNECTED)
      # ==============================================================================

      # ================================ Налаштування дій кнопок =====================
      self.ui.set_action("grid_" + exchange_name, "on_press", lambda *args: self.toggle_connection(exchange_name))
      self.ui.set_action("btn_" + exchange_name, "on_press", lambda *args: self.toggle_connection(exchange_name))
      # ==============================================================================

   # ==========================================================================================

   # ========================================= Блок 2 =========================================

   def create_exchange_balance_block(self, exchange_balance):

      # ================ Загальний блок біржі =============================
      self.ui.add(f"grid_block_balance_{exchange_balance}", "GridLayout", parent="balances", cols=3, rows=1, spacing=10, size_hint=(1, None), height=40)

      # =============== Блок для створення іменні біржі ====================
      self.ui.add(f"label_balance_name_{exchange_balance}", "Label", parent=f"grid_block_balance_{exchange_balance}", text=f"{exchange_balance.upper()}", style_class="deactivated_value_table_balances")
      # ====================================================================

      # ============== Блок для відображення значення балансу ==============
      self.ui.add(f"label_balance_value_{exchange_balance}", "Label", parent=f"grid_block_balance_{exchange_balance}", text=f"0 USDT", style_class="deactivated_value_table_balances")
      # ====================================================================

      # ================ Блок відкритих ордерів ============================
      self.ui.add(f"label_open_orders_{exchange_balance}", "Label", parent=f"grid_block_balance_{exchange_balance}", text=f"0", style_class="deactivated_value_table_balances")
      # ====================================================================

   def get_balance_all_exchanges(self, exchange):
      def worker():
         balances = self.cc.get_balances_value_usdt(exchange)
         Clock.schedule_once(lambda dt: self.update_balances(balances, exchange))
      threading.Thread(target=worker).start()

   def get_open_orders(self, exchange):
      def worker():
         open_orders = self.cc.get_open_orders(exchange)
         Clock.schedule_once(lambda dt: self.update_open_orders(open_orders, exchange))
      threading.Thread(target=worker).start()

   def update_balances(self, balances, exchange):
      label_balance_value = self.ui.registry.get(f"label_balance_value_{exchange}")

      if label_balance_value:
         label_balance_value.text = f"{balances} USDT"

   def update_open_orders(self, open_orders, exchange):
      label_open_orders = self.ui.registry.get(f"label_open_orders_{exchange}")

      if label_open_orders:
         label_open_orders.text = f"{len(open_orders)}"

   # ==========================================================================================

   # ============================ Допоміжні методи Блоку 1 та 2 ============================================

   def update_status(self, exchange_name, success):

      # Отримуємо елементи інтерфейсу, з яких будемо оновлювати статус
      label = self.ui.registry["label_" + exchange_name]
      btn   = self.ui.registry["btn_" + exchange_name]

      label_balance_name = self.ui.registry["label_balance_name_" + exchange_name]
      label_balance_value = self.ui.registry["label_balance_value_" + exchange_name]
      label_open_orders = self.ui.registry["label_open_orders_" + exchange_name]

      list_elements = [label, btn, label_balance_name, label_balance_value, label_open_orders]

      # Перевірка наявності елементів, якщо їх немає - виходимо
      for i in list_elements:
         if not i:
            return print("Error: label or button not found!")

      if success:
         # Якщо підключення успішне, змінюємо текст та стиль кнопки
         label.text = self.TEXT_CONNECTION
         btn.style_class = "exchange_connection"

         label_balance_name.style_class = "active_value_table_balances"
         label_balance_value.style_class = "active_value_table_balances"
         label_open_orders.style_class = "active_value_table_balances"

         self.get_balance_all_exchanges(exchange_name)

         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         if exchange_name != "Binance":                        # !!!!!!!!!!!!!!!!!!!!!!!!
            self.get_open_orders(exchange_name)                # !!!!!!!!!!!!!!!!!!!!!!!!
         else:                                                 # !!!!!!!!!!!!!!!!!!!!!!!!
            label_open_orders.text = "Тимчасово не працює"     # !!!!!!!!!!!!!!!!!!!!!!!!
         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

         self.get_history_trades(exchange_name)

      else:
         # Якщо підключення не вдалося, змінюємо текст та стиль кнопки
         label.text = self.TEXT_DISCONNECTED
         btn.style_class = "exchange_disconnection"

         label_balance_name.style_class = "deactivated_value_table_balances"
         label_balance_value.style_class = "deactivated_value_table_balances"
         label_open_orders.style_class = "deactivated_value_table_balances"

         label_balance_value.text = "0" + " USDT"

      # Перемалювати стиль негайно:
      self.change_style(list_elements)

   def change_style(self, list_elements):
      for item in list_elements:
         style = getattr(item, "style_class", None)   # безпечне читання
         if style:
            self.ui.style_manager.decorate(item, style)

   def blocked_button(self, exchange_name):
      for exchange in self.exchanges:
         if not exchange == exchange_name:
            self.ui.registry["grid_"+exchange].disabled = True

   def unblocked_button(self):
      for exchange in self.exchanges:
         self.ui.registry["grid_"+exchange].disabled = False

      parent_grid = self.ui.registry.get("history_grid_content")
      parent_grid.clear_widgets()

   # ==========================================================================================

   # ========================================= Блок 3 =========================================

   def get_history_trades(self, exchange):
      """Запускає отримання історії у фоновому потоці."""
      symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT', 'DOGE/USDT', 'SOL/USDT', 'BNB/USDT']
      # Показати індикатор завантаження (опціонально)
      # self.ui.registry["history_grid_content"].clear_widgets()
      # self.ui.registry["history_grid_content"].add_widget(self.ui.dynamic.create("Label", text="Завантаження..."))
      
      def worker():
         # Припускаємо, що цей метод повертає список словників
         history = self.cc.get_trade_history_for_last_month(exchange, symbols=symbols, total_limit=10)
         # Передаємо дані в головний потік для оновлення UI
         Clock.schedule_once(lambda dt: self.update_history_trades_block(exchange, history))
         Clock.schedule_once(lambda dt: self.get_parse_json_trade(history))
      
      threading.Thread(target=worker, daemon=True).start()

   def update_history_trades_block(self, exchange, trades_history):
      """
      Очищує та динамічно заповнює блок історії торгів.
      Цей метод викликається з головного потоку Kivy.
      """
      # Знаходимо батьківський віджет для контенту
      parent_grid = self.ui.registry.get("history_grid_content")
      if not parent_grid:
         print(f"Error: 'history_grid_content' for {exchange} not found!")
         return
      
      # Очищуємо старі записи
      parent_grid.clear_widgets()

      if not trades_history:
         # Додаємо повідомлення, якщо історія порожня
         no_trades_label = self.ui.dynamic.create("Label", text=f"Немає даних про угоди для {exchange.upper()}", style_class="deactivated_value_table_balances", size_hint=(1, None), height=60)
         parent_grid.add_widget(no_trades_label)
         return

      # Динамічно створюємо і додаємо кожен рядок з даними
      for trade in trades_history:
         row = self.ui.dynamic.create("GridLayout", cols=4, spacing=10, size_hint=(1, None), height=40)
         
         symbol = trade.get('symbol', 'N/A')
         side = trade.get('side', 'N/A').upper()
         amount = f"{trade.get('amount', 0):.4f}"
         
         # Форматуємо час
         timestamp = trade.get('timestamp')
         if timestamp:
             # Біржі зазвичай надають час у мілісекундах, тому ділимо на 1000
             dt_object = datetime.fromtimestamp(timestamp / 1000)
             # Форматуємо час у зручний вигляд (Рік-Місяць-День Година:Хвилина:Секунда)
             trade_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
         else:
             trade_time = "N/A"

         row.add_widget(self.ui.dynamic.create("Label", text=symbol, style_class="BUY_history_style" if side == "BUY" else "SELL_history_style", size_hint_x=0.25))
         row.add_widget(self.ui.dynamic.create("Label", text=side, style_class="BUY_history_style" if side == "BUY" else "SELL_history_style", size_hint_x=0.15))
         row.add_widget(self.ui.dynamic.create("Label", text=amount, style_class="BUY_history_style" if side == "BUY" else "SELL_history_style", size_hint_x=0.15))
         row.add_widget(self.ui.dynamic.create("Label", text=str(trade_time), style_class="BUY_history_style" if side == "BUY" else "SELL_history_style", size_hint_x=0.35))
         
         parent_grid.add_widget(row)

   from datetime import datetime, timezone

   def get_parse_json_trade(self, trades):
      """
      trades: iterable з елементами формату ccxt trade:
         {
         'id': '...',
         'info': {  # сирий Bybit execution (result.list[] item)
            'symbol': 'ETHUSDT', 'orderId': '...', 'execId': '...', 'execTime': '1755...',
            'execPrice': '...', 'execQty': '...', 'execValue': '...', 'feeRate': '...',
            'execFee': '...', 'side': 'Sell', 'createType': 'CreateByClosing', ...
         },
         'timestamp': 1755..., 'symbol': 'ETH/USDT:USDT', 'side': 'sell',
         'price': 4739.29, 'amount': 0.03, 'cost': 142.1787, 'fee': {'currency':'USDT','cost':0.1421787,'rate':0.001}
         }
      """
      def _to_float(x):
         try:
               return float(x) if x not in (None, "") else None
         except Exception:
               return None

      def _ms_to_dt(ms):
         if ms is None:
               return None
         try:
               ms = int(ms)
               # якщо це секунди — конвертуємо як s; якщо мс — як ms
               if ms > 10_000_000_000:
                  return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
               return datetime.fromtimestamp(ms, tz=timezone.utc)
         except Exception:
               return None

      def _norm_symbol(s):
         # "ETH/USDT:USDT" -> "ETHUSDT"
         if not s:
               return None
         return s.split(":")[0].replace("/", "")

      closing_types = {"CreateByClosing", "CreateByTakeProfit", "CreateByStopLoss"}

      for item in trades or []:
         try:
               info = (item.get("info") or {})
               # 1) Зберегти fill у executions (дедуп по exec_id вже в HistoryDB.save_execution)
               self.history_db.save_execution(info, category=None)

               # 2) Зібрати дані для trades (upsert по exchange_order_id = orderId)
               order_id = info.get("orderId") or item.get("order")
               if not order_id:
                  # без order_id не робимо upsert — пропускаємо трейс, але fill вже збережений
                  continue

               # джерела даних з пріоритетом: info[...] (bybit) -> ccxt top-level
               symbol = info.get("symbol") or _norm_symbol(item.get("symbol"))
               side_raw = (info.get("side") or item.get("side") or "").strip().lower()
               side = "Buy" if side_raw == "buy" else ("Sell" if side_raw == "sell" else None)

               ts = info.get("execTime") or item.get("timestamp")
               dt = _ms_to_dt(ts)

               price = _to_float(info.get("execPrice")) or _to_float(item.get("price"))
               qty   = _to_float(info.get("execQty")) or _to_float(info.get("orderQty")) or _to_float(item.get("amount"))
               value = _to_float(info.get("execValue")) or _to_float(item.get("cost"))
               fee   = _to_float(info.get("execFee")) or _to_float((item.get("fee") or {}).get("cost"))
               exec_id = info.get("execId")

               create_type = (info.get("createType") or "").strip()

               trade_data = {
                  "exchange": "bybit",
                  "account": getattr(self, "account", "main"),
                  "symbol": symbol,
                  "side": side,
                  "exchange_order_id": order_id,
               }

               # якщо це "закриття" або тейк/стоп — заповнюємо exit_*,
               # інакше — вхід (entry_*)
               if create_type in closing_types:
                  trade_data.update({
                     "status": "closed",            # якщо у тебе бувають часткові виходи й хочеш не закривати повністю — поміняй на "open"
                     "exit_time": dt,
                     "exit_price": price,
                     "exit_qty": qty,
                     "exit_value": value,
                     "exit_fee": fee,
                     "exit_exec_ids": [exec_id] if exec_id else None,
                  })
               else:
                  trade_data.update({
                     "status": "open",
                     "entry_time": dt,
                     "entry_price": price,
                     "entry_qty": qty,
                     "entry_value": value,
                     "entry_fee": fee,
                     "entry_exec_ids": [exec_id] if exec_id else None,
                     "size": qty,
                     # за замовчуванням напрям: buy -> long, sell -> short
                     "direction": "long" if side == "Buy" else ("short" if side == "Sell" else None),
                  })

               # прибрати None-значення, щоб не засмічувати апдейт
               trade_data = {k: v for k, v in trade_data.items() if v is not None}

               # 3) upsert у trades по orderId (exchange_order_id)
               self.history_db.record_trade(
                  trade_data=trade_data,
                  settings_data=None,
                  upsert_on="exchange_order_id"
               )

         except Exception as e:
               # лог, але цикл не падає
               print(f"[parse_trade] order_id={info.get('orderId') or item.get('order')} error: {e}")

   # ==========================================================================================

   # ================================= Блок 4 =================================================
   
   