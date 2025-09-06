# pages/home_blocks.py
import threading

from kivy.clock import Clock

class HomePage:
   TEXT_CONNECTION = "Підключено"
   TEXT_DISCONNECTED = "Відключено"

   def __init__(self, ui, connection_controller, fms):
      self.ui = ui
      self.fms = fms

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

      # +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++
      # +++++++ +++++++ +++++++ +++++++  ПОТРІБНО ДЛЯ ЦЬОГО БЛОКУ ДОБАВИТИ СКРОЛ ЯК І НАЛАШТУВАННЯХ +++++++ +++++++ +++++++ +++++++ +++
      # +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ +++++++ 

      for exchange_balance in self.exchanges:
         self.create_exchange_balance_block(exchange_balance)

      # ============================================

      # ================ БЛОК 3 ====================
      self.ui.add("block_l3", "BoxLayout", parent="left_col", style_class="block", orientation="vertical", size_hint=(1, 1))

      scroll_history = self.ui.add("scroll_history", "ScrollView", parent="block_l3", do_scroll_y=True, do_scroll_x=False, size_hint=(1, 1))
      self.ui.set_size("scroll_history", size_hint=(1, 1))
      
      # GridLayout для контенту історії, який будемо заповнювати динамічно
      grid_history = self.ui.add("history_grid_content","GridLayout", parent="scroll_history", cols=1, spacing=10, size_hint_y=None)
      self.ui.set_size("settings_list", height=0)

      Clock.schedule_once(lambda dt:
         self.ui.registry["history_grid_content"].bind(minimum_height=self.ui.registry["history_grid_content"].setter("height")), 0)

      # ============================================
      # =================================================================================================================

      # ================================================ ПРАВИЙ СТОВПЕЦЬ ================================================
      # ================ БЛОК 4 ====================
      self.ui.add("block_right", "BoxLayout", parent=ID_WIDGETS_BLOCKS,style_class="block", orientation="vertical", size_hint=(0.4, 1))
      self.ui.add("block_right_title", "Label", parent="block_right",text="Правий блок", halign="center", valign="middle")
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

      threading.Thread(target=worker).start()

   def disconnect_exchange(self, exchange_name):
      def worker():
         # Припускаємо, що у вашому ConnectionController є метод disconnect
         self.cc.disconnect(exchange_name)
         # Оновлюємо статус у головному потоці Kivy
         Clock.schedule_once(lambda dt: self.update_status(exchange_name, False))
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
      self.ui.add(f"label_open_orders_{exchange_balance}", "Label", parent=f"grid_block_balance_{exchange_balance}", text=f"5", style_class="deactivated_value_table_balances")
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
         row = self.ui.dynamic.create("GridLayout", cols=4, spacing=10, size_hint_y=None, height=40)
         
         symbol = trade.get('symbol', 'N/A')
         side = trade.get('side', 'N/A').upper()
         amount = f"{trade.get('amount', 0):.4f}"
         
         # Форматуємо час
         timestamp = trade.get('timestamp')
         if timestamp:
             trade_time = Clock.get_boottime() # Використовуємо Kivy Clock для часу
         else:
             trade_time = "N/A"

         row.add_widget(self.ui.dynamic.create("Label", text=symbol, style_class="active_value_table_balances"))
         row.add_widget(self.ui.dynamic.create("Label", text=side, style_class="active_value_table_balances", color=(0.2, 0.8, 0.2, 1) if side == 'BUY' else (0.8, 0.2, 0.2, 1)))
         row.add_widget(self.ui.dynamic.create("Label", text=amount, style_class="active_value_table_balances"))
         row.add_widget(self.ui.dynamic.create("Label", text=str(trade_time), style_class="deactivated_value_table_balances"))
         
         parent_grid.add_widget(row)


   # ==========================================================================================