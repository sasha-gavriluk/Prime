# gui/StartedWindow.py

from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from gui.UIManager import UIManager
import threading

class StartedWindow(Screen):
    def __init__(self, connection_controller, **kwargs):
        super().__init__(**kwargs)
        self.connection_controller = connection_controller
        self.ui = UIManager(canvas_size=(800, 600))
        self.exchanges = ["Binance", "Bybit"]

        self.build_ui()
        self.add_widget(self.ui.build())

    def build_ui(self):
        # Вітальний напис
        self.ui.add("greeting_label", type="Label", text="Ласкаво просимо в мій бот!", style_class="label_header", font_size='22sp')
        self.ui.set_bounds("greeting_label", 0, 10, 800, 60)

        for idx, exchange in enumerate(self.exchanges):
            row_y = 100 + idx * 50

            self.ui.add(f"label_{exchange}", type="Label", text=exchange)
            self.ui.add(f"btn_{exchange}", type="Button", text="Підключити", style_class="button_primary")
            self.ui.add(f"btn_disc_{exchange}", type="Button", text="Відключити", style_class="button_secondary")
            self.ui.add(f"status_{exchange}", type="Label", text="")

            self.ui.set_bounds(f"label_{exchange}", 20, row_y, 140, row_y + 40)
            self.ui.set_bounds(f"btn_{exchange}", 150, row_y, 300, row_y + 40)
            self.ui.set_bounds(f"btn_disc_{exchange}", 310, row_y, 460, row_y + 40)
            self.ui.set_bounds(f"status_{exchange}", 470, row_y, 700, row_y + 40)

            self.ui.set_action(f"btn_{exchange}", "on_press", lambda inst, ex=exchange: self.check_connection(ex))
            self.ui.set_action(f"btn_disc_{exchange}", "on_press", lambda inst, ex=exchange: self.disconnect_exchange(ex))

        # Кнопка "Продовжити"
        self.ui.add("btn_continue", type="Button", text="Продовжити -->", style_class="button_primary")
        self.ui.set_bounds("btn_continue", 630, 540, 780, 590)
        self.ui.set_action("btn_continue", "on_press", self.continue_clicked)

        # Кнопка "Налаштування"
        self.ui.add("btn_settings", type="Button", text="Налаштування", style_class="button_secondary")
        self.ui.set_bounds("btn_settings", 10, 540, 160, 590)
        self.ui.set_action("btn_settings", "on_press", self.open_settings)

    def open_settings(self, instance):
        self.manager.current = 'settings_simple'

    def on_enter(self, *args):
        any_connected = False

        for exchange in self.exchanges:
            is_connected = self.connection_controller.get_api(exchange).is_connected()
            any_connected |= is_connected
            self.update_status(exchange, is_connected)

        self.ui.registry["btn_continue"].disabled = not any_connected

    def check_connection(self, exchange_name):
        def worker():
            success = self.connection_controller.connect(exchange_name)
            Clock.schedule_once(lambda dt: self.update_status(exchange_name, success))
            Clock.schedule_once(lambda dt: self.update_continue_state())
        threading.Thread(target=worker).start()

    def disconnect_exchange(self, exchange_name):
        self.connection_controller.disconnect(exchange_name)
        self.update_status(exchange_name, False)
        self.update_continue_state()

    def update_status(self, exchange_name, is_connected):
        status_label = self.ui.registry[f"status_{exchange_name}"]
        status_label.text = "Підключено" if is_connected else "Не підключено"
        status_label.color = (0.1, 0.8, 0.1, 1) if is_connected else (1, 1, 1, 1)

        self.ui.registry[f"btn_disc_{exchange_name}"].disabled = not is_connected
        self.ui.registry[f"btn_{exchange_name}"].disabled = is_connected

    def update_continue_state(self):
        any_connected = any(
            api.is_connected() for api in self.connection_controller.connections.values()
        )
        self.ui.registry["btn_continue"].disabled = not any_connected

    def continue_clicked(self, instance):
        self.manager.current = 'job'