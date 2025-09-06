# gui/HistoryWindow.py

import json
import threading
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivy.graphics import Color, RoundedRectangle
from kivy.utils import get_color_from_hex

from gui.UIManager import UIManager
from utils.common.HistoryDB import HistoryDB
from utils.common.FileStructureManager import FileStructureManager

# Оновлена палітра кольорів у стилі "Нуар"
colors = {
    "background": get_color_from_hex("#111111"),      # Темний фон
    "primary": get_color_from_hex("#1E1E1E"),         # Основний фон елементів
    "secondary": get_color_from_hex("#101010"),       # Другорядний фон елементів
    "accent": get_color_from_hex("#00FF00"),          # Неоновий зелений акцент
    "text": get_color_from_hex("#E0E0E0"),            # Основний текст
    "buy": get_color_from_hex("#00FF00"),             # Колір для покупок (зелений)
    "sell": get_color_from_hex("#FF0000"),            # Колір для продажів (червоний)
    "pnl_profit": get_color_from_hex("#00FF00"),      # Колір для прибутку
    "pnl_loss": get_color_from_hex("#FF0000"),        # Колір для збитків
}


class HistoryWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_db = HistoryDB()
        self.fsm = FileStructureManager()
        self.mono_font = self.fsm.get_path("Font_RobotoMono-Regular.ttf", is_file=True) or 'RobotoMono-Regular'

        self.ui = UIManager(canvas_size=(800, 600))
        self.build_ui_skeleton() # Створюємо каркас
        self.add_widget(self.ui.build())

        main_layout = self.ui.registry["main_layout"]
        with main_layout.canvas.before:
            Color(rgb=colors["background"])
            self.rect = RoundedRectangle(size=main_layout.size, pos=main_layout.pos, radius=[0])
        main_layout.bind(size=self._update_rect, pos=self._update_rect)

    def build_ui_skeleton(self):
        """Створює статичний каркас екрана за допомогою UIManager."""
        self.ui.add("main_layout", type="BoxLayout", orientation='vertical', padding=10, spacing=10)
        self.ui.add("top_panel", type="BoxLayout", parent="main_layout", size_hint_y=None, height=40, spacing=10)
        self.ui.add("btn_back", type="Button", parent="top_panel", text="<-- Назад", style_class="button_secondary")
        self.ui.add("btn_refresh", type="Button", parent="top_panel", text="Оновити", style_class="button_primary")

        self.ui.set_action("btn_back", "on_press", lambda x: setattr(self.manager, 'current', 'job'))
        self.ui.set_action("btn_refresh", "on_press", self.load_trades)

        self.ui.add("scroll_view", type="ScrollView", parent="main_layout", size_hint=(1, 1), bar_width=10, bar_color=colors['accent'])
        self.ui.add("trades_grid", type="GridLayout", parent="scroll_view", cols=1, size_hint_y=None, spacing=8)
        
        Clock.schedule_once(lambda dt: self.ui.registry["trades_grid"].bind(minimum_height=self.ui.registry["trades_grid"].setter('height')))

    def on_enter(self, *args):
        self.load_trades()

    def load_trades(self, *args):
        trades_grid = self.ui.registry["trades_grid"]
        trades_grid.clear_widgets()
        loading_label = self.ui.dynamic.create(
            "Label",
            text="Завантаження історії...",
            height=40,
            size_hint_y=None
        )
        trades_grid.add_widget(loading_label)
        threading.Thread(target=self._load_trades_thread, daemon=True).start()


    def _load_trades_thread(self):
        try:
            trades = self.history_db.get_all_trades(limit=200)
            Clock.schedule_once(lambda dt: self._populate_ui_with_trades(trades))
        except Exception as e:
            print(f"Помилка завантаження історії угод: {e}")
            Clock.schedule_once(lambda dt: self._populate_ui_with_trades([]))

    def _populate_ui_with_trades(self, trades: list):
        trades_grid = self.ui.registry["trades_grid"]
        trades_grid.clear_widgets()

        if not trades:
            trades_grid.add_widget(self.ui.dynamic.create("Label", text="Історія угод порожня.", height=40, size_hint_y=None))
            return

        for i, trade in enumerate(trades):
            row = self.ui.dynamic.create("BoxLayout", orientation='horizontal', size_hint_y=None, height=60, padding=10, spacing=15)
            
            direction_icon = "▲" if trade['direction'] == 'buy' else "▼"
            direction_color = colors['buy'] if trade['direction'] == 'buy' else colors['sell']
            row.add_widget(self.ui.dynamic.create("Label", text=direction_icon, font_size='24sp', color=direction_color, size_hint_x=None, width=40))

            info_text = (f"[b]{trade['symbol']}[/b]  |  {trade['exchange']}\n"
                         f"[size=12sp][color=888888]{trade['entry_time']}[/color][/size]")
            info_label = self.ui.dynamic.create("Label", text=info_text, markup=True, size_hint_x=0.6, halign='left', valign='middle')
            info_label.bind(size=info_label.setter('text_size'))
            row.add_widget(info_label)
            
            pnl = trade.get('profit_loss')
            pnl_text = f"[b]${pnl:+.2f}[/b]" if pnl is not None else trade.get('status', 'N/A').capitalize()
            pnl_color = (colors['pnl_profit'] if pnl is not None and pnl >= 0 else colors['pnl_loss']) if pnl is not None else colors['text']
            row.add_widget(self.ui.dynamic.create("Label", text=pnl_text, color=pnl_color, size_hint_x=0.25, halign='right', valign='middle', font_size='16sp', markup=True))

            details_btn = self.ui.dynamic.create("Button", text="Деталі", size_hint_x=None, width=90, font_size='14sp', style_class="button_secondary")
            details_btn.bind(on_press=lambda inst, t=trade: self.show_trade_details(t))
            row.add_widget(details_btn)

            # Додаємо фон для рядка
            bg_color = colors['primary'] if i % 2 == 0 else colors['secondary']
            with row.canvas.before:
                Color(rgb=bg_color)
                rect = RoundedRectangle(pos=row.pos, size=row.size, radius=[8])
            row.bind(pos=lambda inst, val, r=rect: setattr(r, 'pos', inst.pos), 
                     size=lambda inst, val, r=rect: setattr(r, 'size', inst.size))

            trades_grid.add_widget(row)

    def show_trade_details(self, trade_data: dict):
        settings = self.history_db.get_settings_for_trade(trade_data['trade_uuid'])
        
        content = self.ui.dynamic.create("BoxLayout", orientation='vertical', padding=10, spacing=10)

        trade_info_text = "".join([f"[b]{key.replace('_', ' ').capitalize()}:[/b] {value}\n" for key, value in trade_data.items()])
        trade_label = self.ui.dynamic.create("Label", text=trade_info_text, markup=True, size_hint_y=None)
        trade_label.bind(texture_size=trade_label.setter('size'))

        settings_text = json.dumps(settings, indent=4, ensure_ascii=False) if settings else "Налаштування не знайдено."
        settings_input = self.ui.dynamic.create("TextInput", text=settings_text, readonly=True, font_name=self.mono_font, style_class="text_input_main", size_hint_y=1)
        
        content.add_widget(self.ui.dynamic.create("Label", text="Інформація про угоду", font_size='18sp', size_hint_y=None, height=30))
        content.add_widget(trade_label)
        content.add_widget(self.ui.dynamic.create("Label", text="Знімок налаштувань", font_size='18sp', size_hint_y=None, height=30))
        content.add_widget(settings_input)
        
        btn_close = self.ui.dynamic.create("Button", text="Закрити", size_hint_y=None, height=40, style_class="button_primary")
        content.add_widget(btn_close)

        popup = self.ui.dynamic.create("Popup", title="Деталі угоди", content=content, size_hint=(0.9, 0.9), separator_color=colors['accent'], background_color=colors['primary'])
        btn_close.bind(on_press=popup.dismiss)
        popup.open()

    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size