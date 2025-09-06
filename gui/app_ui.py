from kivy.app import App
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.clock import Clock

from utils.connectors.ConnectionController import ConnectionController
from utils.common.FileStructureManager import FileStructureManager

from gui.pages.HomePage import HomePage
from gui.pages.settings_page import add_settings_blocks


fms = FileStructureManager()

try:
    from gui.UIManager import UIManager
except ImportError:
    from UIManager import UIManager

PANEL_W = 250
TOPBAR_H = 50

def build_root():
    ui = UIManager(canvas_size=Window.size)
    сс = ConnectionController()
    home_page = HomePage(ui, сс, fms)

    ui.style_manager.load_styles_file(fms.get_path("gui_Style.py"))
    ui.style_manager.set_window_background("#0c1014")
    # ---------------------------------------------------------------------

    # 1) Зона контенту (буде ПІД топбаром)
    ui.add("content", "FloatLayout")
    # розмір виставимо нижче у _layout()

    # --- КОНТЕЙНЕР СТОРІНКИ -----------------------------------
    # "Домашня сторінка" (контейнер-обгортка для 4 блоків)
    ui.add("home_page", "BoxLayout", parent="content", orientation="vertical", size_hint=(1, 1))
    home_page.add_home_build("home_page")
    # -----------------------------------------------------------

    # --- СТОРІНКА НАЛАШТУВАНЬ ---------------------------------
    ui.add("settings_page", "BoxLayout", parent="content", orientation="vertical")
    add_settings_blocks(ui, page_id="settings_page")

    # -----------------------------------------------------------

    # 2) Напівпрозора димка (всередині content)
    ui.add("scrim", "Button", parent="content", style_class="scrim")
    ui.set_size("scrim", size_hint=(0, 0))

    # 3) Ліва панель (всередині content)
    ui.add("dash", "AnchorLayout", style_class="dashboard", anchor_x="left", anchor_y="top")
    ui.set_size("dash", width=PANEL_W, height=Window.height - TOPBAR_H)
    ui.place_absolute("dash", x=-300, y=0)  # старт як у анімації "close"

    ui.add("dash_stack", "BoxLayout", parent="dash", orientation="vertical", padding=(12, 12, 12, 12), spacing=8, size_hint=(1, None))
    # після побудови: висота дорівнює мінімальній (тобто сумі дітей)
    def _after_build_stack_height(*_):
        stack = ui.registry["dash_stack"]
        stack.bind(minimum_height=stack.setter("height"))
    Clock.schedule_once(_after_build_stack_height, 0)

    # --- КНОПКИ МЕНЮ ------------------------------------------
    # самі кнопки тепер кладемо в dash_stack (а не напряму в dash)
    ui.add("btn_home", "IconButton", parent="dash_stack", style_class="btn", icon_source=fms.get_path("icons_home.png"), text="Головна", icon_size=20, icon_side="left", size_hint=(1, None), height=44)
    ui.add("btn_settings", "IconButton", parent="dash_stack", style_class="btn", icon_source=fms.get_path("icons_settings.png"), text="Налаштування", icon_size=20, icon_side="left", size_hint=(1, None), height=44)
    # ui.add("btn_close", "IconButton", parent="dash_stack", style_class="btn", icon_source=fms.get_path("icons_close.png"), text="Закрити", icon_size=20, icon_side="left", size_hint=(1, None), height=44)

    ui.set_action("btn_home",     "on_press", lambda *_: _navigate("home_page", "Головне вікно"))
    ui.set_action("btn_settings", "on_press", lambda *_: _navigate("settings_page", "Налаштування"))
    # -----------------------------------------------------------

    # --- Верхня панель -----------------------------------------
    ui.add("topbar", "BoxLayout", orientation="horizontal", padding=(8, 8), spacing=8, style_class="topbar")
    ui.set_action("topbar", "on_touch_down", lambda *_: _close_dash())

    def _layout(*_):
        # Топбар зверху на всю ширину
        ui.place_absolute("topbar", 0, Window.height - TOPBAR_H, width=Window.width, height=TOPBAR_H)
        # Контент — під ним (вся решта площі)
        ui.place_absolute("content", 0, 0, width=Window.width, height=Window.height - TOPBAR_H)

    _layout()
    Window.bind(size=lambda *_: _layout())

    # Вміст топбару
    ui.add("menu_btn", "IconButton", parent="topbar", icon_source=fms.get_path("icons_menu.png"), text="Меню", icon_side="left", size_hint=(None, 1), width=100, style_class="btn_menu")
    ui.add("title", "Label", parent="topbar", text="Головне вікно", size_hint=(1, 1), halign="left", valign="middle")
    ui.add("title_time_auto_update", "Label", parent="topbar", text="Автооновлення: 5 хв", size_hint=(0.3, 1), halign="center", valign="middle")

    # -----------------------------------------------------------

    # === Анімації відкриття/закриття панелі ===
    def _open_dash(*_):
        # показати димку і відкрити меню
        ui.resize("scrim", size_hint=(1, 1))
        is_open = ui.registry["dash"].x > -1  # невеликий допуск
        if is_open:
            ui.anim_play("dash", "close")
            ui.registry["scrim"].disabled = True
            ui.resize("scrim", size_hint=(0, 0))
        else:
            ui.anim_play("dash", "open")
            ui.registry["scrim"].disabled = False

    def _close_dash(*_):
        # закрити меню і прибрати димку (щоб не блокувала скрол)
        ui.anim_play("dash", "close")
        ui.registry["scrim"].disabled = True   # ← вимкнути події
        ui.resize("scrim", size_hint=(0, 0))

    def _navigate(show_id, title_text):
        ui.show_only(["home_page", "settings_page"], show_id)
        ui.registry["title"].text = title_text
        _close_dash()


    ui.set_action("menu_btn",   "on_press", _open_dash)
    ui.set_action("btn_close",  "on_press", _close_dash)
    ui.set_action("scrim",      "on_press", _close_dash)


    # Порядок шарів: content(→ scrim → dash) -> topbar
    root = ui.build()

    ui.show_only(["home_page", "settings_page"], "home_page")
    ui.registry["scrim"].disabled = True
    return root

class MainApp(App):
    def build(self):
        return build_root()

def run():
    MainApp().run()
