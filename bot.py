# Точка входу: bot.py викликає окремий модуль з інтерфейсом (app_ui.py)

# Опціональні залежності проєкту — на випадок, якщо вони відсутні,
# не зупиняємо запуск (щоб "хоча б запускалося").
try:
    from utils.common.FileStructureManager import FileStructureManager
except Exception:
    FileStructureManager = None

try:
    from utils.common.Installer import Installer
except Exception:
    Installer = None

from gui.app_ui import run as run_app

# Kivy/GUI
# === РАНІШЕ: використовували WindowManager (тепер тимчасово вимкнено) ===
# from gui.old.WindowManager import WindowManager
# from kivy.app import App
# ...
# class MainApp(App):
#    def build(self):
#        return WindowManager()
# ========================================================================

# === ТЕПЕР: делегуємо запуск у окремий файл, з яким будемо працювати ===


if __name__ == "__main__":
    # Якщо інсталер є, можна виконати налаштування (тимчасово вимкнуто)
    # if Installer:
    #     Installer().run_setup()
    # MainApp().run()

    # Оновлюємо структуру файлів, якщо менеджер доступний
    if FileStructureManager:
        try:
            fsm = FileStructureManager()
            fsm.refresh_structure()
        except Exception as e:
            print(f"[WARN] FileStructureManager не зміг оновити структуру: {e}")

    # Стартуємо застосунок з окремого модуля
    run_app()
