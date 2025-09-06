# pages/settings_blocks.py

from kivy.clock import Clock

"""
Додає контент сторінки «Налаштування» всередину вже існуючого контейнера.
НІЧОГО не створює поза межами переданого parent (page_id).
"""

def add_settings_blocks(ui, page_id="settings_page"):
    """
    ui       — ваш UIManager
    page_id  — id контейнера сторінки (наприклад, "settings_page"),
               який уже створено у вашому app_ui.py
    Повертає id кореневого контейнера ("settings_body").
    """

    ID_WIDGETS_SETTINGS = "settings_body"
    # Кореневий контейнер налаштувань
    ui.add(ID_WIDGETS_SETTINGS, "BoxLayout", parent=page_id,
           orientation="vertical", padding=(12, 12, 12, 12), spacing=10, size_hint=(1, 1))

    ui.add("settings_scroll", "ScrollView", parent=ID_WIDGETS_SETTINGS, do_scroll_y=True, do_scroll_x=False)
    ui.set_size("settings_scroll", size_hint=(1, 1))

    ui.add("settings_list", "GridLayout", parent="settings_scroll", cols=1, spacing=12, padding=(12, 12, 12, 12), size_hint_y=None)
    ui.set_size("settings_list", height=0)

    # зв'язка для скролу: контентна висота -> реальна висота
    Clock.schedule_once(lambda dt:
        ui.registry["settings_list"].bind(minimum_height=ui.registry["settings_list"].setter("height")), 0)

    for i in range(1, 12 + 1):
        slot_id = f"settings_slot_{i}"
        ui.add(slot_id, "BoxLayout", parent="settings_list", orientation="vertical", size_hint=(1, None), height=120)
        ui.add(f"{slot_id}_block", "BoxLayout", parent=slot_id, style_class="block", orientation="vertical", size_hint=(1, 1), spacing=8)
        ui.add(f"{slot_id}_title", "Label", parent=f"{slot_id}_block", text=f"Налаштування {i}", halign="left", valign="middle")

    return ID_WIDGETS_SETTINGS
