# === УНІВЕРСАЛЬНИЙ UIManager З РОЗУМНИМ РОЗТАШУВАННЯМ ===

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.switch import Switch
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.spinner import Spinner
from kivy.animation import Animation

try:
    from gui.StyleManager import StyleManager, IconButton, GridButton
except ImportError:
    from StyleManager import StyleManager, IconButton, GridButton

class SimplePacker:
    def __init__(self, width, height):
        # спочатку весь канвас вільний
        self.free_rects = [(0, 0, width, height)]

    def place(self, w_id, w, h):
        # шукаємо перший вільний прямокутник, куди влізе w×h
        for i, (x, y, fw, fh) in enumerate(self.free_rects):
            if w <= fw and h <= fh:
                # знайшли місце — ставимо
                placed = (x, y)
                # видаляємо цей фр-рект
                self.free_rects.pop(i)
                # розщеплюємо його на дві області:
                # 1) справа від нашого віджета
                if fw - w > 0:
                    self.free_rects.append((x + w, y, fw - w, h))
                # 2) над нашим віджетом
                if fh - h > 0:
                    self.free_rects.append((x, y + h, fw, fh - h))
                return placed

        # якщо не вмістилися ніде — помилка
        raise RuntimeError(f"No space for widget {w_id}({w}×{h})")

    def reserve(self, x, y, w, h):
        """
        Вирізає прямокутник (x,y,w,h) із ВСІХ free_rects,
        розбиваючи кожен інтерсектований фрагмент на до 4 нових.
        """
        new_free = []
        for fx, fy, fw, fh in self.free_rects:
            # якщо цей фрагмент не перетинається з (x,y,w,h), лишаємо його:
            if not (x < fx + fw and x + w > fx and y < fy + fh and y + h > fy):
                new_free.append((fx, fy, fw, fh))
            else:
                # інакше — сплітимо його на до чотирьох частин:
                # 1) ліва смужка
                if fx < x:
                    new_free.append((fx, fy, x - fx, fh))
                # 2) права смужка
                if fx + fw > x + w:
                    new_free.append((x + w, fy, fx + fw - (x + w), fh))
                # 3) нижня
                if fy < y:
                    new_free.append((fx, fy, fw, y - fy))
                # 4) верхня
                if fy + fh > y + h:
                    new_free.append((fx, y + h, fw, fy + fh - (y + h)))
        self.free_rects = new_free

class UIManager:
        
    WIDGET_MAP = {
        "BoxLayout": BoxLayout,
        "GridLayout": GridLayout,
        "FloatLayout": FloatLayout,
        "AnchorLayout": AnchorLayout,
        "Label": Label,
        "Button": Button,
        "IconButton": IconButton,
        "TextInput": TextInput,
        "ScrollView": ScrollView,
        "CheckBox": CheckBox,
        "Switch": Switch,
        "Popup": Popup,
        "RecycleView": RecycleView,
        "RecycleBoxLayout": RecycleBoxLayout,
        "Spinner": Spinner,
        "GridButton": GridButton,
    }

    def __init__(self, grid_size=(100, 100), canvas_size=(800, 600)):
        self.root = FloatLayout()
        self.registry = {}

        self.style_manager = StyleManager()

        self.anim_specs = {}  # id -> нормалізована мапа анімації
        self._post_build = []
        self.structure = {}
        self.config = self.structure
        self.children_map = {}
        self.positions = {}
        self.occupied_areas = set()
        self._occupied_rects = []
        self.absolute_positions = {}
        self.actions = {}
        self.sizes = {}
        self.grid_size = grid_size
        self.canvas_width = 800
        self.canvas_height = 600
        self._flow_x = 0
        self._flow_y = 0
        self._flow_row_height = 0
        self.canvas_width, self.canvas_height = canvas_size
        self._packer = SimplePacker(self.canvas_width, self.canvas_height)
        self.dynamic = DynamicWidgetManager(self)

    def _resolve_widget(self, type_name):
        try:
            return self.WIDGET_MAP[type_name]
        except KeyError:
            raise ValueError(f"Unknown widget type '{type_name}'")
        
    def _ensure_container(self, parent_id):
        if parent_id in self.structure:
            return
        children = self.children_map.get(parent_id, [])
        # якщо хоча б один має cols -> GridLayout, інакше BoxLayout
        layout_type = "GridLayout" if any("cols" in self.structure[c]["props"] for c in children) else "BoxLayout"
        default_props = {
            "orientation": "vertical",
            "size_hint": (1, None),
            "height": 300 if layout_type=="BoxLayout" else None
        }
        self.structure[parent_id] = {
            "type": layout_type,
            "parent": None,
            "props": default_props
        }
        self.children_map.setdefault(None, []).append(parent_id)

    def _wrap_with_float(self, widget, pos_hint):
        wrapper = FloatLayout()
        widget.pos_hint = pos_hint
        wrapper.add_widget(widget)
        return wrapper

    def add(self, id, type, parent=None, bounds=None, style_class=None, **props):
        if id in self.registry or id in self.structure:
            raise ValueError(f"Widget '{id}' already exists")
        
        self.structure[id] = {"type": type, "parent": parent, "props": props, "style_class": style_class}
        self.children_map.setdefault(parent, []).append(id)
        if parent and parent not in self.structure:
            self._ensure_container(parent)

        if bounds:
            x1,y1,x2,y2 = bounds
            # відкладаємо в абсолютні позиції вже зараз
            self.set_bounds(id, x1,y1,x2,y2)

    def set_size(self, id, width=None, height=None, size_hint=None):
        self.sizes[id] = {"width": width, "height": height, "size_hint": size_hint}

    def set_action(self, id, event, handler):
        self.actions.setdefault(id, []).append((event, handler))

    def open_popup(self, title, content, **props):
        """Convenience helper to create and open a Popup via DynamicWidgetManager."""
        popup = self.dynamic.create("Popup", title=title, content=content, **props)
        popup.open()
        return popup

    def place_absolute(self, id_, x, y, width=None, height=None, reserve_area=True):
        """
        Тепер це тільки малює віджет за координатами, але вже не резервує,
        бо всі explicit-зони ми забронювали в build().
        """
        widget = self.registry.get(id_)
        w = width  or self.sizes.get(id_, {}).get("width", self.grid_size[0])
        h = height or self.sizes.get(id_, {}).get("height", self.grid_size[1])

        # перевірка на конфлікт за старими записами
        try:
            self._check_conflicts(id_, x, y, w, h)
            placed_x, placed_y = x, y
        except ValueError:
            # якщо конфлікт — переходимо на авто-розміщення
            placed_x, placed_y = self._packer.place(id_, w, h)
            print(f"[UIManager] Конфлікт у ({x},{y}) — пересунув '{id_}' у ({placed_x},{placed_y})")

        # фіксуємо
        self._occupied_rects.append((id_, placed_x, placed_y, w, h))

        # і нарешті малюємо/запам'ятовуємо для build()
        if widget:
            widget.size_hint = (None, None)
            widget.width, widget.height = w, h
            widget.pos_hint = {}
            widget.pos = (placed_x, placed_y)
        else:
            self.absolute_positions[id_] = {
                "x": placed_x, "y": placed_y, "width": w, "height": h
            }

    def auto_place(self, id_, width=None, height=None):
        # обчислюємо реальні w,h
        size = self.sizes.get(id_, {})
        w = width  or size.get("width",  self.grid_size[0])
        h = height or size.get("height", self.grid_size[1])

        # отримуємо першу вільну позицію (і packer.place уже split’ить free_rects)
        x, y = self._packer.place(id_, w, h)

        # ставимо віджет, але reserve_area=False — щоби не дублювати вирізку
        self.place_absolute(id_, x, y, width=w, height=h, reserve_area=False)

    def _conflicts_with_existing(self, x, y, w, h):
        for _, ox, oy, ow, oh in self._occupied_rects:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy):
                return True
        return False

    def _check_conflicts(self, id_, x, y, w, h):
        for other_id, ox, oy, ow, oh in self._occupied_rects:
            if id_ == other_id:
                continue
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy):
                raise ValueError(f"Conflict: '{id_}' overlaps '{other_id}'")

    def _filter_props(self, widget_cls, props):
        valid = widget_cls().properties().keys()
        return {k: v for k, v in props.items() if k in valid and v is not None}

    def _build_widget(self, id_):
        conf = self.structure[id_]
        cls = self._resolve_widget(conf["type"])

        # 1) Стиль: може бути ім’я (str) або мапа (dict)
        style_class = conf.get("style_class")
        style_props = (self.style_manager.get_style(style_class) if isinstance(style_class, str)
                    else (style_class or {}))

        # 2) Мердж стилю з props (props мають пріоритет)
        final_props = {**style_props, **conf["props"]}

        # 3) <--- ЗМІНЕНО: Явно вилучаємо конфігурацію анімації перед створенням віджета
        # Це гарантує, що блок "anim" не буде переданий у конструктор віджета як властивість.
        anim_spec = (final_props.pop("anim", None) or 
                     final_props.pop("animation", None) or 
                     final_props.pop("transitions", None))

        # 4) Фільтруємо властивості, які розуміє Kivy-віджет
        filtered = self._filter_props(cls, final_props)

        # 5) Створення віджета
        if conf["type"] == "Popup":
            filtered.setdefault("auto_dismiss", False)
            widget = cls(**filtered)
            self.registry[id_] = widget
        else:
            widget = cls(**filtered)
            self.registry[id_] = widget
            # 6) Після створення — декорація (canvas, hover, edge тощо)
            if style_class:
                # Для декорації використовуємо вихідну мапу стилів, 
                # оскільки StyleManager сам знає, які ключі йому потрібні (включно з 'edge-shadow' і т.д.).
                # Використовуємо `style_props`, бо вони не містять `props` віджета.
                self.style_manager.decorate(widget, style_class if isinstance(style_class, str) else style_props)

        # 7) <--- ЗМІНЕНО: Реєстрація анімації відбувається з вилученої конфігурації
        if anim_spec:
            self.anim_config(id_, anim_spec)

        # 8) Прив’язка подій
        for event, handler in self.actions.get(id_, []):
            if hasattr(widget, event):
                widget.bind(**{event: handler})

        # 9) Розміри
        size = self.sizes.get(id_, {})
        if size.get("size_hint") is not None:
            widget.size_hint = size["size_hint"]
        if size.get("width") is not None:
            widget.size_hint_x = None
            widget.width = size["width"]
        if size.get("height") is not None:
            widget.size_hint_y = None
            widget.height = size["height"]

        # 10) Абсолютна позиція
        if id_ in self.absolute_positions:
            pos = self.absolute_positions[id_]
            x, y = pos["x"], pos["y"]
            w = pos.get("width") or widget.width
            h = pos.get("height") or widget.height

            already_placed = any(item[0] == id_ for item in self._occupied_rects)
            if not already_placed:
                self._check_conflicts(id_, x, y, w, h)
                self._occupied_rects.append((id_, x, y, w, h))

            widget.pos_hint = {}
            widget.pos = (x, y)
            if pos.get("width") is not None:
                widget.size_hint_x = None
                widget.width = w
            if pos.get("height") is not None:
                widget.size_hint_y = None
                widget.height = h

        for cb in getattr(self, "_post_build", []):
            try:
                cb()
            except Exception as e:
                print("UIManager post_build callback error:", e)
        self._post_build.clear()
        return widget

    def _attach(self, parent_widget, child_id):
        # Цей метод залишається з останнім виправленням
        widget = self._build_widget(child_id)
        conf = self.structure[child_id]
        parent_id = conf.get("parent")

        if parent_id is None:
            if child_id not in self.absolute_positions:
                pass 
            parent_widget.add_widget(widget)
        else:
            parent_widget.add_widget(widget)

        for sub in self.children_map.get(child_id, []):
            self._attach(widget, sub)

    def build(self):
        # 1) Спочатку пробіжимося по всіх жорстких bounds
        #    і зарезервуємо їх в packer:
        for wid, pos in self.absolute_positions.items():
            x, y = pos["x"], pos["y"]
            w = pos["width"]
            h = pos["height"]
            # резервуємо цю ділянку одразу
            self._packer.reserve(x, y, w, h)
            # і фіксуємо в occupied_rects, щоби _check_conflicts їх теж бачив
            self._occupied_rects.append((wid, x, y, w, h))

        # 2) Після цього — звичайне «attach» (авто-розміщення всіх, хто без bounds)
        for top_id in self.children_map.get(None, []):
            self._attach(self.root, top_id)

        return self.root

    def move(self, id_, align="center", corner=None):
        widget = self.registry.get(id_)
        if not widget:
            raise ValueError(f"Widget '{id_}' not found")
        pos = {"left": {"x": 0}, "center": {"center_x": 0.5}, "right": {"right": 1}}
        corner_pos = {"top": {"top": 1}, "bottom": {"y": 0}}
        hint = {}
        hint.update(pos.get(align, {}))
        if corner:
            hint.update(corner_pos.get(corner, {}))
        widget.pos_hint = hint

    def resize(self, id_, width=None, height=None, size_hint=None):
        widget = self.registry.get(id_)
        if not widget:
            raise ValueError(f"Widget '{id_}' not found")
        if size_hint is not None:
            widget.size_hint = size_hint
        if width is not None:
            widget.size_hint_x = None
            widget.width = width
        if height is not None:
            widget.size_hint_y = None
            widget.height = height

    def set_bounds(self, id_, x1, y1, x2, y2, from_top=True):
        """
        Задає положення через дві точки:
          (x1,y1) — лівий-верхній,
          (x2,y2) — правий-нижній.
        Якщо from_top=True, то y-координати передаються від верхнього краю,
        і ми їх конвертуємо в систему Kivy (від низу).
        """
        # Конвертуємо y-координати в Kivy (якщо від верхнього краю)
        if from_top:
            y1b = self.canvas_height - y1
            y2b = self.canvas_height - y2
        else:
            y1b, y2b = y1, y2

        # Правий-ліве / верх-низ
        x_left  = min(x1, x2)
        x_right = max(x1, x2)
        y_bot   = min(y1b, y2b)
        y_top   = max(y1b, y2b)

        w = x_right - x_left
        h = y_top   - y_bot

        self.place_absolute(id_, x_left, y_bot, width=w, height=h)

    def get_pixel_map(self):
        """
        Повертає текст, де кожен піксель полотна:
          – '0'       якщо він перетинається з будь-яким віджетом,
          – 'X,Y'     якщо вільний (X,Y — 1-based координати пікселя).
        Всього рядків = canvas_height, елементів на рядок = canvas_width.
        """
        w, h = self.canvas_width, self.canvas_height
        lines = []
        # проходимо по y від верху (1) до низу (h)
        for y in range(1, h+1):
            row = []
            py = h - y  # конвертація в систему Kivy (0-based від нижнього краю)
            for x in range(1, w+1):
                px = x - 1
                occupied = any(
                    (px < rx+rw and px >= rx and
                     py < ry+rh and py >= ry)
                    for _, rx, ry, rw, rh in self._occupied_rects
                )
                row.append("0" if occupied else f"{x},{y}")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def write_pixel_map(self, filename="pixel_map.txt", encoding="utf-8"):
        """
        Записує результат get_pixel_map() у файл filename.
        """
        txt = self.get_pixel_map()
        with open(filename, "w", encoding=encoding) as f:
            f.write(txt)
        print(f"Піксельна мапа записана у {filename}")

    def get_scaled_map(self, scale=50, occupied_char="0", free_char="."):
        """
        Повертає стиснутий текстовий огляд полотна:
          - 'scale' — розмір блоку в пікселях (scale×scale)
          - 'occupied_char' — символ для блоків із віджетами
          - 'free_char'     — символ для порожніх блоків
        Результат має розміри:
          cols = ceil(canvas_width/scale)
          rows = ceil(canvas_height/scale)
        """
        from math import ceil

        w, h = self.canvas_width, self.canvas_height
        cols = ceil(w / scale)
        rows = ceil(h / scale)

        lines = []
        # y-блоки: зверху вниз
        for row in range(rows):
            y0 = h - (row + 1) * scale
            y1 = h - row * scale
            row_chars = []
            for col in range(cols):
                x0 = col * scale
                x1 = min(w, (col + 1) * scale)
                # чи є у цьому блоці хоч один зайнятий піксель?
                occupied = any(
                    (rx < x1 and rx + rw > x0 and
                     ry < y1 and ry + rh > y0)
                    for _, rx, ry, rw, rh in self._occupied_rects
                )
                row_chars.append(occupied and occupied_char or free_char)
            lines.append("".join(row_chars))
        return "\n".join(lines)

    def write_scaled_map(self, filename="scaled_map.txt", scale=50, **kwargs):
        """
        Записує стиснутий огляд у файл.
        Параметри:
          - filename: ім’я файлу
          - scale:    розмір блоків (в пікселях)
          - **kwargs: передасться в get_scaled_map (наприклад, occupied_char)
        """
        txt = self.get_scaled_map(scale=scale, **kwargs)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"Стиснута мапа ({self.canvas_width//scale}×{self.canvas_height//scale}) записана у «{filename}»")

    # === А Н І М А Ц І Ї ===================================================

    def anim_config(self, id_, spec: dict):
        if not isinstance(spec, dict):
            raise ValueError("anim_config expects dict with 'open'/'close'")
        s = {
            "duration": float(spec.get("duration", 0.25)),
            "transition": spec.get("transition", "linear"),
            "open": dict(spec.get("open", {})),
            "close": dict(spec.get("close", {})),
        }
        self.anim_specs[id_] = s
        return s

    def _anim_resolve_target(self, default_widget, key: str):
        if key.startswith("#"):
            if "." not in key:
                raise ValueError(f"Animation key '{key}' must be like '#id.prop'")
            tgt_id, prop = key[1:].split(".", 1)
            tgt = self.registry.get(tgt_id)
            if tgt is None:
                raise ValueError(f"Animation target id '{tgt_id}' not found")
            return tgt, prop
        return default_widget, key

    def _is_shadow_opacity_prop(self, prop: str) -> bool:
        p = prop.replace("-", "_").lower()
        return p in ("shadow_opacity", "edge_shadow_opacity")

    def anim_play(self, id_, phase: str = "open"):
        """
        Програє фазу 'open' або 'close' для віджета id_ за раніше
        зареєстрованою специфікацією (див. self.anim_config).

        Підтримує:
        - Числові властивості (анімуються стандартно через Kivy Animation)
        - Нечислові (встановлюються: для 'open' одразу, для 'close' після завершення)
        - Спец-властивість 'shadow_opacity'/'edge_shadow_opacity' (0..1)
        - ✅ НОВЕ: Анімація кольорів для стилів, що малюються через StyleManager
            (background-color, border-color, shadow-color).
        """
        spec = self.anim_specs.get(id_)
        if not spec:
            raise ValueError(f"No animation configured for '{id_}'")
        if phase not in ("open", "close"):
            raise ValueError("phase must be 'open' or 'close'")

        widget = self.registry.get(id_)
        if widget is None:
            raise ValueError(f"Widget '{id_}' not found")

        # <--- НОВЕ: Мапа для зв'язку назв стилів з внутрішніми об'єктами Kivy
        STYLE_ANIM_MAP = {
            "background": "_bg_color_instr",
            "background-color": "_bg_color_instr",
            "border-color": "_border_color_instr",
            "edge-color": "_border_color_instr",
            "shadow-color": "_shadow_color_instr"
        }

        to_map = dict(spec[phase])
        dur = float(to_map.pop("duration", spec.get("duration", 0.25)))
        trn = to_map.pop("transition", spec.get("transition", "linear"))

        per_target_numeric = {}
        immediate_sets = []
        deferred_sets = []
        deferred_calls = []
        special_tweens = []
        
        # <--- НОВЕ: Окремий список для анімацій кольору
        rgba_animations = []

        # <--- НОВЕ: Спочатку обробимо анімації кольорів
        for style_prop, instr_name in STYLE_ANIM_MAP.items():
            if style_prop in to_map:
                color_val = to_map.pop(style_prop) # Видаляємо, щоб не обробляти вдруге
                # Перевіряємо, чи віджет має відповідний графічний елемент
                if hasattr(widget, instr_name):
                    instr = getattr(widget, instr_name)
                    if instr:
                        # Нормалізуємо колір у формат RGBA, який розуміє Kivy
                        target_rgba = self.style_manager._normalize_rgba(color_val)
                        anim = Animation(rgba=target_rgba, duration=dur, t=trn)
                        rgba_animations.append((anim, instr))


        # Цей цикл тепер обробляє решту властивостей
        for k, v in to_map.items():
            tgt, prop = self._anim_resolve_target(widget, k)

            if isinstance(v, (int, float)):
                pnorm = prop.replace("-", "_").lower()
                if pnorm in ("shadow_opacity", "edge_shadow_opacity"):
                    self.style_manager.ensure_edge_shadow(tgt)
                    start = self.style_manager.get_edge_shadow_opacity(tgt)

                    class _Tween:
                        __slots__ = ("uid", "_v", "_setter")
                        def __init__(self, setter, init):
                            self.uid = id(self)
                            self._v = float(init)
                            self._setter = setter
                        @property
                        def v(self): return self._v
                        @v.setter
                        def v(self, x):
                            self._v = float(x)
                            self._setter(self._v)

                    tw = _Tween(lambda a: self.style_manager.set_edge_shadow_opacity(tgt, a), start)
                    try:
                        Animation.stop_all(tw)
                    except Exception:
                        pass
                    anim = Animation(v=float(v), duration=dur, t=trn)
                    special_tweens.append((anim, tw))

                    if phase == "close" and float(v) <= 0.0:
                        deferred_calls.append(lambda t=tgt: self.style_manager.remove_edge_shadow(t))
                else:
                    per_target_numeric.setdefault(tgt, {})[prop] = v
            else:
                (immediate_sets if phase == "open" else deferred_sets).append((tgt, prop, v))

        for tgt, prop, val in immediate_sets:
            try:
                setattr(tgt, prop, val)
            except Exception:
                pass

        fired = {"done": False}
        def fire_deferred(*_):
            if fired["done"]:
                return
            fired["done"] = True
            for t2, p2, v2 in deferred_sets:
                try:
                    setattr(t2, p2, v2)
                except Exception:
                    pass
            for fn in deferred_calls:
                try:
                    fn()
                except Exception:
                    pass

        started = False
        for tgt, props in per_target_numeric.items():
            started = True
            try:
                Animation.stop_all(tgt)
            except Exception:
                pass
            anim = Animation(duration=dur, t=trn, **props)
            anim.bind(on_complete=lambda _a, _t: fire_deferred())
            anim.start(tgt)

        for anim, tw in special_tweens:
            started = True
            anim.bind(on_complete=lambda _a, _t: fire_deferred())
            anim.start(tw)
            
        # <--- НОВЕ: Запускаємо анімації кольору
        for anim, instr in rgba_animations:
            started = True
            try: Animation.stop_all(instr)
            except Exception: pass
            anim.bind(on_complete=lambda _a, _t: fire_deferred())
            anim.start(instr)

        if not started:
            fire_deferred()

    def anim_toggle(self, id_, is_open=None):
        widget = self.registry.get(id_)
        if widget is None:
            raise ValueError(f"Widget '{id_}' not found")
        opened = (is_open(widget) if callable(is_open) else (getattr(widget, "x", 0) >= 0))
        self.anim_play(id_, "close" if opened else "open")

    def show_only(self, ids, show_id):
        """Показати лише show_id, інші повністю сховати (ще й розміром)."""
        for _id in ids:
            w = self.registry.get(_id)
            if not w:
                continue
            is_show = (_id == show_id)
            # робимо сторінку «нульового розміру», щоби її не було видно й не впливала на лейаут
            try:
                w.size_hint = (1, 1) if is_show else (0, 0)
            except Exception:
                pass
            # додатково — прозорість/інтеракція
            try: w.opacity  = 1 if is_show else 0
            except: pass
            try: w.disabled = not is_show
            except: pass

class DynamicWidgetManager:
    """
    Допоміжний клас для миттєвого створення віджетів з підтримкою стилів.
    Призначений для динамічного контенту (списків, попапів).
    """
    def __init__(self, main_ui_manager: UIManager):
        """
        Ініціалізується, отримуючи доступ до ресурсів головного UIManager.
        """
        self.main_ui = main_ui_manager
        self.style_manager = main_ui_manager.style_manager
        self.WIDGET_MAP = main_ui_manager.WIDGET_MAP

    def create(self, widget_type: str, style_class: str = None, **props) -> Widget:
        try:
            WidgetClass = self.WIDGET_MAP[widget_type]
        except KeyError:
            raise ValueError(f"Невідомий тип віджета '{widget_type}'")

        # 1) Базові стилі як мапа (щоб мати і padding/spacing, і канвас-ключі)
        style_props = self.style_manager.get_style(style_class) if style_class else {}

        # 2) props мають пріоритет над стилем
        final_props   = {**style_props, **props}
        filtered_props = self.main_ui._filter_props(WidgetClass, final_props)

        # 3) Створюємо віджет
        widget = WidgetClass(**filtered_props)

        # 4) ✅ Накладаємо канвас-стиль (фон, рамка, hover, edge-shadow тощо)
        if style_class:
            self.style_manager.decorate(widget, style_class if isinstance(style_class, str) else style_props)

        return widget

