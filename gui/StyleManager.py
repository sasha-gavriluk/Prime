# gui/StyleManager.py

import re
from kivy.utils import get_color_from_hex
from utils.common.SettingsLoader import SettingsLoader

import os, json, importlib.util
from pathlib import Path

# YAML не обов'язковий — якщо нема, просто пропустимо
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# gui/StyleManager.py
from kivy.core.window import Window
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, BooleanProperty, VariableListProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.core.window import Window

def _normalize_rgba(value):
    # приймає "#0B0E11", "rgb(11,14,17)", (11,14,17), (0.05,0.1,0.2,1)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        comps = list(value) + [1] * (4 - len(value))
        if any(c > 1 for c in comps[:3]):
            comps[:3] = [c / 255.0 for c in comps[:3]]
        return tuple(comps[:4])
    if isinstance(value, str):
        s = value.strip().lower()
        if s.startswith("#"):
            return tuple(get_color_from_hex(s))
        m = re.match(r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*([0-9.]+))?\s*\)", s)
        if m:
            r, g, b, a = m.groups()
            comps = [float(r), float(g), float(b)]
            if any(c > 1 for c in comps):
                comps = [c / 255.0 for c in comps]
            a = float(a) if a is not None else 1.0
            return (comps[0], comps[1], comps[2], a)
    raise ValueError(f"Unsupported color format: {value!r}")

def _parse_px(val, default=0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower()
    if s.endswith("px"): s = s[:-2]
    try: return int(float(s))
    except Exception: return default

def _parse_padding(val):
    # CSS: 1..4 значення (top right bottom left) → Kivy: [l, t, r, b]
    if val is None: return None
    if isinstance(val, (int, float)): v = int(val); return [v, v, v, v]
    if isinstance(val, (list, tuple)):
        arr = list(map(_parse_px, val))
    else:
        arr = [ _parse_px(x) for x in re.split(r"[,\s]+", str(val).strip()) if x ]
    if   len(arr) == 1: top, right, bottom, left = arr[0], arr[0], arr[0], arr[0]
    elif len(arr) == 2: top, right, bottom, left = arr[0], arr[1], arr[0], arr[1]
    elif len(arr) == 3: top, right, bottom, left = arr[0], arr[1], arr[2], arr[1]
    else:               top, right, bottom, left = arr[0], arr[1], arr[2], arr[3]
    return [left, top, right, bottom]

def _parse_border(style: dict):
    width = style.get("border-width")
    color = style.get("border-color")
    b = style.get("border")
    if isinstance(b, dict):
        width = b.get("width", width)
        color = b.get("color", color)
    elif isinstance(b, str):
        # приклад: "1px solid #fff" або "2 #ffffff" або "1px rgba(0,0,0,.2)"
        m = re.search(r"(\d+(?:\.\d+)?)\s*px", b)
        if m: width = m.group(1)
        m = re.search(r"(#[0-9a-f]{3,8}|rgba?\([^)]+\))", b, flags=re.I)
        if m: color = m.group(1)
    return _parse_px(width, 0), (_normalize_rgba(color) if color else None)

def _parse_shadow(style: dict):
    # приймаємо:
    #  - "shadow": {"color": "...", "offset": [x,y]}
    #  - "box-shadow": "0 2px rgba(0,0,0,0.35)"
    color = style.get("shadow-color")
    offset = style.get("shadow-offset")
    sd = style.get("shadow") or style.get("box-shadow")
    if isinstance(sd, dict):
        color = sd.get("color", color)
        offset = sd.get("offset", offset)
    elif isinstance(sd, str):
        # "x y color" (px опціонально)
        parts = sd.strip().split()
        if len(parts) >= 2:
            x = _parse_px(parts[0], 0); y = _parse_px(parts[1], 0)
            offset = [x, y]
        m = re.search(r"(#[0-9a-f]{3,8}|rgba?\([^)]+\))", sd, flags=re.I)
        if m: color = m.group(1)
    ox, oy = 0, 0
    if isinstance(offset, (list, tuple)) and len(offset) >= 2:
        ox, oy = _parse_px(offset[0], 0), _parse_px(offset[1], 0)
    return (_normalize_rgba(color) if color else None), (ox, oy)

class IconButton(ButtonBehavior, BoxLayout):
    font_name       = StringProperty("", allownone=True)   # дефолт для сумісності
    text_font_name  = StringProperty("", allownone=True)   # окремо для підпису
    icon_font_name  = StringProperty("", allownone=True)   # окремо для іконки

    icon_source = StringProperty("")
    icon_text   = StringProperty("")
    text        = StringProperty("")
    icon_size   = NumericProperty(22)
    spacing     = NumericProperty(8)
    padding     = VariableListProperty([12, 8, 12, 8], length=4)
    icon_side = StringProperty("left")      # 'left' або 'right'
    center_when_no_text = BooleanProperty(True)  # якщо text == "", центримо іконку

    def __init__(self, **kwargs):
        icon_text   = kwargs.pop("icon_text", "")
        icon_src    = kwargs.pop("icon_source", "")
        icon_size   = kwargs.pop("icon_size", 22)
        label_text  = kwargs.pop("text", "")              # ← забрали з kwargs
        base_font   = kwargs.pop("font_name", "")         # (або None) → "" як тобі зручно
        text_font   = kwargs.pop("text_font_name", base_font or "")
        icon_font   = kwargs.pop("icon_font_name", base_font or "")

        super().__init__(**kwargs)
        self.orientation = "horizontal"

        self._icon_widget = None
        self._label = Label(halign="left", valign="middle", shorten=True)
        self._label.bind(size=lambda inst, *_: setattr(inst, "text_size", inst.size))
        self.add_widget(self._label)

        # синхронізуємо властивість text → включиться bind(text=...)
        self.text = label_text
        # одразу оновимо сам лейбл (не обов’язково, але приємно)
        self._label.text = self.text

        # решта як було
        if text_font: self._label.font_name = text_font
        self.icon_size      = icon_size
        self.font_name      = base_font
        self.text_font_name = text_font
        self.icon_font_name = icon_font
        self.icon_source    = icon_src
        self.icon_text      = icon_text
        self._left_spacer = None
        self._right_spacer = None

        Clock.schedule_once(lambda dt: self._rebuild_icon())
        self.bind(text=lambda *_: (self._apply_text(), self._relayout()))
        self.bind(font_name=lambda *_: self._apply_font())
        self.bind(text_font_name=lambda *_: self._apply_font())
        self.bind(icon_font_name=lambda *_: self._apply_font())
        self.bind(icon_side=lambda *_: self._relayout())


    def _apply_text(self):
        if self._label:
            self._label.text = self.text

    def _apply_font(self):
        # підпис
        if self._label:
            if self.text_font_name:
                self._label.font_name = self.text_font_name
            elif self.font_name:
                self._label.font_name = self.font_name
        # іконка-лейбл (якщо іконка саме текстом)
        if self._icon_widget and isinstance(self._icon_widget, Label):
            if self.icon_font_name:
                self._icon_widget.font_name = self.icon_font_name
            elif self.font_name:
                self._icon_widget.font_name = self.font_name

    def _rebuild_icon(self, *_):
        if self._icon_widget is not None and self._icon_widget.parent is self:
            self.remove_widget(self._icon_widget)
        self._icon_widget = None

        if self.icon_source:
            self._icon_widget = Image(source=self.icon_source, size_hint=(None, 1), width=self.icon_size)
        elif self.icon_text:
            self._icon_widget = Label(
                text=self.icon_text, size_hint=(None, 1), width=self.icon_size,
                halign="center", valign="middle", font_size=self.icon_size * 0.9
            )
            if self.font_name:
                self._icon_widget.font_name = self.font_name
            self._icon_widget.bind(size=lambda inst, *_: setattr(inst, "text_size", inst.size))

        self._relayout()

    def _relayout(self, *_):
        # повністю перевпорядковуємо дітей
        self.clear_widgets()
        iw = self._icon_widget
        has_text = bool(self.text)

        # гарантуємо існування спейсерів
        if self._left_spacer is None:
            self._left_spacer = Widget(size_hint=(1, 1))
            self._right_spacer = Widget(size_hint=(1, 1))

        if has_text:
            # звичайний режим: іконка ліворуч/праворуч від тексту
            self.spacing = 8
            if self.icon_side == "right":
                self.add_widget(self._label)
                if iw: self.add_widget(iw)
            else:
                if iw: self.add_widget(iw)
                self.add_widget(self._label)
        else:
            # без тексту — центр іконки
            self.spacing = 0
            if iw:
                if self.center_when_no_text:
                    self.add_widget(self._left_spacer)
                    self.add_widget(iw)
                    self.add_widget(self._right_spacer)
                else:
                    self.add_widget(iw)

    def on_icon_source(self, *a): Clock.schedule_once(self._rebuild_icon, 0)
    def on_icon_text(self, *a):   Clock.schedule_once(self._rebuild_icon, 0)
    def on_icon_size(self, *a):
        iw = self._icon_widget
        if iw is not None:
            iw.width = self.icon_size
            if isinstance(iw, Label):
                iw.font_size = self.icon_size * 0.9

from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.gridlayout import GridLayout

class GridButton(ButtonBehavior, GridLayout):
    pass

# ====== МЕНЕДЖЕР СТИЛІВ ======
class StyleManager:
    def __init__(self, theme: str = "default", styles_path: str = None):
        self.theme = theme
        self.styles = {}  # ім'я → мапа стилю
        # Centralized hover dispatcher registry
        self._hover_widgets = set()
        self._hover_bound = False

        # кольори, якщо захочеш юзати як дефолт у своїх мапах
        self.color_hover = (20/255.0, 37/255.0, 29/255.0, 1.0)
        self.color_idle  = (0, 0, 0, 0)

        # опціонально підвантажуємо файл стилів
        if styles_path:
            self.load_styles_file(styles_path, merge=True)

    # --- API ---
    def add_style(self, name: str, style, *, merge: bool = True) -> dict:
        if isinstance(style, str):
            style = self._parse_css_string(style)
        norm = self._normalize_style_map(style)
        if merge and name in self.styles:
            merged = self.styles[name].copy()
            merged.update(norm)
            self.styles[name] = merged
        else:
            self.styles[name] = norm
        return self.styles[name].copy()

    def get_style(self, name: str) -> dict:
        return self.styles.get(name, {}).copy()

    def decorate(self, widget, style_or_name):
        style = self.get_style(style_or_name) if isinstance(style_or_name, str) else self._normalize_style_map(style_or_name or {})
        if not style:
            return  # стиль не задано — лишаємо стандартний вигляд Kivy

        # --- значення з мапи ---
        bg = style.get("background") or style.get("background-color")
        hover_bg = (style.get("hover_background") or
                    style.get("hover-background") or
                    style.get("background-hover"))

        radius = _parse_px(style.get("radius", style.get("border-radius", 0)), 0)

        # бордер (край)
        border_w, border_rgba = _parse_border(style)
        hover_border = style.get("hover-border-color")  # може приїхати зі синоніма hover-edge-color
        hover_border_rgba = _normalize_rgba(hover_border) if hover_border else None

        # padding / spacing
        pad = _parse_padding(style.get("padding"))
        if pad is not None and hasattr(widget, "padding"):
            widget.padding = pad
        if "spacing" in style and hasattr(widget, "spacing"):
            widget.spacing = _parse_px(style.get("spacing"), 0)

        # Обробка кольору тексту
        text_color = style.get("color")
        if text_color and hasattr(widget, "color"):
            # Конвертуємо значення кольору (напр., "#FFFFFF") у формат Kivy (R, G, B, A)
            widget.color = _normalize_rgba(text_color)

        # малюємо бокс (фон + рамка + тінь), якщо хоч щось задано
        draw_any = any([bg, hover_bg, border_w, border_rgba,
                        style.get("border"), style.get("border-color"),
                        style.get("shadow"), style.get("box-shadow")])

        if draw_any:
            idle_rgba = _normalize_rgba(bg) if bg else (0, 0, 0, 0)

            # якщо вказали колір краю, але не вказали товщину — зробимо 1px за замовчуванням
            if (border_rgba is not None) and (border_w == 0):
                border_w = 1

            # тінь (і все інше) як у тебе було
            shadow_rgba, (sx, sy) = _parse_shadow(style)
            self._apply_box(widget, idle_rgba, radius, border_w, border_rgba, shadow_rgba, (sx, sy))

            # hover і для фону, і для краю (якщо задано)
            if hover_bg or hover_border_rgba is not None:
                self._enable_hover(widget,
                                idle_bg=idle_rgba,
                                hover_bg=_normalize_rgba(hover_bg) if hover_bg else None,
                                idle_border=border_rgba,
                                hover_border=hover_border_rgba)
        
        # --- edge-shadow (м’яка тінь з боку) ---
        es = style.get("edge-shadow") or style.get("edge_shadow") or style.get("shadow-edge")
        if isinstance(es, dict):
            side  = es.get("side", "right")
            size  = int(es.get("size", 32))
            color = es.get("color", "rgba(0,0,0,0.35)")
            steps = int(es.get("steps", 8))
            self._apply_edge_shadow(widget, side=side, size=size, color=color, steps=steps)

            # початкова прозорість тіні: за замовчуванням 0 (щоб її не було, доки панель «закрита»)
            init_a = es.get("opacity") or es.get("alpha") or style.get("edge-shadow-opacity") or style.get("shadow-opacity")
            self.set_edge_shadow_opacity(widget, float(init_a) if init_a is not None else 0.0)

        # === font-size ===
        fs = style.get("font_size")  # уже нормалізовано вище
        if fs is not None:
            try:
                size_px = _parse_px(fs)
            except Exception:
                size_px = None
            if size_px is not None:
                # Якщо сам віджет має font_size — встановлюємо напряму (Label, Button, TextInput, тощо)
                if hasattr(widget, "font_size"):
                    try:
                        widget.font_size = size_px
                    except Exception:
                        pass
                # Спецвипадок: наш IconButton з внутрішнім self._label
                try:
                    from gui.StyleManager import IconButton  # або відносний імпорт, якщо поряд
                except Exception:
                    IconButton = None
                if IconButton and isinstance(widget, IconButton) and getattr(widget, "_label", None) is not None:
                    try:
                        widget._label.font_size = size_px
                    except Exception:
                        pass


    def set_window_background(self, color_value):
        Window.clearcolor = _normalize_rgba(color_value)

    # --- helpers ---
    def _normalize_style_map(self, style) -> dict:
        if isinstance(style, str):
            style = self._parse_css_string(style)
        style = (style or {}).copy()

        props = style.pop("props", {})
        if isinstance(props, dict):
            style.update(props)

        # background
        if "background-color" in style and "background" not in style:
            style["background"] = style.pop("background-color")

        # radius
        if "border-radius" in style and "radius" not in style:
            style["radius"] = style.pop("border-radius")

        # hover background
        if "hover-background" in style and "hover_background" not in style:
            style["hover_background"] = style["hover-background"]
        if "background-hover" in style and "hover_background" not in style:
            style["hover_background"] = style["background-hover"]

        # ✅ edge → border (синоніми)
        if "edge-color" in style and "border-color" not in style:
            style["border-color"] = style.pop("edge-color")
        if "edge_width" in style and "border-width" not in style:
            style["border-width"] = style.pop("edge_width")
        if "edge-width" in style and "border-width" not in style:
            style["border-width"] = style.pop("edge-width")

        # ✅ hover-edge-color → hover-border-color
        if "hover-edge-color" in style and "hover-border-color" not in style:
            style["hover-border-color"] = style.pop("hover-edge-color")

        if "font-size" in style and "font_size" not in style:
            style["font_size"] = style.pop("font-size")
        if "text-size" in style and "font_size" not in style:
            style["font_size"] = style.pop("text-size")

        return style

    def _apply_box(self, widget, bg_rgba, radius, border_w=0, border_rgba=None,
               shadow_rgba=None, shadow_offset=(0,0)):
        # Вимикаємо нативний фон ТІЛЬКИ коли малюємо власний
        if hasattr(widget, "background_normal"):
            try:
                widget.background_normal = ""
                widget.background_down = ""
                widget.background_color = (0, 0, 0, 0)
            except Exception:
                pass

        # ---- SHADOW (опційно, заливка під віджетом) ----
        if shadow_rgba and getattr(widget, "_shadow_rect", None) is None:
            with widget.canvas.before:
                widget._shadow_color_instr = Color(*shadow_rgba)
                widget._shadow_rect = RoundedRectangle()
        elif shadow_rgba and getattr(widget, "_shadow_color_instr", None) is not None:
            widget._shadow_color_instr.rgba = shadow_rgba

        # ---- BACKGROUND (заливка) ----
        if bg_rgba is not None:
            if getattr(widget, "_bg_rect", None) is None:
                with widget.canvas.before:
                    widget._bg_color_instr = Color(*bg_rgba)
                    widget._bg_rect = RoundedRectangle()
            else:
                widget._bg_color_instr.rgba = bg_rgba

        # ---- BORDER / EDGE (КОНТУР, НЕ ЗАЛИВКА!) ----
        # Малюємо лінію з округленням, товщина = border_w
        if border_w > 0 and border_rgba is not None:
            if getattr(widget, "_border_line", None) is None:
                with widget.canvas.before:
                    widget._border_color_instr = Color(*border_rgba)
                    widget._border_line = Line(width=float(border_w))
            else:
                widget._border_color_instr.rgba = border_rgba
                widget._border_line.width = float(border_w)

        def _sync(_w, *_):
            x, y = widget.pos
            w, h = widget.size

            # shadow
            if shadow_rgba and getattr(widget, "_shadow_rect", None) is not None:
                sx, sy = shadow_offset
                widget._shadow_rect.pos = (x + sx, y + sy)
                widget._shadow_rect.size = (w, h)
                widget._shadow_rect.radius = [(radius, radius, radius, radius)]

            # background
            if getattr(widget, "_bg_rect", None) is not None:
                widget._bg_rect.pos = (x, y)
                widget._bg_rect.size = (w, h)
                widget._bg_rect.radius = [(radius, radius, radius, radius)]

            # border (stroke)
            if getattr(widget, "_border_line", None) is not None:
                # Лінія малюється по периметру, половина ширини «всередину», половина «назовні»
                widget._border_line.rounded_rectangle = (x, y, w, h, radius)

        widget.bind(pos=_sync, size=_sync)
        _sync(widget)

    def _enable_hover(self, widget, *, idle_bg, hover_bg=None, idle_border=None, hover_border=None):
        widget._hovered = False

        def on_mouse_pos(_win, pos):
            if not widget.get_parent_window():
                return
            inside = widget.collide_point(*widget.to_widget(*pos))
            if inside != widget._hovered:
                widget._hovered = inside
                # фон
                if hover_bg is not None and getattr(widget, "_bg_color_instr", None) is not None:
                    widget._bg_color_instr.rgba = (hover_bg if inside else idle_bg)
                # край (stroke)
                if hover_border is not None and getattr(widget, "_border_color_instr", None) is not None:
                    widget._border_color_instr.rgba = (hover_border if inside else (idle_border or hover_border))

        Window.bind(mouse_pos=on_mouse_pos)
        def _cleanup(_w, parent):
            if parent is None:
                try:
                    Window.unbind(mouse_pos=on_mouse_pos)
                except Exception:
                    pass
        widget.bind(parent=_cleanup)
    def _on_global_mouse_pos(self, _win, pos):
        # Iterate over registered hover widgets and update their hover state
        to_remove = []
        for widget in list(self._hover_widgets):
            try:
                if not widget.get_parent_window():
                    continue
                inside = widget.collide_point(*widget.to_widget(*pos))
                prev = getattr(widget, "_hovered", False)
                if inside != prev:
                    widget._hovered = inside
                    cfg = getattr(widget, "_hover_cfg", None) or {}
                    # background
                    if cfg.get("hover_bg") is not None and getattr(widget, "_bg_color_instr", None) is not None:
                        widget._bg_color_instr.rgba = (cfg["hover_bg"] if inside else cfg.get("idle_bg", cfg.get("hover_bg")))
                    # border (stroke)
                    if cfg.get("hover_border") is not None and getattr(widget, "_border_color_instr", None) is not None:
                        widget._border_color_instr.rgba = (cfg["hover_border"] if inside else (cfg.get("idle_border") or cfg.get("hover_border")))
            except ReferenceError:
                # In case widget was GC'ed
                to_remove.append(widget)
            except Exception:
                # Be robust to any unusual widget states
                pass
        for w in to_remove:
            try:
                self._hover_widgets.discard(w)
            except Exception:
                pass

    def _parse_css_string(self, css: str) -> dict:
        parsed = {}
        for part in css.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                parsed[k.strip().lower()] = v.strip()
        return parsed

    def load_styles(self, mapping: dict, *, merge: bool = True) -> dict:
        """
        Приймає {"name": dict|css_string, ...} і додає стилі у реєстр.
        """
        out = {}
        for name, style in (mapping or {}).items():
            if isinstance(style, str):
                style = self._parse_css_string(style)
            norm = self._normalize_style_map(style)
            if merge and name in self.styles:
                m = self.styles[name].copy(); m.update(norm); self.styles[name] = m
            else:
                self.styles[name] = norm
            out[name] = self.styles[name].copy()
        return out

    def load_styles_file(self, path: str, *, merge: bool = True) -> dict:
        """
        Завантажує стилі з JSON/YAML/Python-файлу.
        - JSON/YAML: звичайний об'єкт {"name": {...}} або {"name": "background: ...;"}
        - Python: модуль з STYLES (dict) і/або STYLES_CSS (dict name->css_string)
        """
        p = Path(path)
        if not p.exists():
            print(f"[StyleManager] styles file not found: {path}")
            return {}

        ext = p.suffix.lower()
        data = None

        try:
            if ext == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif ext in (".yml", ".yaml"):
                if yaml is None:
                    print("[StyleManager] PyYAML not installed; skipping YAML load.")
                    return {}
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            elif ext == ".py":
                spec = importlib.util.spec_from_file_location("loaded_styles_module", str(p))
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
                data = {}
                if hasattr(mod, "STYLES") and isinstance(mod.STYLES, dict):
                    data.update(mod.STYLES)
                if hasattr(mod, "STYLES_CSS") and isinstance(mod.STYLES_CSS, dict):
                    # перетворимо css-рядки у dict і зміксуємо
                    for k, v in mod.STYLES_CSS.items():
                        css_map = self._parse_css_string(v) if isinstance(v, str) else (v or {})
                        if k in data and isinstance(data[k], dict):
                            m = data[k].copy(); m.update(css_map); data[k] = m
                        else:
                            data[k] = css_map
            else:
                print(f"[StyleManager] unsupported styles file: {path}")
                return {}
        except Exception as e:
            print(f"[StyleManager] failed to load styles from {path}: {e}")
            return {}

        if not isinstance(data, dict):
            print(f"[StyleManager] styles file must define an object/dict: {path}")
            return {}

        return self.load_styles(data, merge=merge)

    def _apply_edge_shadow(self, widget, side="right", size=32, color="rgba(0,0,0,0.35)", steps=8, radius=0):
        rgba = _normalize_rgba(color)
        if rgba is None or size <= 0 or steps <= 0:
            return

        # прибрати попередні смуги, якщо були
        old = getattr(widget, "_edge_shadow_items", None)
        if old:
            for entry in old:
                if len(entry) == 3: col, rect, _ = entry
                else:                col, rect = entry
                try:
                    widget.canvas.after.remove(col); widget.canvas.after.remove(rect)
                except Exception:
                    pass
        widget._edge_shadow_items = []

        step_w = float(size) / steps
        for i in range(steps):
            t = (i + 1) / steps
            base_a = rgba[3] * (1.0 - t) ** 2
            col = Color(rgba[0], rgba[1], rgba[2], base_a)
            rect = Rectangle()
            widget.canvas.after.add(col); widget.canvas.after.add(rect)
            widget._edge_shadow_items.append((col, rect, base_a))

        widget._edge_shadow_conf = {"side": side, "size": size, "color": color, "steps": steps, "radius": radius}

        def _sync(_w, *_):
            x, y = widget.pos; w, h = widget.size
            for i, entry in enumerate(widget._edge_shadow_items):
                _col, rect, _ba = entry
                offset = step_w * i
                if side == "right":
                    rect.pos = (x + w + offset, y); rect.size = (step_w, h)
                elif side == "left":
                    rect.pos = (x - size + offset, y); rect.size = (step_w, h)
                elif side == "top":
                    rect.pos = (x, y + h + offset); rect.size = (w, step_w)
                else:
                    rect.pos = (x, y - size + offset); rect.size = (w, step_w)

        widget.bind(pos=_sync, size=_sync)
        _sync(widget)

    def ensure_edge_shadow(self, widget, conf: dict = None):
        if getattr(widget, "_edge_shadow_items", None):
            return
        conf = conf or getattr(widget, "_edge_shadow_conf", None) or {"side": "right", "size": 32, "color": "rgba(0,0,0,0.35)", "steps": 8, "radius": 0}
        self._apply_edge_shadow(widget, **conf)

    def set_edge_shadow_opacity(self, widget, alpha: float):
        items = getattr(widget, "_edge_shadow_items", None)
        if not items:
            return
        a = max(0.0, min(1.0, float(alpha)))
        for entry in items:
            if len(entry) == 3:
                col, _rect, base_a = entry
            else:
                col, _rect = entry; base_a = col.a
            col.a = base_a * a

    def get_edge_shadow_opacity(self, widget) -> float:
        items = getattr(widget, "_edge_shadow_items", None)
        if not items: return 0.0
        entry = items[0]
        if len(entry) == 3:
            col, _rect, base_a = entry
            return 0.0 if base_a == 0 else col.a / base_a
        col, _rect = entry
        return 1.0
    
    def remove_edge_shadow(self, widget):
        items = getattr(widget, "_edge_shadow_items", None)
        if not items: return
        for entry in items:
            if len(entry) == 3: col, rect, _ = entry
            else:               col, rect = entry
            try:
                widget.canvas.after.remove(col); widget.canvas.after.remove(rect)
            except Exception:
                pass
        widget._edge_shadow_items = []
