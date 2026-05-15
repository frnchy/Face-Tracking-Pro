from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import customtkinter as ctk
import tkinter as tk

from . import constants as C


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _font(size: int = 11, *, bold: bool = False, mono: bool = False) -> ctk.CTkFont:
    family = C.FONT_MONO if mono else C.FONT_SANS
    return ctk.CTkFont(family=family, size=size,
                       weight="bold" if bold else "normal")


class Tooltip:
    def __init__(self, widget, text: str, *, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self._tipwindow: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

    def _schedule(self, _e=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        try:
            tw.attributes("-alpha", 0.97)
        except tk.TclError:
            pass
        frame = tk.Frame(tw, bg=C.UI_SURFACE_4,
                         highlightbackground=C.UI_BORDER_HI, highlightthickness=1)
        frame.pack()
        tk.Label(
            frame, text=self.text, bg=C.UI_SURFACE_4, fg=C.UI_TEXT,
            font=(C.FONT_SANS, 9), padx=10, pady=5, justify="left",
        ).pack()
        tw.update_idletasks()
        tw.wm_geometry(f"+{x - tw.winfo_width() // 2}+{y}")
        self._tipwindow = tw

    def _hide(self, _e=None):
        self._cancel()
        if self._tipwindow is not None:
            try:
                self._tipwindow.destroy()
            except Exception:
                pass
            self._tipwindow = None


class BrandLogo(ctk.CTkFrame):
    def __init__(self, master, size: int = 28, image_path=None,
                 bg_color: str = C.UI_BG, tint=(229, 226, 218)) -> None:
        super().__init__(master, fg_color="transparent", width=size, height=size)
        self.size = size
        self._photo = None
        if image_path is not None:
            try:
                from PIL import Image, ImageTk
                im = Image.open(image_path).convert("RGBA")
                im.thumbnail((size, size), Image.LANCZOS)
                r, g, b, a = im.split()
                tinted = Image.new("RGBA", im.size, tint + (0,))
                tinted.putalpha(a)
                self._photo = ImageTk.PhotoImage(tinted)
                tk.Label(self, image=self._photo,
                         bg=bg_color, bd=0).pack()
                return
            except Exception:
                self._photo = None
        self._cv = tk.Canvas(self, width=size, height=size,
                             bg=C.UI_BG, highlightthickness=0, bd=0)
        self._cv.pack()
        self._render_logo()

    def _render_logo(self) -> None:
        c = self._cv
        c.delete("all")
        s = self.size
        cx, cy = s / 2, s / 2
        r = s / 2 - 1
        c.create_rectangle(2, 2, s - 2, s - 2,
                           outline=C.UI_BORDER_HI, width=1)
        c.create_oval(cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5,
                      outline=C.UI_ACCENT, width=1)
        c.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                      fill=C.UI_ACCENT, outline="")


class Keycap(ctk.CTkLabel):
    def __init__(self, master, text: str) -> None:
        super().__init__(
            master, text=text,
            text_color=C.UI_TEXT_DIM,
            font=_font(10, bold=True, mono=True),
            fg_color=C.UI_SURFACE_3,
            corner_radius=3,
            padx=5, pady=1,
        )


class IconButton(ctk.CTkButton):
    def __init__(self, master, *, text: str, command: Optional[Callable] = None,
                 shortcut: Optional[str] = None, width: int = 0,
                 accent: bool = False, danger: bool = False,
                 tooltip: Optional[str] = None) -> None:
        display = f"{text}   {shortcut}" if shortcut else text
        kwargs = dict(
            text=display, command=command, height=30,
            corner_radius=4,
            font=_font(11),
            border_width=1,
        )
        if accent:
            kwargs.update(
                fg_color=C.UI_ACCENT, hover_color=C.UI_ACCENT_HI,
                text_color=C.UI_ACCENT_TXT, border_color=C.UI_ACCENT,
            )
        elif danger:
            kwargs.update(
                fg_color=C.UI_SURFACE, hover_color="#2a0e0e",
                text_color=C.UI_BAD, border_color="#3a1818",
            )
        else:
            kwargs.update(
                fg_color=C.UI_SURFACE, hover_color=C.UI_SURFACE_3,
                text_color=C.UI_TEXT, border_color=C.UI_BORDER,
            )
        if width:
            kwargs["width"] = width
        super().__init__(master, **kwargs)
        self._accent = accent
        self._danger = danger
        if tooltip:
            Tooltip(self, tooltip)
        self._active = False

    def set_active(self, on: bool) -> None:
        self._active = on
        if on:
            self.configure(
                fg_color=C.UI_ACCENT_LO, text_color=C.UI_ACCENT_HI,
                border_color=C.UI_BORDER_HOT,
            )
        elif self._accent:
            self.configure(
                fg_color=C.UI_ACCENT, text_color=C.UI_ACCENT_TXT,
                border_color=C.UI_ACCENT,
            )
        elif self._danger:
            self.configure(
                fg_color=C.UI_SURFACE, text_color=C.UI_BAD,
                border_color="#3a1818",
            )
        else:
            self.configure(
                fg_color=C.UI_SURFACE, text_color=C.UI_TEXT,
                border_color=C.UI_BORDER,
            )


GlowButton = IconButton


class Switch(tk.Canvas):
    W = 48
    H = 24
    PAD = 3

    def __init__(self, master, variable: tk.BooleanVar,
                 *, on_change: Optional[Callable[[bool], None]] = None,
                 bg_color: str = C.UI_BG) -> None:
        super().__init__(master, width=self.W, height=self.H,
                         bg=bg_color, highlightthickness=0,
                         bd=0, cursor="hand2")
        self.var = variable
        self._on_change = on_change
        self._bg = bg_color
        self._hover = False
        self._knob_x = float(self._target_knob_x())
        self._track_t = 1.0 if bool(self.var.get()) else 0.0
        self._track_target = self._track_t
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        try:
            self.var.trace_add("write", self._on_var_change)
        except AttributeError:
            self.var.trace("w", self._on_var_change)
        self._redraw()
        self._tick()

    def _target_knob_x(self) -> float:
        if bool(self.var.get()):
            return float(self.W - self.H + self.PAD)
        return float(self.PAD)

    def _on_click(self, _e=None) -> None:
        self.var.set(not bool(self.var.get()))
        if self._on_change is not None:
            try:
                self._on_change(bool(self.var.get()))
            except Exception:
                pass

    def _on_enter(self, _e=None) -> None:
        self._hover = True
        self._redraw()

    def _on_leave(self, _e=None) -> None:
        self._hover = False
        self._redraw()

    def _on_var_change(self, *_a) -> None:
        self._track_target = 1.0 if bool(self.var.get()) else 0.0

    def _tick(self) -> None:
        target_x = self._target_knob_x()
        dx = target_x - self._knob_x
        if abs(dx) > 0.4:
            self._knob_x += dx * 0.30
        else:
            self._knob_x = target_x
        dt = self._track_target - self._track_t
        if abs(dt) > 0.01:
            self._track_t += dt * 0.30
        else:
            self._track_t = self._track_target
        self._redraw()
        self.after(28, self._tick)

    @staticmethod
    def _lerp_color(c0: str, c1: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        a = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
        b = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        return "#{:02x}{:02x}{:02x}".format(
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    def _redraw(self) -> None:
        self.delete("all")
        w, h, pad = self.W, self.H, self.PAD
        on = bool(self.var.get())

        track_off = "#404040"
        track_on = "#22c55e"
        track = self._lerp_color(track_off, track_on, self._track_t)

        r = h / 2
        self.create_oval(0, 0, h, h, fill=track, outline="")
        self.create_oval(w - h, 0, w, h, fill=track, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=track, outline="")

        glow = self._lerp_color("#404040", "#86efac", self._track_t)
        self.create_oval(0, 0, h, h, outline=glow, width=1)
        self.create_oval(w - h, 0, w, h, outline=glow, width=1)
        self.create_line(r, 0, w - r, 0, fill=glow, width=1)
        self.create_line(r, h - 1, w - r, h - 1, fill=glow, width=1)

        kx = self._knob_x
        ks = h - pad * 2
        knob_color = "#fafafa" if not self._hover else "#ffffff"
        self.create_oval(kx, pad, kx + ks, pad + ks,
                         fill=knob_color, outline=knob_color)


class ToggleRow(ctk.CTkFrame):
    def __init__(self, master, text: str, var: tk.BooleanVar,
                 *, on_change: Optional[Callable[[bool], None]] = None,
                 bg_color: str = C.UI_BG) -> None:
        super().__init__(master, fg_color="transparent", height=30)
        self.grid_columnconfigure(0, weight=1)
        self.var = var
        self._on_change = on_change
        self._lbl = ctk.CTkLabel(
            self, text=text, anchor="w",
            text_color=C.UI_TEXT_MID, font=_font(11),
            cursor="hand2",
        )
        self._lbl.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._sw = Switch(self, variable=var, on_change=self._fire,
                          bg_color=bg_color)
        self._sw.grid(row=0, column=1, padx=(0, 0), pady=4)
        for w in (self, self._lbl):
            w.bind("<Button-1>", self._on_label_click, add="+")
        try:
            self.var.trace_add("write", lambda *_: self._sync_label())
        except AttributeError:
            self.var.trace("w", lambda *_: self._sync_label())
        self._sync_label()

    def _sync_label(self) -> None:
        try:
            on = bool(self.var.get())
        except Exception:
            on = False
        self._lbl.configure(text_color=C.UI_TEXT if on else C.UI_TEXT_DIM)

    def _on_label_click(self, _e=None) -> None:
        self.var.set(not bool(self.var.get()))
        self._fire(bool(self.var.get()))

    def _fire(self, value: bool) -> None:
        if self._on_change is not None:
            self._on_change(value)


class Card(ctk.CTkFrame):
    def __init__(self, master, title: str = "", subtitle: str = "") -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        self._has_title = bool(title)
        if title:
            hdr = ctk.CTkFrame(self, fg_color="transparent")
            hdr.pack(fill="x", padx=12, pady=(10, 0))
            self._title = ctk.CTkLabel(
                hdr, text=title, anchor="w",
                text_color=C.UI_TEXT_DIM,
                font=_font(10, bold=True),
            )
            self._title.pack(side="left")
            self._subtitle = ctk.CTkLabel(
                hdr, text=subtitle, anchor="e",
                text_color=C.UI_TEXT_FAINT,
                font=_font(10, mono=True),
            )
            self._subtitle.pack(side="right")

    def set_title(self, text: str) -> None:
        if self._has_title:
            self._title.configure(text=text)

    def set_subtitle(self, text: str) -> None:
        if self._has_title:
            self._subtitle.configure(text=text)


class StatTile(ctk.CTkFrame):
    def __init__(self, master, label: str, *, accent: bool = False,
                 unit: str = "", precision: int = 0,
                 tooltip: Optional[str] = None) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        self._target = 0.0
        self._current = 0.0
        self._displayed = ""
        self._unit = unit
        self._precision = precision

        ctk.CTkLabel(
            self, text=label, anchor="w",
            text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
        ).pack(anchor="w", padx=12, pady=(9, 0))

        color = C.UI_ACCENT_HI if accent else C.UI_TEXT
        self._val = ctk.CTkLabel(
            self, text="0", anchor="w",
            text_color=color,
            font=ctk.CTkFont(family=C.FONT_MONO, size=22, weight="bold"),
        )
        self._val.pack(anchor="w", padx=12, pady=(0, 10))

        if tooltip:
            Tooltip(self, tooltip)
        self._tick()

    def set_value(self, v: float) -> None:
        self._target = float(v)

    def _tick(self) -> None:
        self._current = _lerp(self._current, self._target, 0.22)
        if self._precision == 0:
            txt = f"{int(round(self._current))}"
        else:
            txt = f"{self._current:.{self._precision}f}"
        if self._unit:
            txt = f"{txt}{self._unit}"
        if txt != self._displayed:
            self._val.configure(text=txt)
            self._displayed = txt
        self.after(40, self._tick)


class MetricBar(ctk.CTkFrame):
    def __init__(self, master, label: str, *,
                 min_val: float = 0.0, max_val: float = 1.0,
                 fmt: str = "{:.0%}", unit: str = "",
                 signed: bool = False, tooltip: Optional[str] = None) -> None:
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self._min = float(min_val)
        self._max = float(max_val)
        self._fmt = fmt
        self._unit = unit
        self._signed = signed
        self._target = 0.0
        self._current = 0.0
        self._displayed = ""
        self._last_w = -1

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            top, text=label, anchor="w",
            text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
        ).grid(row=0, column=0, sticky="w")
        self._val_lbl = ctk.CTkLabel(
            top, text="0", anchor="e",
            text_color=C.UI_TEXT,
            font=_font(11, bold=True, mono=True),
        )
        self._val_lbl.grid(row=0, column=1, sticky="e")

        self._cv = tk.Canvas(
            self, height=4, bg=C.UI_SURFACE_2, highlightthickness=0, bd=0,
        )
        self._cv.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._cv.bind("<Configure>", lambda e: self._redraw(force=True))
        if tooltip:
            Tooltip(self, tooltip)
        self._tick()

    def set_value(self, v: float) -> None:
        self._target = float(v)

    def _tick(self) -> None:
        prev = self._current
        self._current = _lerp(self._current, self._target, 0.2)
        self._redraw(force=abs(prev - self._current) > 0.001)
        self.after(40, self._tick)

    def _redraw(self, force: bool = False) -> None:
        c = self._cv
        w = max(1, c.winfo_width())
        h = c.winfo_height()
        if not force and w == self._last_w:
            pass
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill=C.UI_SURFACE_2, outline="")
        if self._signed:
            mid = w / 2.0
            half = w / 2.0
            denom = max(abs(self._min), abs(self._max), 1e-6)
            ratio = max(self._min, min(self._max, self._current)) / denom
            color = C.UI_ACCENT if ratio >= 0 else C.UI_WARN
            if ratio >= 0:
                c.create_rectangle(mid, 0, mid + half * ratio, h, fill=color, outline="")
            else:
                c.create_rectangle(mid + half * ratio, 0, mid, h, fill=color, outline="")
            c.create_line(mid, 0, mid, h, fill=C.UI_BORDER_HI)
        else:
            rng = self._max - self._min
            ratio = 0.0 if rng <= 0 else (self._current - self._min) / rng
            ratio = max(0.0, min(1.0, ratio))
            c.create_rectangle(0, 0, w * ratio, h, fill=C.UI_ACCENT, outline="")
        self._last_w = w
        txt = self._fmt.format(self._current) + self._unit
        if txt != self._displayed:
            self._val_lbl.configure(text=txt)
            self._displayed = txt


AnimatedBar = MetricBar
AnimatedCounter = StatTile


class Sparkline(ctk.CTkFrame):
    def __init__(self, master, label: str, *,
                 max_points: int = 90, height: int = 56,
                 y_range: Optional[Tuple[float, float]] = None,
                 redraw_every: int = 1) -> None:
        super().__init__(master, fg_color="transparent")
        self._values: Deque[float] = deque(maxlen=max_points)
        self._y_range = y_range
        self._redraw_every = max(1, int(redraw_every))
        self._push_count = 0
        if label:
            ctk.CTkLabel(
                self, text=label, anchor="w",
                text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
            ).pack(anchor="w", pady=(0, 4))
        self._cv = tk.Canvas(self, height=height, bg=C.UI_SURFACE,
                             highlightthickness=0, bd=0)
        self._cv.pack(fill="x", expand=True)
        self._cv.bind("<Configure>", lambda e: self._redraw())

    def push(self, v: float) -> None:
        self._values.append(float(v))
        self._push_count += 1
        if self._push_count % self._redraw_every == 0:
            self._redraw()

    def _redraw(self) -> None:
        c = self._cv
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        n = len(self._values)
        if n == 0:
            return
        if self._y_range is None:
            lo = min(self._values); hi = max(self._values)
            if hi - lo < 1e-6: hi = lo + 1.0
        else:
            lo, hi = self._y_range
        pad = 2
        pts: List[float] = []
        for i, v in enumerate(self._values):
            x = pad + (w - 2 * pad) * (i / max(1, n - 1))
            y = (h - pad) - (h - 2 * pad) * ((v - lo) / (hi - lo))
            pts.extend([x, y])
        if len(pts) >= 4:
            poly = pts + [w - pad, h - pad, pad, h - pad]
            c.create_polygon(*poly, fill=C.UI_ACCENT_LO, outline="")
            c.create_line(*pts, fill=C.UI_ACCENT, width=1, smooth=True)


class RadialGaze(ctk.CTkFrame):
    def __init__(self, master, *, size: int = 124) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        self._size = size
        ctk.CTkLabel(
            self, text="GAZE",
            text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
        ).pack(pady=(10, 2))
        self._cv = tk.Canvas(self, width=size, height=size,
                             bg=C.UI_SURFACE, highlightthickness=0)
        self._cv.pack(pady=(0, 2))
        self._dir = ctk.CTkLabel(
            self, text="CENTER",
            text_color=C.UI_ACCENT_HI,
            font=_font(11, bold=True, mono=True),
        )
        self._dir.pack(pady=(0, 10))
        self._gx, self._gy = 0.0, 0.0
        self._cx, self._cy = 0.0, 0.0
        self._tick()

    def set_gaze(self, gx: float, gy: float, label: str) -> None:
        self._gx = max(-1.0, min(1.0, gx))
        self._gy = max(-1.0, min(1.0, gy))
        if label:
            self._dir.configure(text=label.upper())

    def _tick(self) -> None:
        self._cx = _lerp(self._cx, self._gx, 0.25)
        self._cy = _lerp(self._cy, self._gy, 0.25)
        self._redraw()
        self.after(40, self._tick)

    def _redraw(self) -> None:
        c = self._cv
        c.delete("all")
        s = self._size
        cx, cy = s / 2, s / 2
        r = s / 2 - 8
        for k in (1.0, 0.66, 0.33):
            c.create_oval(cx - r * k, cy - r * k, cx + r * k, cy + r * k,
                          outline=C.UI_BORDER_HI)
        c.create_line(cx - r, cy, cx + r, cy, fill=C.UI_BORDER_HI)
        c.create_line(cx, cy - r, cx, cy + r, fill=C.UI_BORDER_HI)
        dx = cx + self._cx * r
        dy = cy + self._cy * r
        c.create_oval(dx - 6, dy - 6, dx + 6, dy + 6,
                      fill=C.UI_ACCENT, outline="")


class PoseDial(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(
            self, text="POSE TELEMETRY",
            text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(padx=6, pady=(2, 10))
        self._dials = []
        for name in ("PITCH", "YAW", "ROLL"):
            cont = ctk.CTkFrame(row, fg_color="transparent")
            cont.pack(side="left", padx=8)
            cv = tk.Canvas(cont, width=66, height=66,
                           bg=C.UI_SURFACE, highlightthickness=0)
            cv.pack()
            txt = ctk.CTkLabel(
                cont, text=f"{name}\n0",
                text_color=C.UI_TEXT,
                font=_font(10, mono=True),
            )
            txt.pack(pady=(2, 0))
            self._dials.append((cv, txt, name, C.UI_ACCENT))
        self._values = [0.0, 0.0, 0.0]
        self._targets = [0.0, 0.0, 0.0]
        self._tick()

    def set_values(self, pitch: float, yaw: float, roll: float) -> None:
        self._targets = [pitch, yaw, roll]

    def _tick(self) -> None:
        for i in range(3):
            self._values[i] = _lerp(self._values[i], self._targets[i], 0.2)
        self._redraw()
        self.after(40, self._tick)

    def _redraw(self) -> None:
        for (cv, txt, name, color), v in zip(self._dials, self._values):
            cv.delete("all")
            cx, cy, r = 33, 33, 26
            cv.create_oval(cx - r, cy - r, cx + r, cy + r,
                           outline=C.UI_BORDER_HI, width=1)
            cv.create_line(cx, cy - r + 1, cx, cy - r + 4, fill=C.UI_TEXT_FAINT)
            cv.create_line(cx, cy + r - 1, cx, cy + r - 4, fill=C.UI_TEXT_FAINT)
            cv.create_line(cx - r + 1, cy, cx - r + 4, cy, fill=C.UI_TEXT_FAINT)
            cv.create_line(cx + r - 1, cy, cx + r - 4, cy, fill=C.UI_TEXT_FAINT)
            ang = max(-90.0, min(90.0, v))
            rad = math.radians(ang - 90)
            ex = cx + math.cos(rad) * (r - 4)
            ey = cy + math.sin(rad) * (r - 4)
            cv.create_line(cx, cy, ex, ey, fill=color, width=2,
                           capstyle="round")
            cv.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                           fill=color, outline="")
            txt.configure(text=f"{name}\n{v:+.0f}")


class ShapeProbabilityList(ctk.CTkFrame):
    def __init__(self, master, shapes: List[str]) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(
            self, text="MORPHOLOGY SCORES",
            text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
        ).pack(anchor="w", padx=12, pady=(10, 6))
        self._rows: Dict[str, dict] = {}
        for s in shapes:
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=1)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=s, anchor="w", width=70,
                         text_color=C.UI_TEXT_MID, font=_font(11),
                         ).grid(row=0, column=0, sticky="w", padx=(0, 8))
            bar = tk.Canvas(row, height=4, bg=C.UI_SURFACE_2,
                            highlightthickness=0, bd=0)
            bar.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            pct = ctk.CTkLabel(row, text="0%", width=38, anchor="e",
                               text_color=C.UI_TEXT_DIM,
                               font=_font(10, mono=True),
                               )
            pct.grid(row=0, column=2, sticky="e")
            self._rows[s] = {"bar": bar, "pct": pct, "v": 0.0, "t": 0.0, "best": False}
        ctk.CTkLabel(self, text="").pack(pady=2)
        self._tick()

    def set_scores(self, scores: dict) -> None:
        if not scores:
            return
        best_key = max(scores.items(), key=lambda kv: kv[1])[0]
        for s, info in self._rows.items():
            info["t"] = float(scores.get(s, 0.0))
            info["best"] = (s == best_key)

    def _tick(self) -> None:
        for s, info in self._rows.items():
            info["v"] = _lerp(info["v"], info["t"], 0.18)
            bar = info["bar"]
            bar.delete("all")
            w = max(1, bar.winfo_width()); h = bar.winfo_height()
            bar.create_rectangle(0, 0, w, h, fill=C.UI_SURFACE_2, outline="")
            color = C.UI_ACCENT if info["best"] else C.UI_BORDER_HI
            bar.create_rectangle(0, 0, w * info["v"], h, fill=color, outline="")
            info["pct"].configure(text=f"{info['v'] * 100:.0f}%")
        self.after(50, self._tick)


class StatPill(ctk.CTkFrame):
    def __init__(self, master, label: str) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=999,
                         border_width=1, border_color=C.UI_BORDER, height=26)
        self._dot = tk.Canvas(self, width=10, height=10,
                              bg=C.UI_SURFACE, highlightthickness=0)
        self._dot.pack(side="left", padx=(10, 6), pady=4)
        self._dot.create_oval(2, 2, 8, 8, fill=C.UI_TEXT_FAINT,
                              outline="", tags="d")
        self._lbl = ctk.CTkLabel(
            self, text=label,
            text_color=C.UI_TEXT_MID, font=_font(11, mono=True),
        )
        self._lbl.pack(side="left", padx=(0, 12), pady=4)
        self._on = False

    def set_state(self, on: bool, text: Optional[str] = None) -> None:
        if on != self._on:
            self._pulse(0, C.UI_GOOD if on else C.UI_TEXT_FAINT)
        self._on = on
        color = C.UI_GOOD if on else C.UI_TEXT_FAINT
        self._dot.itemconfig("d", fill=color)
        if text is not None:
            self._lbl.configure(text=text)

    def _pulse(self, step: int, color: str) -> None:
        steps = 10
        if step >= steps:
            self._dot.delete("ring")
            return
        t = step / steps
        r = 3 + t * 6
        self._dot.delete("ring")
        self._dot.create_oval(5 - r, 5 - r, 5 + r, 5 + r,
                              outline=color, width=1, tags="ring")
        self.after(28, lambda: self._pulse(step + 1, color))


class TabBar(ctk.CTkFrame):
    def __init__(self, master, tabs: List[str],
                 on_change: Callable[[str], None]) -> None:
        super().__init__(master, fg_color="transparent", height=40)
        self._tabs = list(tabs)
        self._buttons: Dict[str, ctk.CTkButton] = {}
        self._on_change = on_change
        self._current = tabs[0] if tabs else ""
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")
        for name in tabs:
            b = ctk.CTkButton(
                row, text=name, command=lambda n=name: self.select(n),
                fg_color="transparent", hover_color=C.UI_SURFACE_3,
                text_color=C.UI_TEXT_MID, width=0, height=28,
                corner_radius=4, font=_font(11, bold=True),
            )
            b.pack(side="left", padx=2)
            self._buttons[name] = b
        self._line = tk.Canvas(self, height=2, bg=C.UI_BG, highlightthickness=0)
        self._line.pack(fill="x", padx=0, pady=(2, 0))
        self._indicator_x = 0.0
        self._indicator_w = 0.0
        self._target_x = 0.0
        self._target_w = 0.0
        self.after(80, lambda: self.select(self._current, animate=False))
        self.after(20, self._tick)

    def select(self, name: str, animate: bool = True) -> None:
        if name not in self._buttons:
            return
        self._current = name
        for k, b in self._buttons.items():
            if k == name:
                b.configure(text_color=C.UI_ACCENT_HI)
            else:
                b.configure(text_color=C.UI_TEXT_DIM)
        b = self._buttons[name]
        b.update_idletasks()
        self._target_x = b.winfo_x()
        self._target_w = b.winfo_width()
        if not animate:
            self._indicator_x = self._target_x
            self._indicator_w = self._target_w
        self._on_change(name)

    def _tick(self) -> None:
        self._indicator_x = _lerp(self._indicator_x, self._target_x, 0.28)
        self._indicator_w = _lerp(self._indicator_w, self._target_w, 0.28)
        c = self._line
        c.delete("all")
        total = max(1, c.winfo_width())
        c.create_rectangle(0, 0, total, c.winfo_height(),
                           fill=C.UI_BORDER, outline="")
        c.create_rectangle(
            self._indicator_x, 0,
            self._indicator_x + self._indicator_w, c.winfo_height(),
            fill=C.UI_ACCENT, outline="",
        )
        self.after(28, self._tick)


class ToastHost:
    def __init__(self, parent: tk.Misc) -> None:
        self._parent = parent
        self._stack: List[dict] = []

    def show(self, message: str, *, kind: str = "info",
             duration_ms: int = 2600) -> None:
        accent = {
            "info": C.UI_ACCENT,
            "success": C.UI_GOOD,
            "warn": C.UI_WARN,
            "error": C.UI_BAD,
        }.get(kind, C.UI_ACCENT)
        top = tk.Toplevel(self._parent)
        top.wm_overrideredirect(True)
        top.attributes("-topmost", True)
        try:
            top.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        outer = tk.Frame(top, bg=C.UI_SURFACE_3,
                         highlightbackground=C.UI_BORDER_HI, highlightthickness=1)
        outer.pack()
        stripe = tk.Frame(outer, bg=accent, width=3)
        stripe.pack(side="left", fill="y")
        body = tk.Frame(outer, bg=C.UI_SURFACE_3)
        body.pack(side="left", padx=(10, 14), pady=8)
        tk.Label(body, text=message, bg=C.UI_SURFACE_3, fg=C.UI_TEXT,
                 font=(C.FONT_MONO, 10),
                 ).pack(anchor="w")
        entry = {"top": top, "alpha": 0.0, "stage": "in",
                 "shown_at": time.time(), "duration": duration_ms / 1000.0}
        self._stack.append(entry)
        self._reposition()
        self._animate(entry)

    def _reposition(self) -> None:
        try:
            self._parent.update_idletasks()
            rx = self._parent.winfo_rootx()
            ry = self._parent.winfo_rooty()
            rw = self._parent.winfo_width()
            rh = self._parent.winfo_height()
        except Exception:
            return
        base_y = ry + rh - 50
        for entry in reversed(self._stack):
            top = entry["top"]
            try:
                top.update_idletasks()
                tw = top.winfo_width()
                th = top.winfo_height()
                x = rx + rw - tw - 20
                top.geometry(f"+{x}+{base_y - th}")
                base_y -= th + 8
            except Exception:
                pass

    def _animate(self, entry: dict) -> None:
        top = entry["top"]
        try:
            if not top.winfo_exists():
                return
        except Exception:
            return
        now = time.time()
        if entry["stage"] == "in":
            entry["alpha"] = min(1.0, entry["alpha"] + 0.14)
            try:
                top.attributes("-alpha", entry["alpha"])
            except tk.TclError:
                pass
            if entry["alpha"] >= 0.99 and (now - entry["shown_at"]) > entry["duration"]:
                entry["stage"] = "out"
        else:
            entry["alpha"] = max(0.0, entry["alpha"] - 0.10)
            try:
                top.attributes("-alpha", entry["alpha"])
            except tk.TclError:
                pass
            if entry["alpha"] <= 0.01:
                try:
                    top.destroy()
                except Exception:
                    pass
                try:
                    self._stack.remove(entry)
                except ValueError:
                    pass
                self._reposition()
                return
        self._reposition()
        self._parent.after(28, self._animate, entry)


class StatusStrip(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=0,
                         height=26, border_width=0)
        self._cells: Dict[str, ctk.CTkLabel] = {}
        self._left = ctk.CTkFrame(self, fg_color="transparent")
        self._left.pack(side="left", padx=12, fill="y")
        self._right = ctk.CTkFrame(self, fg_color="transparent")
        self._right.pack(side="right", padx=12, fill="y")

    def _add(self, container, key: str, text: str, *, accent: bool = False) -> None:
        if self._cells:
            sep = ctk.CTkLabel(container, text="|",
                               text_color=C.UI_TEXT_FAINT, font=_font(11),
                               )
            sep.pack(side="left", padx=8)
        lbl = ctk.CTkLabel(
            container, text=text,
            text_color=C.UI_ACCENT_HI if accent else C.UI_TEXT_DIM,
            font=_font(10, mono=True),
        )
        lbl.pack(side="left")
        self._cells[key] = lbl

    def add_left(self, key: str, text: str = "", *, accent: bool = False) -> None:
        self._add(self._left, key, text, accent=accent)

    def add_right(self, key: str, text: str = "", *, accent: bool = False) -> None:
        self._add(self._right, key, text, accent=accent)

    def set(self, key: str, text: str) -> None:
        if key in self._cells:
            cell = self._cells[key]
            if cell.cget("text") != text:
                cell.configure(text=text)


class HistogramPanel(ctk.CTkFrame):
    def __init__(self, master, *, height: int = 100) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(self, text="RGB HISTOGRAM",
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=12, pady=(8, 0))
        self._cv = tk.Canvas(self, height=height, bg=C.UI_SURFACE,
                             highlightthickness=0, bd=0)
        self._cv.pack(fill="x", expand=True, padx=10, pady=(4, 10))
        self._hist = None

    def set_hist(self, hist):
        self._hist = hist
        self._redraw()

    def _redraw(self) -> None:
        c = self._cv
        c.delete("all")
        if self._hist is None:
            return
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        bins = self._hist.shape[1]
        colors = ("#dc2626", "#22c55e", "#3b82f6")
        for i, color in enumerate(colors):
            pts = []
            for b in range(bins):
                x = (b / max(1, bins - 1)) * w
                y = h - self._hist[i, b] * (h - 2)
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=1, smooth=True)


class PerfGraph(ctk.CTkFrame):
    def __init__(self, master, *, height: int = 80, max_points: int = 120) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(self, text="PERFORMANCE",
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=12, pady=(8, 0))
        self._fps: Deque[float] = deque(maxlen=max_points)
        self._latency: Deque[float] = deque(maxlen=max_points)
        self._cv = tk.Canvas(self, height=height, bg=C.UI_SURFACE,
                             highlightthickness=0, bd=0)
        self._cv.pack(fill="x", expand=True, padx=10, pady=(4, 4))
        self._lbl = ctk.CTkLabel(self, text=" - ",
                                 text_color=C.UI_TEXT_DIM, font=_font(10, mono=True),
                                 )
        self._lbl.pack(anchor="w", padx=12, pady=(0, 8))

    def push(self, fps: float, latency_ms: float) -> None:
        self._fps.append(fps)
        self._latency.append(latency_ms)
        self._redraw()

    def _redraw(self) -> None:
        c = self._cv
        c.delete("all")
        if not self._fps:
            return
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        if self._fps:
            fmax = 60.0
            pts = []
            for i, v in enumerate(self._fps):
                x = (i / max(1, len(self._fps) - 1)) * w
                y = h - (min(v, fmax) / fmax) * (h - 2)
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(*pts, fill=C.UI_ACCENT, width=1, smooth=True)
        if self._latency:
            lmax = 80.0
            pts = []
            for i, v in enumerate(self._latency):
                x = (i / max(1, len(self._latency) - 1)) * w
                y = h - (min(v, lmax) / lmax) * (h - 2)
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(*pts, fill=C.UI_BAD, width=1, smooth=True)
        avg_fps = sum(self._fps) / max(1, len(self._fps))
        avg_lat = sum(self._latency) / max(1, len(self._latency))
        self._lbl.configure(text=f"avg fps {avg_fps:5.1f}   -   avg latency {avg_lat:4.1f} ms")


class DataTable(ctk.CTkFrame):
    def __init__(self, master, title: str, rows: List[str]) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(self, text=title.upper(),
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=12, pady=(10, 4))
        self._rows: Dict[str, ctk.CTkLabel] = {}
        for r in rows:
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=1)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=r, anchor="w",
                         text_color=C.UI_TEXT_MID, font=_font(11, mono=True),
                         ).grid(row=0, column=0, sticky="w")
            val = ctk.CTkLabel(row, text=" - ", anchor="e",
                               text_color=C.UI_ACCENT_HI,
                               font=_font(11, bold=True, mono=True),
                               )
            val.grid(row=0, column=1, sticky="e")
            self._rows[r] = val
        ctk.CTkLabel(self, text="").pack(pady=2)

    def set(self, key: str, value: str) -> None:
        if key in self._rows and self._rows[key].cget("text") != value:
            self._rows[key].configure(text=value)


class HeatmapWidget(ctk.CTkFrame):
    def __init__(self, master, *, height: int = 90) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(self, text="POSITION HEATMAP",
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=12, pady=(8, 0))
        self._coverage = ctk.CTkLabel(self, text="coverage 0%",
                                      text_color=C.UI_TEXT_FAINT,
                                      font=_font(10, mono=True),
                                      )
        self._coverage.pack(anchor="w", padx=12)
        self._cv = tk.Canvas(self, height=height, bg=C.UI_SURFACE,
                             highlightthickness=0, bd=0)
        self._cv.pack(fill="x", expand=True, padx=10, pady=(2, 10))
        self._cv.bind("<Configure>", lambda e: self._redraw())
        self._grid = None

    def set_grid(self, grid, coverage_pct: float) -> None:
        self._grid = grid
        self._coverage.configure(text=f"coverage  {coverage_pct:.1f}%")
        self._redraw()

    def _redraw(self) -> None:
        c = self._cv
        c.delete("all")
        if self._grid is None:
            return
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        gh, gw = self._grid.shape
        cw, ch = w / gw, h / gh
        m = float(self._grid.max()) or 1.0
        for y in range(gh):
            for x in range(gw):
                v = float(self._grid[y, x]) / m
                if v < 0.02:
                    continue
                rr = int(212 * v + 30 * (1 - v))
                gg = int(160 * v + 30 * (1 - v))
                bb = int(23 * v + 30 * (1 - v))
                color = f"#{rr:02x}{gg:02x}{bb:02x}"
                c.create_rectangle(x * cw, y * ch, (x + 1) * cw, (y + 1) * ch,
                                   fill=color, outline="")


class SnapshotStrip(ctk.CTkFrame):
    def __init__(self, master, *, max_thumbs: int = 6,
                 thumb_size: Tuple[int, int] = (78, 44)) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        ctk.CTkLabel(self, text="RECENT SNAPSHOTS",
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=12, pady=(8, 4))
        self._strip = ctk.CTkFrame(self, fg_color="transparent")
        self._strip.pack(fill="x", padx=10, pady=(0, 10))
        self._max = max_thumbs
        self._thumb_size = thumb_size
        self._photos: List = []
        self._empty = ctk.CTkLabel(self._strip, text="(none yet)",
                                   text_color=C.UI_TEXT_FAINT,
                                   font=_font(10, mono=True),
                                   )
        self._empty.pack(anchor="w", padx=4)

    def push(self, frame_bgr) -> None:
        try:
            from PIL import Image, ImageTk
            import cv2
            if self._empty is not None:
                try:
                    self._empty.destroy()
                except Exception:
                    pass
                self._empty = None
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.thumbnail(self._thumb_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            lbl = tk.Label(self._strip, image=photo, bg=C.UI_SURFACE_2,
                           bd=1, relief="solid",
                           highlightbackground=C.UI_BORDER)
            lbl.image = photo
            lbl.pack(side="left", padx=2)
            self._photos.append(photo)
            while len(self._photos) > self._max:
                self._photos.pop(0)
                children = list(self._strip.winfo_children())
                if children:
                    try:
                        children[0].destroy()
                    except Exception:
                        pass
        except Exception:
            pass


class StatusBadge(ctk.CTkFrame):
    def __init__(self, master, title: str) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER)
        self._title = title
        ctk.CTkLabel(self, text=title,
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=12, pady=(8, 0))
        self._value = ctk.CTkLabel(self, text=" - ",
                                   text_color=C.UI_ACCENT_HI,
                                   font=ctk.CTkFont(family=C.FONT_SANS,
                                                    size=18, weight="bold"),
                                   )
        self._value.pack(anchor="w", padx=12, pady=(0, 2))
        self._detail = ctk.CTkLabel(self, text="",
                                    text_color=C.UI_TEXT_MID,
                                    font=_font(10, mono=True),
                                    )
        self._detail.pack(anchor="w", padx=12, pady=(0, 4))
        self._band = tk.Canvas(self, height=3, bg=C.UI_SURFACE_2,
                               highlightthickness=0, bd=0)
        self._band.pack(fill="x", padx=0, pady=(0, 0))
        self._band.bind("<Configure>", lambda e: self._render_band(0.0,
                                                                   C.UI_ACCENT))

    def set(self, value: str, detail: str = "",
            level: float = 0.0, color: str = C.UI_ACCENT) -> None:
        if self._value.cget("text") != value:
            self._value.configure(text=value)
        if self._detail.cget("text") != detail:
            self._detail.configure(text=detail)
        self._render_band(max(0.0, min(1.0, level)), color)

    def _render_band(self, level: float, color: str) -> None:
        c = self._band
        c.delete("all")
        w = max(1, c.winfo_width())
        h = c.winfo_height()
        c.create_rectangle(0, 0, w, h, fill=C.UI_SURFACE_3, outline="")
        c.create_rectangle(0, 0, w * level, h, fill=color, outline="")


class BookmarkList(ctk.CTkScrollableFrame):
    def __init__(self, master, height: int = 200) -> None:
        super().__init__(master, fg_color=C.UI_SURFACE, corner_radius=4,
                         border_width=1, border_color=C.UI_BORDER, height=height,
                         scrollbar_button_color=C.UI_SURFACE_3,
                         scrollbar_button_hover_color=C.UI_BORDER_HI)
        self._items: List[tk.Misc] = []
        ctk.CTkLabel(self, text="BOOKMARKS",
                     text_color=C.UI_TEXT_DIM, font=_font(10, bold=True),
                     ).pack(anchor="w", padx=4, pady=(0, 4))

    def add(self, time_str: str, label: str) -> None:
        row = ctk.CTkFrame(self, fg_color=C.UI_SURFACE_2, corner_radius=4)
        row.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(row, text=time_str, text_color=C.UI_ACCENT_HI,
                     font=_font(10, mono=True),
                     ).pack(side="left", padx=(8, 6), pady=4)
        ctk.CTkLabel(row, text=label, text_color=C.UI_TEXT,
                     font=_font(10),
                     ).pack(side="left", pady=4)
        self._items.append(row)

    def clear(self) -> None:
        for it in self._items:
            try:
                it.destroy()
            except Exception:
                pass
        self._items.clear()
