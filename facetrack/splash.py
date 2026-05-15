from __future__ import annotations

import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk
import tkinter as tk

from . import constants as C


def _find_brand_image() -> Optional[str]:
    candidates: List[Path] = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "assets" / "brand.png")  # type: ignore[attr-defined]
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "assets" / "brand.png")
    candidates.append(here / "assets" / "brand.png")
    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except OSError:
            continue
    return None


@dataclass
class BootStep:
    key: str
    label: str
    func: Callable[[Callable[[float, str], None], Dict[str, Any]], Any]
    weight: float = 1.0
    skippable: bool = False


@dataclass
class BootResult:
    ok: bool
    cancelled: bool
    shared: Dict[str, Any] = field(default_factory=dict)
    failed_step: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    skipped_steps: List[str] = field(default_factory=list)


class SplashScreen:
    WIDTH = 540
    HEIGHT = 460

    def __init__(self, title: str = "Face Tracker Pro") -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title(title)
        self.root.configure(fg_color=C.UI_BG)
        self.root.resizable(False, False)
        self._center_window()

        self._title = title
        self._steps: List[BootStep] = []
        self._shared: Dict[str, Any] = {}
        self._result = BootResult(ok=False, cancelled=False)
        self._done = False
        self._cancel_requested = False
        self._worker: Optional[threading.Thread] = None
        self._spinner_angle = 0
        self._current_progress = 0.0
        self._target_progress = 0.0
        self._failed_step_idx: Optional[int] = None
        self._failed_step_skippable: bool = False
        self._skipped_steps: List[str] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.root.after(33, self._tick_spinner)
        self.root.after(33, self._tick_progress)

    def _center_window(self) -> None:
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - self.WIDTH) // 2
        y = (sh - self.HEIGHT) // 2
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(
            self.root, fg_color=C.UI_SURFACE, corner_radius=4,
            border_width=1, border_color=C.UI_BORDER_HI,
        )
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        header = ctk.CTkFrame(outer, fg_color=C.UI_SURFACE_2, height=28,
                              corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="  OPERATIONS // BOOT SEQUENCE",
                     text_color=C.UI_ACCENT_HI,
                     font=ctk.CTkFont(family=C.FONT_MONO, size=11, weight="bold"),
                     ).pack(side="left", padx=10, pady=4)
        ctk.CTkLabel(header, text=f"v1.0.0  ",
                     text_color=C.UI_TEXT_FAINT,
                     font=ctk.CTkFont(family=C.FONT_MONO, size=10),
                     ).pack(side="right", padx=10, pady=4)

        top = ctk.CTkFrame(outer, fg_color="transparent")
        top.pack(pady=(22, 4))

        brand_path = _find_brand_image()
        self._brand_photo = None
        if brand_path is not None:
            try:
                from PIL import Image, ImageTk
                im = Image.open(brand_path).convert("RGBA")
                im.thumbnail((96, 96), Image.LANCZOS)
                r, g, b, a = im.split()
                tint = Image.new("RGBA", im.size, (212, 160, 23, 0))
                tint.putalpha(a)
                self._brand_photo = ImageTk.PhotoImage(tint)
                tk.Label(top, image=self._brand_photo,
                         bg=C.UI_SURFACE, bd=0).pack(pady=(0, 6))
            except Exception:
                self._brand_photo = None

        self._spinner = tk.Canvas(
            top, width=60, height=60,
            bg=C.UI_SURFACE, highlightthickness=0,
        )
        self._spinner.pack()

        ctk.CTkLabel(
            outer, text="FACE TRACKER",
            text_color=C.UI_TEXT,
            font=ctk.CTkFont(family=C.FONT_SANS, size=20, weight="bold"),
        ).pack(pady=(10, 0))
        ctk.CTkLabel(
            outer, text="real-time biometric analysis  //  classified",
            text_color=C.UI_TEXT_DIM,
            font=ctk.CTkFont(family=C.FONT_MONO, size=10),
        ).pack(pady=(0, 18))

        self._step_label = ctk.CTkLabel(
            outer, text="$ initializing...",
            text_color=C.UI_TEXT,
            font=ctk.CTkFont(family=C.FONT_MONO, size=12),
        )
        self._step_label.pack(pady=(0, 4))

        self._sub_label = ctk.CTkLabel(
            outer, text="",
            text_color=C.UI_TEXT_DIM,
            font=ctk.CTkFont(family=C.FONT_MONO, size=10),
        )
        self._sub_label.pack(pady=(0, 8))

        bar_wrap = ctk.CTkFrame(outer, fg_color="transparent")
        bar_wrap.pack(pady=(2, 10), fill="x", padx=40)
        self._progress = ctk.CTkProgressBar(
            bar_wrap, height=6,
            progress_color=C.UI_ACCENT,
            fg_color=C.UI_SURFACE_3,
            corner_radius=2,
        )
        self._progress.pack(fill="x")
        self._progress.set(0.0)

        self._error_frame = ctk.CTkFrame(outer, fg_color="transparent")
        self._error_label = ctk.CTkLabel(
            self._error_frame, text="",
            text_color=C.UI_BAD,
            font=ctk.CTkFont(family=C.FONT_MONO, size=11),
            wraplength=460, justify="left", anchor="w",
        )
        self._error_label.pack(fill="x", padx=20, pady=(0, 6))

        self._hint_label = ctk.CTkLabel(
            self._error_frame, text="",
            text_color=C.UI_TEXT_DIM,
            font=ctk.CTkFont(family=C.FONT_MONO, size=10),
            wraplength=460, justify="left", anchor="w",
        )
        self._hint_label.pack(fill="x", padx=20, pady=(0, 6))

        btn_row = ctk.CTkFrame(self._error_frame, fg_color="transparent")
        btn_row.pack(pady=(2, 4))
        self._retry_btn = ctk.CTkButton(
            btn_row, text="Retry", width=100,
            fg_color=C.UI_ACCENT, text_color=C.UI_ACCENT_TXT,
            hover_color=C.UI_ACCENT_HI,
            command=self._on_retry,
        )
        self._retry_btn.pack(side="left", padx=3)
        self._skip_btn = ctk.CTkButton(
            btn_row, text="Skip step", width=100,
            fg_color=C.UI_SURFACE_2, text_color=C.UI_TEXT,
            hover_color=C.UI_SURFACE_3,
            border_width=1, border_color=C.UI_BORDER_HI,
            command=self._on_skip,
        )
        self._close_btn = ctk.CTkButton(
            btn_row, text="Abort", width=100,
            fg_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
            hover_color=C.UI_BORDER_HI,
            command=self._on_cancel,
        )
        self._close_btn.pack(side="left", padx=3)

    def add_step(
        self,
        key: str,
        label: str,
        func: Callable[[Callable[[float, str], None], Dict[str, Any]], Any],
        weight: float = 1.0,
        skippable: bool = False,
    ) -> None:
        self._steps.append(BootStep(key=key, label=label, func=func,
                                    weight=weight, skippable=skippable))

    def run(self) -> BootResult:
        self._start_worker()
        self.root.mainloop()
        return self._result

    def _start_worker(self, start_idx: int = 0) -> None:
        self._done = False
        self._cancel_requested = False
        if start_idx == 0:
            self._target_progress = 0.0
            self._current_progress = 0.0
            self._progress.set(0.0)
        self._error_frame.pack_forget()
        self._step_label.configure(text_color=C.UI_TEXT)
        self._worker = threading.Thread(
            target=self._run_steps, args=(start_idx,), daemon=True,
        )
        self._worker.start()

    def _run_steps(self, start_idx: int = 0) -> None:
        try:
            total_weight = sum(s.weight for s in self._steps) or 1.0
            accumulated = sum(s.weight for s in self._steps[:start_idx])
            for i in range(start_idx, len(self._steps)):
                step = self._steps[i]
                if self._cancel_requested:
                    self._result = BootResult(ok=False, cancelled=True,
                                              shared=self._shared,
                                              skipped_steps=list(self._skipped_steps))
                    self.root.after(0, self._finish_close)
                    return
                self.root.after(0, self._set_step_label, step.label, "")
                start_acc = accumulated
                step_weight = step.weight

                def progress_cb(p: float, message: str = "") -> None:
                    frac = max(0.0, min(1.0, p))
                    total = (start_acc + frac * step_weight) / total_weight
                    self.root.after(0, self._set_progress, total, message)

                try:
                    result = step.func(progress_cb, self._shared)
                    self._shared[step.key] = result
                except Exception as e:
                    tb = traceback.format_exc()
                    self._failed_step_idx = i
                    self._failed_step_skippable = step.skippable
                    self._result = BootResult(
                        ok=False, cancelled=False, shared=self._shared,
                        failed_step=step.label,
                        error_message=f"{type(e).__name__}: {e}",
                        error_traceback=tb,
                        skipped_steps=list(self._skipped_steps),
                    )
                    self.root.after(0, self._show_error)
                    return
                accumulated += step_weight
                self.root.after(0, self._set_progress,
                                accumulated / total_weight, "")
                time.sleep(0.10)

            self._result = BootResult(ok=True, cancelled=False,
                                      shared=self._shared,
                                      skipped_steps=list(self._skipped_steps))
            self.root.after(0, self._finish_success)
        except Exception as e:
            tb = traceback.format_exc()
            self._result = BootResult(
                ok=False, cancelled=False, shared=self._shared,
                failed_step="Internal error",
                error_message=f"{type(e).__name__}: {e}",
                error_traceback=tb,
                skipped_steps=list(self._skipped_steps),
            )
            self.root.after(0, self._show_error)

    def _set_step_label(self, label: str, sub: str) -> None:
        self._step_label.configure(text=f"$ {label.lower()}")
        self._sub_label.configure(text=sub)

    def _set_progress(self, value: float, sub: str) -> None:
        self._target_progress = max(0.0, min(1.0, value))
        if sub:
            self._sub_label.configure(text=sub)

    def _tick_progress(self) -> None:
        if self._done:
            return
        d = self._target_progress - self._current_progress
        self._current_progress += d * 0.18
        self._progress.set(self._current_progress)
        self.root.after(33, self._tick_progress)

    def _tick_spinner(self) -> None:
        if self._done:
            return
        c = self._spinner
        c.delete("all")
        cx, cy, r = 34, 34, 26
        self._spinner_angle = (self._spinner_angle + 7) % 360
        c.create_oval(cx - r, cy - r, cx + r, cy + r,
                      outline=C.UI_PANEL_2, width=4)
        c.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=-self._spinner_angle, extent=110,
            outline=C.UI_ACCENT, width=4, style="arc",
        )
        ang = math.radians(-self._spinner_angle + 110)
        dot_x = cx + math.cos(ang) * r
        dot_y = cy + math.sin(ang) * r
        c.create_oval(dot_x - 4, dot_y - 4, dot_x + 4, dot_y + 4,
                      fill=C.UI_ACCENT, outline="")
        c.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                      fill=C.UI_ACCENT_2, outline="")
        self.root.after(33, self._tick_spinner)

    def _show_error(self) -> None:
        self._step_label.configure(
            text=f"$ failed: {self._result.failed_step.lower()}",
            text_color=C.UI_BAD,
        )
        self._error_label.configure(text=self._result.error_message or "Unknown error")
        hint = self._derive_hint()
        if hint:
            self._hint_label.configure(text=hint)
        else:
            self._hint_label.configure(text="")
        try:
            self._skip_btn.pack_forget()
        except Exception:
            pass
        if self._failed_step_skippable:
            self._skip_btn.pack(side="left", padx=3,
                                in_=self._retry_btn.master, before=self._close_btn)
        self._error_frame.pack(pady=(6, 6), fill="x")

    def _derive_hint(self) -> str:
        msg = (self._result.error_message or "").lower()
        if "no module named" in msg or "importerror" in msg or "modulenotfounderror" in msg:
            return ("Hint: a Python dependency is missing.\n"
                    "Run:  python -m pip install -r requirements.txt")
        if "webcam" in msg or "camera" in msg:
            return ("Hint: probed every camera index from 0 to 5 and none returned a frame.\n"
                    "Close any app holding the camera (Zoom / Teams / OBS / Discord) "
                    "and click Retry, or click Skip to launch the app without a camera "
                    "(you can pick one from the sidebar later).")
        if "facemesh" in msg or "mediapipe" in msg or "landmarker" in msg:
            return ("Hint: MediaPipe failed to load its model. Try:\n"
                    "  python -m pip install --force-reinstall mediapipe")
        return ""

    def _on_retry(self) -> None:
        self._error_frame.pack_forget()
        start = self._failed_step_idx if self._failed_step_idx is not None else 0
        self._start_worker(start_idx=start)

    def _on_skip(self) -> None:
        if self._failed_step_idx is None:
            return
        step = self._steps[self._failed_step_idx]
        self._skipped_steps.append(step.key)
        self._shared[step.key] = None
        self._error_frame.pack_forget()
        self._start_worker(start_idx=self._failed_step_idx + 1)

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        self._result = BootResult(ok=False, cancelled=True, shared=self._shared)
        self._finish_close()

    def _finish_success(self) -> None:
        self._done = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _finish_close(self) -> None:
        self._done = True
        try:
            self.root.destroy()
        except Exception:
            pass
