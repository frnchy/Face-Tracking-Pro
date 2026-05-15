from __future__ import annotations

import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from . import constants as C
from . import features as F
from .features import apply_filter
from .analyzer import FaceAnalyzer, FaceMetrics
from .cameras import CameraInfo, best_resolution_for, list_cameras
from .filtering import PreprocessConfig, preprocess
from .monitors import (
    DrowsinessMonitor,
    EyeStrainReminder,
    FaceHeatmap,
    PostureMonitor,
    battery_status,
)
from .session import (
    Bookmark,
    CsvLogger,
    PROFILES_DIR,
    Profile,
    SNAPSHOTS_DIR,
    SessionStats,
    export_metrics_json,
    new_session_id,
)
from .tracker import FaceData, FaceTracker
from .widgets import (
    BookmarkList,
    BrandLogo,
    Card,
    DataTable,
    HeatmapWidget,
    HistogramPanel,
    IconButton,
    MetricBar,
    PerfGraph,
    PoseDial,
    RadialGaze,
    ShapeProbabilityList,
    SnapshotStrip,
    Sparkline,
    StatPill,
    StatTile,
    StatusBadge,
    StatusStrip,
    TabBar,
    ToastHost,
    ToggleRow,
    Tooltip,
)


class CaptureThread(threading.Thread):
    def __init__(self, cam_index: int = 0,
                 resolution: Tuple[int, int] = (1280, 720)) -> None:
        super().__init__(daemon=True)
        self._cam_index = cam_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._opened = False
        self._req_res = resolution
        self._actual_res: Tuple[int, int] = (0, 0)
        self._frame_seq = 0

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        return self._actual_res

    def open(self) -> bool:
        if sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._cam_index)
        if not self._cap or not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_res[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_res[1])
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._actual_res = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._opened = True
        return True

    @property
    def opened(self) -> bool:
        return self._opened

    def run(self) -> None:
        if not self._opened and not self.open():
            return
        self._running = True
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_seq += 1
        try:
            self._cap.release()
        except Exception:
            pass

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, self._frame_seq
            return self._frame.copy(), self._frame_seq

    def stop(self) -> None:
        self._running = False


def _ask_text(parent, title: str, prompt: str, *,
              default: str = "") -> Optional[str]:
    dlg = ctk.CTkToplevel(parent)
    dlg.title(title)
    dlg.configure(fg_color=C.UI_SURFACE)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)
    dlg.geometry("420x150")
    px = parent.winfo_rootx() + parent.winfo_width() // 2 - 210
    py = parent.winfo_rooty() + parent.winfo_height() // 2 - 75
    dlg.geometry(f"+{px}+{py}")

    ctk.CTkLabel(dlg, text=prompt,
                 text_color=C.UI_TEXT_MID,
                 font=ctk.CTkFont(family=C.FONT_SANS, size=12),
                 ).pack(anchor="w", padx=16, pady=(14, 6))
    entry = ctk.CTkEntry(
        dlg, font=ctk.CTkFont(family=C.FONT_MONO, size=12),
        fg_color=C.UI_SURFACE_2, text_color=C.UI_TEXT,
        border_color=C.UI_BORDER_HI, border_width=1,
    )
    entry.insert(0, default)
    entry.pack(fill="x", padx=16, pady=(0, 12))
    entry.focus_set()
    entry.select_range(0, "end")

    result: Dict[str, Optional[str]] = {"value": None}
    row = ctk.CTkFrame(dlg, fg_color="transparent")
    row.pack(fill="x", padx=16, pady=(0, 14))

    def ok(*_):
        result["value"] = entry.get().strip() or None
        dlg.destroy()

    def cancel(*_):
        result["value"] = None
        dlg.destroy()

    ctk.CTkButton(row, text="Cancel", command=cancel, width=90,
                  fg_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
                  hover_color=C.UI_BORDER_HI,
                  ).pack(side="right", padx=(6, 0))
    ctk.CTkButton(row, text="OK", command=ok, width=90,
                  fg_color=C.UI_ACCENT, text_color=C.UI_ACCENT_TXT,
                  hover_color=C.UI_ACCENT_HI,
                  ).pack(side="right")
    dlg.bind("<Return>", ok)
    dlg.bind("<Escape>", cancel)
    parent.wait_window(dlg)
    return result["value"]


class FaceTrackerApp:
    APP_TITLE = "FACE TRACKER // OPS"
    APP_VERSION = "1.0.0"

    def __init__(
        self,
        tracker: Optional[FaceTracker] = None,
        analyzer: Optional[FaceAnalyzer] = None,
    ) -> None:
        ctk.set_appearance_mode("dark")

        self._setup_dpi_awareness()

        self.root = ctk.CTk()
        try:
            scaling = float(self.root.tk.call("tk", "scaling"))
        except tk.TclError:
            scaling = 1.0
        widget_scale = max(0.9, min(1.6, scaling * 0.72))
        self._widget_scale = widget_scale
        ctk.set_widget_scaling(widget_scale)
        ctk.set_window_scaling(widget_scale)
        self._last_dpi: Optional[float] = None

        self.session_id = new_session_id()
        self.root.title(f"{self.APP_TITLE} // {self.session_id}")
        self.root.geometry("1600x950")
        self.root.minsize(960, 640)
        self.root.configure(fg_color=C.UI_BG)
        self._brand_png_path = self._find_asset("brand.png")
        ico = self._find_asset("brand.ico")
        if ico:
            try:
                self.root.iconbitmap(default=ico)
            except tk.TclError:
                pass

        self.tracker = tracker if tracker is not None \
            else FaceTracker(max_faces=2, refine_landmarks=True)
        self.analyzer = analyzer if analyzer is not None else FaceAnalyzer()

        self.capture: Optional[CaptureThread] = None
        self.cameras: List[CameraInfo] = []
        self.cam_index = 0
        self.frame_for_display: Optional[np.ndarray] = None
        self.paused = False
        self.mirror = True
        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[str] = None
        self._record_started_at = 0.0

        self.show_mesh = tk.BooleanVar(value=False)
        self.show_contours = tk.BooleanVar(value=True)
        self.show_iris = tk.BooleanVar(value=True)
        self.show_bbox = tk.BooleanVar(value=True)
        self.show_pose = tk.BooleanVar(value=True)
        self.show_landmarks = tk.BooleanVar(value=False)
        self.show_hud = tk.BooleanVar(value=True)
        self.show_scanlines = tk.BooleanVar(value=False)
        self.show_velocity = tk.BooleanVar(value=False)
        self.show_iris_color = tk.BooleanVar(value=False)
        self.filter_name = tk.StringVar(value="None")
        self.grid_mode = tk.StringVar(value="None")
        self.anonymize_mode = tk.StringVar(value="Off")

        self.preprocess_cfg = PreprocessConfig()
        self.smoothing = tk.BooleanVar(value=True)
        self.iris_refine = tk.BooleanVar(value=True)

        self.auto_record = tk.BooleanVar(value=False)
        self.audio_beep = tk.BooleanVar(value=False)
        self.always_on_top = tk.BooleanVar(value=False)
        self.fullscreen_video = tk.BooleanVar(value=False)
        self.compact_mode = tk.BooleanVar(value=False)
        self.crop_to_face = tk.BooleanVar(value=False)
        self.csv_logging = tk.BooleanVar(value=False)
        self.show_diff = tk.BooleanVar(value=False)
        self.auto_snap_smile = tk.BooleanVar(value=False)
        self.posture_alerts = tk.BooleanVar(value=False)
        self.drowsiness_alerts = tk.BooleanVar(value=False)
        self.eye_strain_reminder = tk.BooleanVar(value=False)
        self.fps_cap = tk.StringVar(value="30")

        self._sidebar_visible = True
        self._panel_visible = True
        self._panel_width_target = 380

        self._frame_times: Deque[float] = deque(maxlen=60)
        self._latency_ms: float = 0.0
        self._last_frame_seq = -1
        self._frame_idx = 0
        self._prev_frame_small: Optional[np.ndarray] = None
        self._face_first_seen_at: Optional[float] = None
        self._last_blink_count = 0
        self._last_metrics: Optional[FaceMetrics] = None
        self._consecutive_track_errors = 0
        self._record_size: Optional[Tuple[int, int]] = None
        self._record_fps: float = 24.0

        self.session_stats = SessionStats()
        self.replay = F.ReplayBuffer(duration_s=8.0)
        self.bookmarks: List[Bookmark] = []
        self.csv_logger: Optional[CsvLogger] = None

        self.drowsiness = DrowsinessMonitor()
        self.posture = PostureMonitor()
        self.eye_strain = EyeStrainReminder()
        self.face_heatmap = FaceHeatmap()
        self._smile_snapshot_cooldown_until = 0.0
        self._last_battery_check = 0.0
        self._last_clock_update = 0.0
        self._face_present_stable = False
        self._face_hits = 0
        self._face_misses = 0
        self._active_tab = "Telemetry"
        self._compact_geom: Optional[str] = None
        self._record_locked_resolution: Optional[Tuple[int, int]] = None

        self.toasts: Optional[ToastHost] = None

        self._build_ui()
        self.toasts = ToastHost(self.root)

        self._refresh_cameras()
        self._start_capture()

        self.tracker.set_smoothing(self.smoothing.get())

        self.root.bind("<Configure>", self._on_root_configure)
        self._bind_keys()
        self.root.after(20, self._update_loop)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_body()
        self._build_statusbar()

    @staticmethod
    def _setup_dpi_awareness() -> None:
        if not sys.platform.startswith("win"):
            return
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except (AttributeError, OSError):
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            return
        except (AttributeError, OSError):
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

    def _check_dpi_change(self) -> None:
        try:
            dpi = float(self.root.winfo_fpixels("1i"))
        except tk.TclError:
            return
        if self._last_dpi is None:
            self._last_dpi = dpi
            return
        if abs(dpi - self._last_dpi) < 4.0:
            return
        ratio = dpi / 96.0
        new_scale = max(0.9, min(2.0, ratio * 0.72))
        try:
            ctk.set_widget_scaling(new_scale)
            ctk.set_window_scaling(new_scale)
        except Exception:
            pass
        self._widget_scale = new_scale
        self._last_dpi = dpi
        try:
            geom = self.root.geometry()
            w, h = (int(p) for p in geom.split("+")[0].split("x"))
            if w < 800 or h < 500:
                self.root.geometry("1280x800")
        except Exception:
            pass

    @staticmethod
    def _find_asset(name: str) -> Optional[str]:
        candidates: List[Path] = []
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / "assets" / name)  # type: ignore[attr-defined]
        here = Path(__file__).resolve().parent
        candidates.append(here.parent / "assets" / name)
        candidates.append(here / "assets" / name)
        for p in candidates:
            try:
                if p.exists():
                    return str(p)
            except OSError:
                continue
        return None

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self.root, fg_color=C.UI_BG, corner_radius=0,
                              height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        sep = ctk.CTkFrame(self.root, fg_color=C.UI_BORDER, height=1)
        sep.grid(row=0, column=0, sticky="sew")

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=18, pady=10)
        BrandLogo(left, size=28,
                  image_path=self._brand_png_path).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(left, text=self.APP_TITLE,
                     text_color=C.UI_TEXT,
                     font=ctk.CTkFont(family=C.FONT_SANS, size=14, weight="bold"),
                     ).pack(side="left")
        ctk.CTkLabel(left, text=f"   {self.session_id}",
                     text_color=C.UI_TEXT_DIM,
                     font=ctk.CTkFont(family=C.FONT_MONO, size=11),
                     ).pack(side="left")
        ctk.CTkLabel(left, text=f"   v{self.APP_VERSION}",
                     text_color=C.UI_TEXT_FAINT,
                     font=ctk.CTkFont(family=C.FONT_MONO, size=10),
                     ).pack(side="left")

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.grid(row=0, column=2, sticky="e", padx=18, pady=10)

        self.pause_btn = IconButton(right, text="Pause", shortcut="Space",
                                    command=self._toggle_pause, width=110,
                                    tooltip="Pause / resume the live feed")
        self.pause_btn.pack(side="left", padx=3)
        self.snap_btn = IconButton(right, text="Snapshot", shortcut="S",
                                   command=self._snapshot, width=120,
                                   tooltip="Save current frame")
        self.snap_btn.pack(side="left", padx=3)
        self.rec_btn = IconButton(right, text="Record", shortcut="R",
                                  command=self._toggle_recording, width=110,
                                  tooltip="Start / stop recording")
        self.rec_btn.pack(side="left", padx=3)
        self.replay_btn = IconButton(right, text="Save replay", shortcut="B",
                                     command=self._save_replay, width=130,
                                     tooltip="Save the last 8 seconds buffered in memory")
        self.replay_btn.pack(side="left", padx=3)
        self.bookmark_btn = IconButton(right, text="Bookmark", shortcut="M",
                                       command=self._add_bookmark, width=120,
                                       tooltip="Tag this moment + save metrics snapshot")
        self.bookmark_btn.pack(side="left", padx=3)
        self.panel_btn = IconButton(right, text="Panel",
                                    command=self._toggle_panel, width=80,
                                    tooltip="Show / hide the right side panel")
        self.panel_btn.pack(side="left", padx=(12, 3))

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self.root, fg_color=C.UI_BG, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, weight=0)
        self.body = body
        self._build_sidebar(body)
        self._build_center(body)
        self._build_right_panel(body)

    def _build_sidebar(self, parent) -> None:
        side = ctk.CTkScrollableFrame(
            parent, fg_color=C.UI_BG, corner_radius=0, width=240,
            scrollbar_button_color=C.UI_SURFACE_2,
            scrollbar_button_hover_color=C.UI_BORDER_HI,
        )
        side.grid(row=0, column=0, sticky="ns", padx=(14, 0), pady=14)
        self.sidebar = side

        def header(text: str, top: int = 18) -> None:
            ctk.CTkLabel(side, text=text,
                         text_color=C.UI_TEXT_DIM,
                         font=ctk.CTkFont(family=C.FONT_SANS, size=10, weight="bold"),
                         ).pack(anchor="w", padx=4, pady=(top, 6))

        header("CAMERA", top=4)
        self.camera_menu = ctk.CTkOptionMenu(
            side, values=["No camera"], command=self._on_camera_picked,
            fg_color=C.UI_SURFACE, button_color=C.UI_SURFACE_2,
            button_hover_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
            dropdown_fg_color=C.UI_SURFACE_2, dropdown_text_color=C.UI_TEXT,
            dropdown_hover_color=C.UI_SURFACE_3, anchor="w",
            font=ctk.CTkFont(family=C.FONT_SANS, size=11),
            corner_radius=4, height=30,
        )
        self.camera_menu.pack(fill="x")
        cam_row = ctk.CTkFrame(side, fg_color="transparent")
        cam_row.pack(fill="x", pady=(6, 0))
        cam_row.grid_columnconfigure(0, weight=1)
        cam_row.grid_columnconfigure(1, weight=1)
        IconButton(cam_row, text="Refresh",
                   command=self._refresh_cameras).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        IconButton(cam_row, text="Mirror",
                   command=self._toggle_mirror).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        header("OVERLAYS")
        for label, var in [
            ("Contours",          self.show_contours),
            ("Iris tracking",     self.show_iris),
            ("Bounding box",      self.show_bbox),
            ("Head-pose axes",    self.show_pose),
            ("Full mesh",         self.show_mesh),
            ("All landmarks",     self.show_landmarks),
            ("On-screen HUD",     self.show_hud),
            ("Velocity vector",   self.show_velocity),
            ("Iris color sample", self.show_iris_color),
            ("CRT scan-lines",    self.show_scanlines),
        ]:
            ToggleRow(side, label, var).pack(fill="x", pady=1)

        header("GRID OVERLAY")
        ctk.CTkOptionMenu(side, values=C.GRID_MODES, variable=self.grid_mode,
                          fg_color=C.UI_SURFACE, button_color=C.UI_SURFACE_2,
                          button_hover_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
                          dropdown_fg_color=C.UI_SURFACE_2, dropdown_text_color=C.UI_TEXT,
                          dropdown_hover_color=C.UI_SURFACE_3, anchor="w",
                          font=ctk.CTkFont(family=C.FONT_SANS, size=11),
                          corner_radius=4, height=30,
                          ).pack(fill="x", pady=(0, 4))

        header("ANONYMIZE")
        ctk.CTkOptionMenu(side, values=C.ANONYMIZE_MODES,
                          variable=self.anonymize_mode,
                          fg_color=C.UI_SURFACE, button_color=C.UI_SURFACE_2,
                          button_hover_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
                          dropdown_fg_color=C.UI_SURFACE_2, dropdown_text_color=C.UI_TEXT,
                          dropdown_hover_color=C.UI_SURFACE_3, anchor="w",
                          font=ctk.CTkFont(family=C.FONT_SANS, size=11),
                          corner_radius=4, height=30,
                          ).pack(fill="x", pady=(0, 4))

        header("DISPLAY FILTER")
        ctk.CTkOptionMenu(side, values=C.FILTERS, variable=self.filter_name,
                          fg_color=C.UI_SURFACE, button_color=C.UI_SURFACE_2,
                          button_hover_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
                          dropdown_fg_color=C.UI_SURFACE_2, dropdown_text_color=C.UI_TEXT,
                          dropdown_hover_color=C.UI_SURFACE_3, anchor="w",
                          font=ctk.CTkFont(family=C.FONT_SANS, size=11),
                          corner_radius=4, height=30,
                          ).pack(fill="x", pady=(0, 4))

        header("IMAGE ENHANCEMENT")
        self.clahe_var = tk.BooleanVar(value=self.preprocess_cfg.clahe)
        self.denoise_var = tk.BooleanVar(value=self.preprocess_cfg.denoise)
        self.sharpen_var = tk.BooleanVar(value=self.preprocess_cfg.sharpen)
        self.auto_gamma_var = tk.BooleanVar(value=self.preprocess_cfg.auto_gamma)
        self.upscale_var = tk.BooleanVar(value=self.preprocess_cfg.upscale_small)

        def _on_pp(_):
            self.preprocess_cfg.clahe = bool(self.clahe_var.get())
            self.preprocess_cfg.denoise = bool(self.denoise_var.get())
            self.preprocess_cfg.sharpen = bool(self.sharpen_var.get())
            self.preprocess_cfg.auto_gamma = bool(self.auto_gamma_var.get())
            self.preprocess_cfg.upscale_small = bool(self.upscale_var.get())

        for label, var, tip in [
            ("Adaptive contrast (CLAHE)", self.clahe_var,
             "Equalizes lighting on the L channel. Huge for backlit faces."),
            ("Denoise",       self.denoise_var,
             "Bilateral filter. Reduces grain on cheap or low-light webcams."),
            ("Sharpen",       self.sharpen_var,
             "Unsharp mask. Helps soft / out-of-focus cameras."),
            ("Auto gamma",    self.auto_gamma_var,
             "Pulls under/over-exposed frames toward middle gray."),
            ("Upscale small inputs", self.upscale_var,
             "Bilinear upscale to 720p before tracking."),
        ]:
            row = ToggleRow(side, label, var, on_change=_on_pp)
            row.pack(fill="x", pady=1)
            Tooltip(row, tip)

        def _on_smooth(_):
            self.tracker.set_smoothing(bool(self.smoothing.get()))

        ToggleRow(side, "Landmark smoothing (One-Euro)",
                  self.smoothing, on_change=_on_smooth).pack(fill="x", pady=(8, 1))
        ToggleRow(side, "Sub-pixel iris refinement",
                  self.iris_refine).pack(fill="x", pady=1)

        header("SESSION")
        session_rows = [
            ("Auto-record on face",   self.auto_record, None,
             "Begin recording automatically when a face is detected."),
            ("CSV metrics log",       self.csv_logging, self._on_csv_logging,
             "Append a metrics row to ~/FaceTrackerSnapshots/metrics.csv every second."),
            ("Audio beep on blink",   self.audio_beep, None,
             "Short system beep on every detected blink (Windows only)."),
            ("Always on top",         self.always_on_top, self._on_always_on_top, ""),
            ("Fullscreen video",      self.fullscreen_video, self._on_fullscreen_video, ""),
            ("Compact mode",          self.compact_mode, self._on_compact_mode, ""),
            ("Crop to face",          self.crop_to_face, None, ""),
            ("Frame difference",      self.show_diff, None,
             "Show pixel-level motion since the previous frame."),
        ]
        for label, var, handler, tip in session_rows:
            row = ToggleRow(side, label, var, on_change=handler)
            row.pack(fill="x", pady=1)
            if tip:
                Tooltip(row, tip)

        header("ASSISTANT")
        assist_rows = [
            ("Drowsiness alerts",   self.drowsiness_alerts, None,
             "Warn you (toast + beep) when blink rate + yawns suggest you're tired."),
            ("Posture alerts",      self.posture_alerts, None,
             "Warn when you're too close, too far, or your head drops."),
            ("20-20-20 reminder",   self.eye_strain_reminder, self._on_eye_strain,
             "Every 20 minutes, remind you to look 20 feet away for 20 seconds."),
            ("Auto-snap on smile",  self.auto_snap_smile, None,
             "Save a snapshot the moment you smile big (60% cooldown)."),
        ]
        for label, var, handler, tip in assist_rows:
            row = ToggleRow(side, label, var, on_change=handler)
            row.pack(fill="x", pady=1)
            if tip:
                Tooltip(row, tip)

        header("FPS CAP")
        ctk.CTkOptionMenu(side, values=["15", "20", "30", "45", "60", "uncapped"],
                          variable=self.fps_cap,
                          fg_color=C.UI_SURFACE, button_color=C.UI_SURFACE_2,
                          button_hover_color=C.UI_SURFACE_3, text_color=C.UI_TEXT,
                          dropdown_fg_color=C.UI_SURFACE_2, dropdown_text_color=C.UI_TEXT,
                          dropdown_hover_color=C.UI_SURFACE_3, anchor="w",
                          font=ctk.CTkFont(family=C.FONT_SANS, size=11),
                          corner_radius=4, height=30,
                          ).pack(fill="x", pady=(0, 4))

        header("TOOLS")
        IconButton(side, text="Calibrate baselines",
                   command=self._calibrate).pack(fill="x", pady=2)
        IconButton(side, text="Reset blink counter",
                   command=self._reset_blinks).pack(fill="x", pady=2)
        IconButton(side, text="Export metrics -> JSON",
                   command=self._export_json).pack(fill="x", pady=2)
        IconButton(side, text="Save profile...",
                   command=self._save_profile).pack(fill="x", pady=2)
        IconButton(side, text="Load profile...",
                   command=self._load_profile).pack(fill="x", pady=2)
        IconButton(side, text="Open snapshots folder",
                   command=self._open_snapshots).pack(fill="x", pady=2)

    def _build_center(self, parent) -> None:
        center = ctk.CTkFrame(parent, fg_color=C.UI_BG, corner_radius=0)
        center.grid(row=0, column=1, sticky="nsew", padx=16, pady=14)
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self.center = center

        self.video_frame = ctk.CTkFrame(
            center, fg_color=C.UI_SURFACE, corner_radius=4,
            border_width=1, border_color=C.UI_BORDER,
        )
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_label = tk.Label(
            self.video_frame, bg=C.UI_SURFACE, bd=0, highlightthickness=0,
            text="ACQUIRING FEED...", fg=C.UI_TEXT_DIM,
            font=(C.FONT_MONO, 13),
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        pills = ctk.CTkFrame(center, fg_color="transparent")
        pills.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.pill_face = StatPill(pills, "FACE: NONE")
        self.pill_face.pack(side="left", padx=(0, 6))
        self.pill_lock = StatPill(pills, "LOCK: STANDBY")
        self.pill_lock.pack(side="left", padx=6)
        self.pill_blink = StatPill(pills, "BLINK:  - ")
        self.pill_blink.pack(side="left", padx=6)
        self.pill_smile = StatPill(pills, "SMILE:  - ")
        self.pill_smile.pack(side="left", padx=6)
        self.pill_talk = StatPill(pills, "VOX: idle")
        self.pill_talk.pack(side="left", padx=6)
        self.pill_yawn = StatPill(pills, "YAWN:  - ")
        self.pill_yawn.pack(side="left", padx=6)

    def _build_right_panel(self, parent) -> None:
        panel = ctk.CTkFrame(parent, fg_color=C.UI_BG, corner_radius=0,
                             width=self._panel_width_target)
        panel.grid(row=0, column=2, sticky="ns", padx=(0, 14), pady=14)
        panel.grid_propagate(False)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)
        self.right_panel = panel

        self.tab_bar = TabBar(
            panel,
            tabs=["Telemetry", "Morphology", "Expression", "Diagnostics"],
            on_change=self._on_tab,
        )
        self.tab_bar.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 6))

        content = ctk.CTkFrame(panel, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.tabs: Dict[str, ctk.CTkScrollableFrame] = {}
        for name in ("Telemetry", "Morphology", "Expression", "Diagnostics"):
            f = ctk.CTkScrollableFrame(
                content, fg_color=C.UI_BG, corner_radius=0,
                scrollbar_button_color=C.UI_SURFACE_2,
                scrollbar_button_hover_color=C.UI_BORDER_HI,
            )
            f.grid(row=0, column=0, sticky="nsew")
            f.grid_columnconfigure(0, weight=1)
            f.grid_columnconfigure(1, weight=1)
            self.tabs[name] = f

        self._build_telemetry_tab(self.tabs["Telemetry"])
        self._build_morphology_tab(self.tabs["Morphology"])
        self._build_expression_tab(self.tabs["Expression"])
        self._build_diagnostics_tab(self.tabs["Diagnostics"])
        self._show_tab("Telemetry")

    def _on_tab(self, name: str) -> None:
        self._active_tab = name
        self._show_tab(name)

    def _show_tab(self, name: str) -> None:
        for k, f in self.tabs.items():
            if k == name:
                f.tkraise()

    def _build_telemetry_tab(self, parent) -> None:
        self.cnt_fps = StatTile(parent, "FRAMES / SEC", accent=True)
        self.cnt_fps.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.cnt_faces = StatTile(parent, "FACES")
        self.cnt_faces.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self.cnt_blinks = StatTile(parent, "BLINKS", accent=True)
        self.cnt_blinks.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.cnt_blinkrate = StatTile(parent, "BLINKS / MIN", unit=" /m")
        self.cnt_blinkrate.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        self.session_table = DataTable(parent, "session", [
            "Session ID", "Frame", "Detect rate", "Runtime",
            "Attention", "Look-aways", "Yawns", "Recording",
        ])
        self.session_table.grid(row=2, column=0, columnspan=2,
                                sticky="ew", padx=4, pady=4)

        dist_card = Card(parent, "Distance")
        dist_card.grid(row=3, column=0, columnspan=2, sticky="ew",
                       padx=4, pady=4)
        self.bar_dist = MetricBar(
            dist_card, "Range to camera",
            min_val=0.0, max_val=120.0, fmt="{:.0f}", unit=" cm",
        )
        self.bar_dist.pack(fill="x", padx=14, pady=(2, 12))

        ear_card = Card(parent, "Eye openness (EAR)")
        ear_card.grid(row=4, column=0, columnspan=2, sticky="ew",
                      padx=4, pady=4)
        self.spark_ear = Sparkline(ear_card, "", height=52,
                                   y_range=(0.0, 0.45), redraw_every=2)
        self.spark_ear.pack(fill="x", padx=14, pady=(2, 14))

        head_card = Card(parent, "Head speed (px/s)")
        head_card.grid(row=5, column=0, columnspan=2, sticky="ew",
                       padx=4, pady=4)
        self.spark_speed = Sparkline(head_card, "", height=44,
                                     y_range=(0.0, 600.0), redraw_every=2)
        self.spark_speed.pack(fill="x", padx=14, pady=(2, 12))

        self.badge_drowsy = StatusBadge(parent, "DROWSINESS")
        self.badge_drowsy.grid(row=6, column=0, sticky="ew", padx=4, pady=4)
        self.badge_posture = StatusBadge(parent, "POSTURE")
        self.badge_posture.grid(row=6, column=1, sticky="ew", padx=4, pady=4)

        self.heatmap_widget = HeatmapWidget(parent)
        self.heatmap_widget.grid(row=7, column=0, columnspan=2,
                                 sticky="ew", padx=4, pady=4)

        self.snapshot_strip = SnapshotStrip(parent)
        self.snapshot_strip.grid(row=8, column=0, columnspan=2,
                                 sticky="ew", padx=4, pady=4)

    def _build_morphology_tab(self, parent) -> None:
        shape_card = Card(parent, "Face shape")
        shape_card.grid(row=0, column=0, columnspan=2, sticky="ew",
                        padx=4, pady=4)
        self.shape_value = ctk.CTkLabel(
            shape_card, text=" - ",
            text_color=C.UI_ACCENT_HI,
            font=ctk.CTkFont(family=C.FONT_SANS, size=20, weight="bold"),
        )
        self.shape_value.pack(anchor="w", padx=14, pady=(2, 12))

        self.shape_probs = ShapeProbabilityList(parent, C.FACE_SHAPES)
        self.shape_probs.grid(row=1, column=0, columnspan=2,
                              sticky="ew", padx=4, pady=4)

        self.pose_dial = PoseDial(parent)
        self.pose_dial.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        self.gaze_radial = RadialGaze(parent)
        self.gaze_radial.grid(row=2, column=1, sticky="ew", padx=4, pady=4)

        sym_card = Card(parent, "Symmetry & area")
        sym_card.grid(row=3, column=0, columnspan=2, sticky="ew",
                      padx=4, pady=4)
        self.bar_sym = MetricBar(sym_card, "Symmetry",
                                 min_val=0.0, max_val=1.0, fmt="{:.0%}")
        self.bar_sym.pack(fill="x", padx=14, pady=(2, 4))
        self.bar_area = MetricBar(sym_card, "Face area % of frame",
                                  min_val=0.0, max_val=60.0,
                                  fmt="{:.1f}", unit="%")
        self.bar_area.pack(fill="x", padx=14, pady=(2, 12))

        misc = DataTable(parent, "biometric", [
            "Glasses", "Iris L", "Iris R", "Stability",
        ])
        misc.grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self.misc_table = misc

    def _build_expression_tab(self, parent) -> None:
        em_card = Card(parent, "Inferred emotion")
        em_card.grid(row=0, column=0, columnspan=2, sticky="ew",
                     padx=4, pady=4)
        self.emotion_value = ctk.CTkLabel(
            em_card, text=" - ",
            text_color=C.UI_ACCENT_HI,
            font=ctk.CTkFont(family=C.FONT_SANS, size=20, weight="bold"),
        )
        self.emotion_value.pack(anchor="w", padx=14, pady=(2, 12))

        smile_card = Card(parent, "Smile")
        smile_card.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.bar_smile = MetricBar(smile_card, "Intensity", fmt="{:.0%}")
        self.bar_smile.pack(fill="x", padx=14, pady=(2, 12))

        mouth_card = Card(parent, "Mouth")
        mouth_card.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        self.bar_mouth = MetricBar(mouth_card, "Open", fmt="{:.0%}")
        self.bar_mouth.pack(fill="x", padx=14, pady=(2, 12))

        brow_card = Card(parent, "Eyebrows")
        brow_card.grid(row=2, column=0, columnspan=2, sticky="ew",
                       padx=4, pady=4)
        self.bar_brows = MetricBar(brow_card, "Raise (signed)",
                                   min_val=-1.0, max_val=1.0,
                                   fmt="{:+.2f}", signed=True)
        self.bar_brows.pack(fill="x", padx=14, pady=(2, 12))

        events_card = Card(parent, "Event log")
        events_card.grid(row=3, column=0, columnspan=2, sticky="ew",
                         padx=4, pady=4)
        self.bookmarks_widget = BookmarkList(events_card, height=180)
        self.bookmarks_widget.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    def _build_diagnostics_tab(self, parent) -> None:
        self.perf_graph = PerfGraph(parent)
        self.perf_graph.grid(row=0, column=0, columnspan=2,
                             sticky="ew", padx=4, pady=4)

        self.histogram = HistogramPanel(parent)
        self.histogram.grid(row=1, column=0, columnspan=2,
                            sticky="ew", padx=4, pady=4)

        info = DataTable(parent, "system", [
            "Backend", "Resolution", "Detected res", "Latency",
            "Landmarks", "Smoothing", "Iris refine", "Auto-record",
            "CSV log", "Replay buffer",
        ])
        info.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self.diag_table = info

        shortcuts = Card(parent, "Keyboard shortcuts")
        shortcuts.grid(row=3, column=0, columnspan=2,
                       sticky="ew", padx=4, pady=4)
        ks_text = (
            "  Space          pause / resume the live feed\n"
            "  S              save a PNG snapshot to ~/FaceTrackerSnapshots\n"
            "  R              start / stop MP4 recording\n"
            "  B              save the last 8s replay buffer to MP4\n"
            "  M              drop a bookmark + thumbnail at this moment\n"
            "  F11            toggle the right side panel\n"
            "  Q              quit"
        )
        ctk.CTkLabel(
            shortcuts, text=ks_text, anchor="w", justify="left",
            text_color=C.UI_TEXT_MID,
            font=ctk.CTkFont(family=C.FONT_MONO, size=11),
        ).pack(anchor="w", padx=14, pady=(2, 10))

        about = Card(parent, "About")
        about.grid(row=4, column=0, columnspan=2,
                   sticky="ew", padx=4, pady=4)
        about_text = (
            "  Face Tracker Pro - ops build\n"
            "  478-point landmark tracking - MediaPipe Tasks - OpenCV - Tkinter\n"
            "  Snapshots -> ~/FaceTrackerSnapshots\n"
            "  Profiles  -> ~/.facetrack/profiles\n"
            "  Built by frnchy - github.com/frnchy - 2026"
        )
        ctk.CTkLabel(
            about, text=about_text, anchor="w", justify="left",
            text_color=C.UI_TEXT_DIM,
            font=ctk.CTkFont(family=C.FONT_MONO, size=10),
        ).pack(anchor="w", padx=14, pady=(2, 10))

    def _build_statusbar(self) -> None:
        sep = ctk.CTkFrame(self.root, fg_color=C.UI_BORDER, height=1)
        sep.grid(row=2, column=0, sticky="new")
        self.status = StatusStrip(self.root)
        self.status.grid(row=3, column=0, sticky="ew")
        self.status.add_left("status", "STANDBY")
        self.status.add_left("backend", "backend:  - ")
        self.status.add_left("model", "model: face_landmarker_v2")
        self.status.add_left("sig", "built by frnchy - 2026")
        self.status.add_right("latency", "0 ms")
        self.status.add_right("res", "0x0")
        self.status.add_right("landmarks", "0 pts")
        self.status.add_right("rec", "")
        self.status.add_right("battery", "")
        self.status.add_right("clock", time.strftime("%H:%M:%S"))
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)

    def _refresh_cameras(self) -> None:
        self.cameras = list_cameras(max_probe=6)
        if not self.cameras:
            self.camera_menu.configure(values=["No camera detected"])
            self.camera_menu.set("No camera detected")
            return
        labels = [c.label for c in self.cameras]
        self.camera_menu.configure(values=labels)
        current_label = None
        for c in self.cameras:
            if c.index == self.cam_index:
                current_label = c.label
                break
        if current_label is None:
            current_label = labels[0]
            self.cam_index = self.cameras[0].index
        self.camera_menu.set(current_label)
        if self.toasts is not None:
            self.toasts.show(f"Found {len(self.cameras)} camera"
                             f"{'s' if len(self.cameras) != 1 else ''}",
                             kind="info")

    def _on_camera_picked(self, label: str) -> None:
        target: Optional[CameraInfo] = None
        for c in self.cameras:
            if c.label == label:
                target = c
                break
        if target is None or target.index == self.cam_index:
            return
        self.cam_index = target.index
        self._start_capture()

    def _start_capture(self) -> None:
        if self.capture:
            self.capture.stop()
        try:
            w, h = best_resolution_for(self.cam_index)
        except Exception:
            w, h = 1280, 720
        self.capture = CaptureThread(self.cam_index, resolution=(w, h))
        if not self.capture.open():
            self.status.set("status", f"FAILED to open camera {self.cam_index}")
            if self.toasts is not None:
                self.toasts.show(f"Could not open camera #{self.cam_index}", kind="error")
            return
        self.capture.start()
        cam_name = next(
            (c.name for c in self.cameras if c.index == self.cam_index),
            f"Camera {self.cam_index}",
        )
        aw, ah = self.capture.actual_resolution
        self.status.set("status", f"FEED ACQUIRED - {cam_name}")
        self.status.set("res", f"{aw}x{ah}")
        if self.toasts is not None:
            self.toasts.show(f"Connected to {cam_name} ({aw}x{ah})", kind="success")

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.configure(text="Resume" if self.paused else "Pause")
        self.pause_btn.set_active(self.paused)

    def _toggle_mirror(self) -> None:
        self.mirror = not self.mirror

    def _toggle_panel(self) -> None:
        self._panel_visible = not self._panel_visible
        if self._panel_visible:
            self.right_panel.grid()
            self.panel_btn.configure(text="Panel")
        else:
            self.right_panel.grid_remove()
            self.panel_btn.configure(text="Panel ^")

    def _on_always_on_top(self, on: bool) -> None:
        try:
            self.root.attributes("-topmost", on)
        except tk.TclError:
            pass

    def _on_fullscreen_video(self, on: bool) -> None:
        if on:
            if self._panel_visible:
                self._toggle_panel()
            if self._sidebar_visible:
                self.sidebar.grid_remove()
                self._sidebar_visible = False
        else:
            if not self._sidebar_visible:
                self.sidebar.grid()
                self._sidebar_visible = True

    def _on_compact_mode(self, on: bool) -> None:
        if on:
            self._compact_geom = self.root.geometry()
            self.root.geometry("780x560")
        elif self._compact_geom:
            self.root.geometry(self._compact_geom)
            self._compact_geom = None
        else:
            self.root.geometry("1600x950")

    def _on_csv_logging(self, on: bool) -> None:
        if on and self.csv_logger is None:
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            self.csv_logger = CsvLogger(SNAPSHOTS_DIR / "metrics.csv")
            if self.toasts:
                self.toasts.show("CSV logging started "
                                 "(~/FaceTrackerSnapshots/metrics.csv)",
                                 kind="info")
        elif not on and self.csv_logger is not None:
            self.csv_logger.close()
            self.csv_logger = None
            if self.toasts:
                self.toasts.show("CSV logging stopped", kind="info")

    def _on_eye_strain(self, on: bool) -> None:
        if on:
            self.eye_strain.reset()
            if self.toasts:
                self.toasts.show("20-20-20 reminder armed (next in 20 min)",
                                 kind="info")

    def _calibrate(self) -> None:
        self.analyzer.calibrate()
        if self.toasts:
            self.toasts.show("Baselines calibrated", kind="success")

    def _reset_blinks(self) -> None:
        self.analyzer.reset_blinks()
        if self.toasts:
            self.toasts.show("Blink counter reset", kind="info")

    def _open_snapshots(self) -> None:
        import subprocess
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = str(SNAPSHOTS_DIR)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            if self.toasts:
                self.toasts.show(f"Could not open folder: {e}", kind="error")

    def _snapshot(self, *, reason: str = "") -> None:
        if self.frame_for_display is None:
            return
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = "snapshot"
        if reason:
            safe = "".join(ch for ch in reason if ch.isalnum() or ch in "-_")
            prefix = f"snap_{safe}" if safe else prefix
        path = SNAPSHOTS_DIR / time.strftime(f"{prefix}_%Y%m%d_%H%M%S.png")
        cv2.imwrite(str(path), self.frame_for_display)
        try:
            self.snapshot_strip.push(self.frame_for_display)
        except Exception:
            pass
        if self.toasts:
            label = f"Snapshot - {path.name}"
            if reason:
                label = f"Snapshot ({reason}) - {path.name}"
            self.toasts.show(label, kind="success")

    def _toggle_recording(self) -> None:
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self.frame_for_display is None:
            return
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOTS_DIR / time.strftime("recording_%Y%m%d_%H%M%S.mp4")
        h, w = self.frame_for_display.shape[:2]
        measured = self._compute_fps()
        rec_fps = max(15.0, min(60.0, measured if measured > 1.0 else 24.0))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(path), fourcc, rec_fps, (w, h))
        if not self.video_writer.isOpened():
            if self.toasts:
                self.toasts.show("Could not start recorder", kind="error")
            self.video_writer = None
            return
        self.recording = True
        self._record_path = str(path)
        self._record_started_at = time.time()
        self._record_size = (w, h)
        self._record_fps = rec_fps
        self.rec_btn.configure(text="Stop")
        self.rec_btn.set_active(True)
        self.status.set("rec", f"* REC {rec_fps:.0f}fps")
        if self.toasts:
            self.toasts.show(f"Recording - {path.name} @ {rec_fps:.0f}fps",
                             kind="info")

    def _stop_recording(self) -> None:
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.rec_btn.configure(text="Record")
        self.rec_btn.set_active(False)
        self.status.set("rec", "")
        if self.toasts and self._record_path:
            self.toasts.show(
                f"Saved - {Path(self._record_path).name}", kind="success",
            )

    def _save_replay(self) -> None:
        if self.replay.seconds_buffered < 0.5:
            if self.toasts:
                self.toasts.show("Replay buffer empty", kind="warn")
            return
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOTS_DIR / time.strftime("replay_%Y%m%d_%H%M%S.mp4")
        ok = self.replay.export_mp4(str(path))
        if self.toasts:
            if ok:
                self.toasts.show(f"Replay saved - {path.name}", kind="success")
            else:
                self.toasts.show("Failed to save replay", kind="error")

    def _add_bookmark(self) -> None:
        ts = time.time()
        label = f"event #{len(self.bookmarks) + 1}"
        thumb_path: Optional[str] = None
        if self.frame_for_display is not None:
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            tp = SNAPSHOTS_DIR / time.strftime("bookmark_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(tp), self.frame_for_display)
            thumb_path = str(tp)
        bm = Bookmark(timestamp=ts, label=label,
                      metrics_snapshot={}, image_path=thumb_path)
        self.bookmarks.append(bm)
        self.bookmarks_widget.add(time.strftime("%H:%M:%S", time.localtime(ts)),
                                  label)
        if self.toasts:
            self.toasts.show(f"Bookmark - {label}", kind="info")

    def _export_json(self) -> None:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOTS_DIR / time.strftime("metrics_%Y%m%d_%H%M%S.json")
        m = getattr(self, "_last_metrics", None)
        extras = {
            "session_id": self.session_id,
            "frame": self._frame_idx,
            "stats": {
                "runtime_s": self.session_stats.runtime_s,
                "frame_count": self.session_stats.frame_count,
                "detection_rate": self.session_stats.detection_rate,
                "blink_total": self.session_stats.blink_total,
                "attention_time_s": self.session_stats.attention_time_s,
                "look_away_count": self.session_stats.look_away_count,
                "yawn_count": self.session_stats.yawn_count,
            },
            "bookmarks": [
                {"t": b.timestamp, "label": b.label, "image": b.image_path}
                for b in self.bookmarks
            ],
        }
        try:
            export_metrics_json(path, m, extras)
            if self.toasts:
                self.toasts.show(f"Exported - {path.name}", kind="success")
        except Exception as e:
            if self.toasts:
                self.toasts.show(f"Export failed: {e}", kind="error")

    def _save_profile(self) -> None:
        name = _ask_text(
            self.root, "Save profile",
            "Profile name (letters / digits / hyphen):",
            default="default",
        )
        if not name:
            return
        name = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "default"
        prof = Profile(
            name=name,
            smile_baseline=getattr(self.analyzer, "_smile_baseline", None),
            eyebrow_baseline=getattr(self.analyzer, "_eyebrow_baseline", None),
            blink_threshold=self.analyzer.blink_threshold,
            preprocess={
                "clahe": self.preprocess_cfg.clahe,
                "denoise": self.preprocess_cfg.denoise,
                "sharpen": self.preprocess_cfg.sharpen,
                "auto_gamma": self.preprocess_cfg.auto_gamma,
                "upscale_small": self.preprocess_cfg.upscale_small,
            },
            overlays={
                "contours": self.show_contours.get(),
                "iris": self.show_iris.get(),
                "bbox": self.show_bbox.get(),
                "pose": self.show_pose.get(),
                "mesh": self.show_mesh.get(),
                "landmarks": self.show_landmarks.get(),
                "hud": self.show_hud.get(),
            },
        )
        try:
            path = prof.save()
            if self.toasts:
                self.toasts.show(f"Profile saved - {path.name}", kind="success")
        except Exception as e:
            if self.toasts:
                self.toasts.show(f"Save failed: {e}", kind="error")

    def _load_profile(self) -> None:
        existing = Profile.list_all()
        if not existing:
            if self.toasts:
                self.toasts.show(
                    "No profiles saved yet. Click 'Save profile...' first.",
                    kind="warn",
                )
            return
        default_name = existing[0]
        prompt = f"Profile to load - available: {', '.join(existing[:6])}"
        name = _ask_text(self.root, "Load profile", prompt, default=default_name)
        if not name:
            return
        prof = Profile.load(name)
        if prof is None:
            if self.toasts:
                self.toasts.show(f"No profile named '{name}'", kind="warn")
            return
        if prof.preprocess:
            for k, v in prof.preprocess.items():
                if hasattr(self.preprocess_cfg, k):
                    setattr(self.preprocess_cfg, k, bool(v))
            self.clahe_var.set(self.preprocess_cfg.clahe)
            self.denoise_var.set(self.preprocess_cfg.denoise)
            self.sharpen_var.set(self.preprocess_cfg.sharpen)
            self.auto_gamma_var.set(self.preprocess_cfg.auto_gamma)
            self.upscale_var.set(self.preprocess_cfg.upscale_small)
        if prof.overlays:
            mapping = {
                "contours": self.show_contours, "iris": self.show_iris,
                "bbox": self.show_bbox, "pose": self.show_pose,
                "mesh": self.show_mesh, "landmarks": self.show_landmarks,
                "hud": self.show_hud,
            }
            for k, var in mapping.items():
                if k in prof.overlays:
                    var.set(bool(prof.overlays[k]))
        if prof.blink_threshold:
            self.analyzer.blink_threshold = prof.blink_threshold
        self.analyzer._smile_baseline = prof.smile_baseline
        self.analyzer._eyebrow_baseline = prof.eyebrow_baseline
        if self.toasts:
            self.toasts.show(f"Profile loaded - {prof.name}", kind="success")

    def _on_root_configure(self, e) -> None:
        if e.widget is not self.root:
            return
        w = e.width
        SIDEBAR_THRESHOLD = 1000
        PANEL_THRESHOLD = 1280
        if w < PANEL_THRESHOLD and self._panel_visible:
            self._panel_visible = False
            self.right_panel.grid_remove()
        if w < SIDEBAR_THRESHOLD and self._sidebar_visible:
            self._sidebar_visible = False
            self.sidebar.grid_remove()
        if (w >= SIDEBAR_THRESHOLD and not self._sidebar_visible
                and not self.fullscreen_video.get()):
            self._sidebar_visible = True
            self.sidebar.grid()

    def _update_loop(self) -> None:
        try:
            self._tick()
        except Exception:
            traceback.print_exc()
        cap = self.fps_cap.get()
        if cap == "uncapped":
            delay = 8
        else:
            try:
                delay = max(8, int(1000 / int(cap)))
            except ValueError:
                delay = 16
        self.root.after(delay, self._update_loop)

    def _tick(self) -> None:
        now_wall = time.time()
        if now_wall - self._last_clock_update >= 1.0:
            self.status.set("clock", time.strftime("%H:%M:%S"))
            self._last_clock_update = now_wall

        if self._frame_idx % 60 == 0:
            self._check_dpi_change()

        if not self.capture or not self.capture.opened:
            return
        if self.paused:
            return
        raw, seq = self.capture.read()
        if raw is None or seq == self._last_frame_seq:
            return
        self._last_frame_seq = seq
        self._frame_idx += 1
        self.session_stats.frame_count += 1

        t0 = time.time()
        frame = raw
        if self.mirror:
            frame = cv2.flip(frame, 1)

        display_frame = frame
        if self.filter_name.get() != "None":
            display_frame = apply_filter(frame, self.filter_name.get())

        proc_frame = preprocess(frame, self.preprocess_cfg)
        gray_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

        try:
            faces: List[FaceData] = self.tracker.process(proc_frame)
            self._consecutive_track_errors = 0
        except Exception as e:
            self._consecutive_track_errors += 1
            faces = []
            if self._consecutive_track_errors == 1 and self.toasts:
                self.toasts.show(f"Tracker error: {type(e).__name__}",
                                 kind="warn", duration_ms=3000)
            elif self._consecutive_track_errors == 30 and self.toasts:
                self.toasts.show(
                    "Tracker has failed for 30 frames in a row. "
                    "Try toggling preprocessing off, or restart the app.",
                    kind="error", duration_ms=5000)

        raw_present = bool(faces)
        if raw_present:
            self._face_hits = min(self._face_hits + 1, 10)
            self._face_misses = 0
            if not self._face_present_stable and self._face_hits >= 2:
                self._face_present_stable = True
        else:
            self._face_misses = min(self._face_misses + 1, 30)
            if self._face_present_stable and self._face_misses >= 6:
                self._face_present_stable = False
            if self._face_misses > 0:
                self._face_hits = max(self._face_hits - 1, 0)
        primary_metrics: Optional[FaceMetrics] = None

        if faces:
            if proc_frame.shape[:2] != display_frame.shape[:2]:
                sy = display_frame.shape[0] / proc_frame.shape[0]
                sx = display_frame.shape[1] / proc_frame.shape[1]
                for f in faces:
                    f.landmarks_px = (
                        f.landmarks_px.astype(np.float32) * np.array([sx, sy])
                    ).astype(np.int32)
                    x, y, w, h = f.bbox
                    f.bbox = (int(x * sx), int(y * sy),
                              int(w * sx), int(h * sy))
                    f.image_shape = (display_frame.shape[0], display_frame.shape[1])

            if self.iris_refine.get():
                gray_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces[0].landmarks_px = self.analyzer.refine_iris_centers(
                    gray_display, faces[0].landmarks_px,
                )
            primary_metrics = self.analyzer.analyze(faces[0])
            if self.show_iris_color.get():
                self.analyzer.sample_iris_colors(
                    display_frame, faces[0].landmarks_px, primary_metrics,
                )
            self.analyzer.detect_glasses(gray_proc, faces[0].landmarks_px, primary_metrics)
            self._last_metrics = primary_metrics
            self.session_stats.detected_frames += 1
            self.session_stats.last_face_seen_at = time.time()

            if primary_metrics.blink_count > self._last_blink_count:
                if self.audio_beep.get() and sys.platform.startswith("win"):
                    try:
                        import winsound
                        winsound.Beep(1200, 35)
                    except Exception:
                        pass
                self._last_blink_count = primary_metrics.blink_count
                self.session_stats.blink_total = primary_metrics.blink_count

            self.session_stats.yawn_count = primary_metrics.yawn_count
            self.session_stats.look_away_count = primary_metrics.look_away_count
            self.session_stats.attention_time_s = primary_metrics.attention_s

            self.drowsiness.update(
                blink_count=primary_metrics.blink_count,
                yawn_count=primary_metrics.yawn_count,
                head_speed=primary_metrics.head_speed,
                attention_s=primary_metrics.attention_s,
            )
            self.posture.update(primary_metrics.distance_cm,
                                primary_metrics.pitch)

            nose = faces[0].landmarks_px[C.NOSE_TIP]
            self.face_heatmap.push(
                float(nose[0]) / max(1, display_frame.shape[1]),
                float(nose[1]) / max(1, display_frame.shape[0]),
            )

            if (self.auto_snap_smile.get() and primary_metrics.smile > 0.6
                    and time.time() > self._smile_snapshot_cooldown_until):
                self._smile_snapshot_cooldown_until = time.time() + 3.0
                self._snapshot(reason="smile")

            if (self.drowsiness_alerts.get()
                    and self.drowsiness.should_warn()):
                if self.toasts:
                    self.toasts.show(
                        f"Drowsiness high ({self.drowsiness.label}). "
                        f"Take a 30s break.",
                        kind="warn", duration_ms=4500)
                if self.audio_beep.get() and sys.platform.startswith("win"):
                    try:
                        import winsound
                        winsound.Beep(700, 200)
                    except Exception:
                        pass

            if (self.posture_alerts.get() and self.posture.should_warn()):
                if self.toasts:
                    self.toasts.show(f"Posture - {self.posture.state} "
                                     f"({self.posture.detail})",
                                     kind="warn", duration_ms=3500)

            if self.auto_record.get() and not self.recording:
                if self._face_first_seen_at is None:
                    self._face_first_seen_at = time.time()
                elif (time.time() - self._face_first_seen_at) > 1.5:
                    self._start_recording()
        else:
            if self.auto_record.get() and self.recording:
                if (time.time() - self.session_stats.last_face_seen_at) > 3.0:
                    self._stop_recording()
                    self._face_first_seen_at = None
            if (self._face_first_seen_at is not None
                    and (time.time() - self.session_stats.last_face_seen_at) > 2.0):
                self._face_first_seen_at = None

        if self.grid_mode.get() != "None":
            F.draw_grid(display_frame, self.grid_mode.get())

        if faces and self.anonymize_mode.get() != "Off":
            eye_y = int((faces[0].landmarks_px[33, 1] + faces[0].landmarks_px[263, 1]) / 2)
            F.anonymize(display_frame, faces[0].bbox, self.anonymize_mode.get(),
                        eye_y=eye_y,
                        eye_x_left=int(faces[0].landmarks_px[127, 0]),
                        eye_x_right=int(faces[0].landmarks_px[356, 0]))

        for i, face in enumerate(faces):
            if self.show_mesh.get():
                self.tracker.draw_mesh(display_frame, face)
            if self.show_contours.get():
                self.tracker.draw_contours(display_frame, face)
            if self.show_iris.get():
                self.tracker.draw_irises(display_frame, face)
            if self.show_landmarks.get():
                for (x, y) in face.landmarks_px:
                    cv2.circle(display_frame, (int(x), int(y)), 1,
                               (180, 180, 180), -1)
            if self.show_bbox.get():
                hot = (i == 0 and primary_metrics and primary_metrics.head_speed < 60)
                label = f"T-{i + 1:03d}"
                if i == 0 and primary_metrics:
                    label += f" // RNG {primary_metrics.distance_cm:.0f}cm"
                F.draw_tactical_lock(display_frame, face.bbox, label,
                                     color=C.COLOR_TACTICAL, hot=hot)
                nose = face.landmarks_px[C.NOSE_TIP]
                F.draw_target_reticle(display_frame,
                                      (int(nose[0]), int(nose[1])),
                                      color=C.COLOR_TACTICAL)
            if self.show_velocity.get() and i == 0 and primary_metrics:
                nose = face.landmarks_px[C.NOSE_TIP]
                F.draw_velocity_arrow(display_frame,
                                      (int(nose[0]), int(nose[1])),
                                      primary_metrics.velocity_px_per_s,
                                      color=C.COLOR_TACTICAL)
            if self.show_iris_color.get() and i == 0 and primary_metrics:
                ix, iy = face.landmarks_px[C.LEFT_IRIS_CENTER]
                rL = primary_metrics.iris_color_left
                cv2.rectangle(display_frame, (int(ix) + 8, int(iy) - 4),
                              (int(ix) + 22, int(iy) + 10),
                              (rL[2], rL[1], rL[0]), -1)
            if self.show_pose.get() and i == 0 and primary_metrics:
                self.analyzer.draw_pose_axes(display_frame, face, primary_metrics)

        if self.show_diff.get():
            small_now = cv2.resize(frame, (160, 90))
            diff = F.frame_difference(self._prev_frame_small, small_now)
            self._prev_frame_small = small_now
            if diff is not None:
                d_up = cv2.resize(diff, (display_frame.shape[1] // 4,
                                         display_frame.shape[0] // 4))
                dh, dw = d_up.shape[:2]
                display_frame[8:8 + dh, display_frame.shape[1] - dw - 8:display_frame.shape[1] - 8] = d_up

        if self.show_scanlines.get():
            display_frame = F.apply_scanlines(display_frame, 0.08)

        if self.crop_to_face.get() and faces:
            display_frame = F.crop_to_face(display_frame, faces[0].bbox)

        if self.show_hud.get():
            self._draw_hud(display_frame, primary_metrics, len(faces),
                           fps=self._compute_fps())

        self._frame_times.append(time.time())
        fps = self._compute_fps()
        self._latency_ms = (time.time() - t0) * 1000.0
        self.session_stats.peak_fps = max(self.session_stats.peak_fps, fps)

        if self.recording and self.video_writer is not None and self._record_size:
            try:
                rw, rh = self._record_size
                if display_frame.shape[1] == rw and display_frame.shape[0] == rh:
                    self.video_writer.write(display_frame)
                else:
                    resized = cv2.resize(display_frame, (rw, rh),
                                         interpolation=cv2.INTER_LINEAR)
                    self.video_writer.write(resized)
            except Exception:
                pass

        self.replay.push(frame)
        if self.csv_logger and primary_metrics is not None:
            extras = {
                "stability": primary_metrics.stability,
                "is_talking": primary_metrics.is_talking,
                "is_yawning": primary_metrics.is_yawning,
                "attention_s": primary_metrics.attention_s,
                "looking_away": primary_metrics.looking_away,
            }
            self.csv_logger.maybe_write(self._frame_idx, fps,
                                        face_present=True,
                                        metrics=primary_metrics, extras=extras)

        if self._frame_idx % 6 == 0:
            try:
                hist = F.rgb_histogram(frame, bins=64)
                self.histogram.set_hist(hist)
            except Exception:
                pass

        if self.eye_strain_reminder.get() and self.eye_strain.should_fire():
            if self.toasts:
                self.toasts.show(
                    "20-20-20 break: look 20 feet away for 20 seconds.",
                    kind="info", duration_ms=20000)

        if self._frame_idx % 30 == 0:
            self.heatmap_widget.set_grid(self.face_heatmap.grid,
                                         self.face_heatmap.coverage_pct())

        if (time.time() - self._last_battery_check) > 15.0:
            self._last_battery_check = time.time()
            pct, charging = battery_status()
            if pct is not None:
                glyph = "[chg]" if charging else "[bat]"
                self.status.set("battery", f"{glyph} {pct}%")

        self.frame_for_display = display_frame
        self._render_video(display_frame)

        self.cnt_fps.set_value(fps)
        self.cnt_faces.set_value(len(faces))
        if primary_metrics is not None:
            self._update_widgets(primary_metrics)
        elif self._face_present_stable and self._last_metrics is not None:
            self._update_widgets(self._last_metrics)
        else:
            self._update_widgets_empty()

        self._update_status_bar(fps, faces)
        if self._frame_idx % 4 == 0:
            self.perf_graph.push(fps, self._latency_ms)

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        dt = self._frame_times[-1] - self._frame_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / dt

    def _draw_hud(self, frame: np.ndarray, m: Optional[FaceMetrics], n_faces: int,
                  *, fps: float = 0.0) -> None:
        h, w = frame.shape[:2]
        x, y = 12, 12
        bw, bh = 234, 96
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (4, 4, 4), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), C.COLOR_TACTICAL, 1)
        cv2.rectangle(frame, (x, y), (x + bw, y + 16), C.COLOR_TACTICAL, -1)
        cv2.putText(frame, "OPERATIONS // TELEMETRY", (x + 8, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (4, 4, 4), 1, cv2.LINE_AA)
        runtime = self.session_stats.runtime_s
        rt = f"{int(runtime // 3600):02d}:{int((runtime % 3600) // 60):02d}:{int(runtime % 60):02d}"
        cv2.putText(frame, f"FRAME  {self._frame_idx:>6d}", (x + 8, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS    {fps:>6.1f}", (x + 8, y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"T-RUN  {rt}", (x + 8, y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FACES  {n_faces}", (x + 8, y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
        if m is not None:
            cv2.putText(frame, f"BLINKS {m.blink_count}", (x + 130, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, f"RNG    {m.distance_cm:>4.0f}cm", (x + 130, y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, f"YAW    {m.yaw:+5.1f}", (x + 130, y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, f"PITCH  {m.pitch:+5.1f}", (x + 130, y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.COLOR_TEXT, 1, cv2.LINE_AA)

        if self.recording:
            elapsed = int(time.time() - self._record_started_at)
            rt = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            cv2.circle(frame, (w - 26, 22), 7,
                       (0, 0, 220) if (int(time.time() * 2) % 2) else (0, 0, 80), -1)
            cv2.putText(frame, f"REC  {rt}", (w - 100, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 230), 2, cv2.LINE_AA)

        cv2.putText(frame, "CLASSIFIED // INTERNAL USE",
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%Y-%m-%d  %H:%M:%S"),
                    (w - 200, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    C.COLOR_TACTICAL, 1, cv2.LINE_AA)

    def _render_video(self, frame: np.ndarray) -> None:
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()
        if target_w <= 1 or target_h <= 1:
            return
        h, w = frame.shape[:2]
        cache_key = (target_w, target_h, w, h)
        if cache_key != getattr(self, "_resize_cache_key", None):
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            self._resize_cache_key = cache_key
            self._resize_cache_dims = (new_w, new_h)
        new_w, new_h = self._resize_cache_dims
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def _update_widgets(self, m: FaceMetrics) -> None:
        tab = self._active_tab
        if tab == "Telemetry":
            self.cnt_blinks.set_value(m.blink_count)
            self.cnt_blinkrate.set_value(m.blink_rate_per_min)
            self.bar_dist.set_value(m.distance_cm)
            avg_ear = (m.ear_left + m.ear_right) / 2.0
            self.spark_ear.push(avg_ear)
            self.spark_speed.push(m.head_speed)
        elif tab == "Morphology":
            self.bar_sym.set_value(m.symmetry)
            self.bar_area.set_value(m.face_area_pct)
            self.pose_dial.set_values(m.pitch, m.yaw, m.roll)
            self.gaze_radial.set_gaze(m.gaze_x, m.gaze_y, m.gaze_label)
            self.shape_probs.set_scores(m.shape_scores)
            self.shape_value.configure(
                text=f"{m.face_shape}  -  {m.face_shape_conf * 100:.0f}%")
        elif tab == "Expression":
            self.bar_smile.set_value(m.smile)
            self.bar_mouth.set_value(m.mouth_open)
            self.bar_brows.set_value(m.eyebrow_raise)
            self.emotion_value.configure(
                text=f"{m.emotion}  -  {m.emotion_conf * 100:.0f}%")

        self.pill_face.set_state(True, text="FACE: ACQUIRED")
        lock = m.stability > 0.6 and m.face_shape_conf > 0.25
        self.pill_lock.set_state(lock,
                                 text=f"LOCK: {'STABLE' if lock else 'ACTIVE'}")
        self.pill_blink.set_state(m.is_blinking,
                                  text=f"BLINK: {'YES' if m.is_blinking else 'no'}")
        self.pill_smile.set_state(m.smile > 0.4,
                                  text=f"SMILE: {int(m.smile * 100):2d}%")
        self.pill_talk.set_state(m.is_talking,
                                 text=f"VOX: {'speaking' if m.is_talking else 'idle'}")
        self.pill_yawn.set_state(m.is_yawning,
                                 text=f"YAWN: {m.yawn_count}")

        self.session_table.set("Session ID", self.session_id)
        self.session_table.set("Frame", f"{self._frame_idx:,}")
        self.session_table.set("Detect rate",
                               f"{self.session_stats.detection_rate * 100:.1f}%")
        rs = self.session_stats.runtime_s
        self.session_table.set("Runtime",
                               f"{int(rs // 3600):02d}:{int((rs % 3600) // 60):02d}:{int(rs % 60):02d}")
        self.session_table.set("Attention",
                               f"{m.attention_s:.1f}s")
        self.session_table.set("Look-aways", str(m.look_away_count))
        self.session_table.set("Yawns", str(m.yawn_count))
        self.session_table.set("Recording", "* live" if self.recording else "idle")

        self.misc_table.set("Glasses",
                            "likely" if m.glasses_likely else "no signal")
        rL, gL, bL = m.iris_color_left
        rR, gR, bR = m.iris_color_right
        self.misc_table.set("Iris L", f"#{rL:02x}{gL:02x}{bL:02x}")
        self.misc_table.set("Iris R", f"#{rR:02x}{gR:02x}{bR:02x}")
        self.misc_table.set("Stability", f"{m.stability * 100:.0f}%")

        d_score = self.drowsiness.score
        d_label = self.drowsiness.label
        d_color = (C.UI_GOOD if d_score < 0.4
                   else (C.UI_WARN if d_score < 0.7 else C.UI_BAD))
        self.badge_drowsy.set(
            d_label,
            detail=f"score {d_score*100:.0f}%  -  blinks/min {len(self.drowsiness._blink_times)}",
            level=d_score, color=d_color,
        )

        post_color = {
            "ok": C.UI_GOOD,
            "too close": C.UI_WARN,
            "too far": C.UI_INFO,
            "hunched": C.UI_BAD,
        }.get(self.posture.state, C.UI_TEXT_DIM)
        self.badge_posture.set(
            self.posture.state, detail=self.posture.detail,
            level=1.0 if self.posture.state != "ok" else 0.5,
            color=post_color,
        )

    def _update_widgets_empty(self) -> None:
        self.bar_smile.set_value(0)
        self.bar_mouth.set_value(0)
        self.bar_brows.set_value(0)
        self.bar_sym.set_value(0)
        self.bar_dist.set_value(0)
        self.bar_area.set_value(0)
        self.gaze_radial.set_gaze(0, 0, " - ")
        self.pose_dial.set_values(0, 0, 0)
        self.emotion_value.configure(text=" - ")
        self.shape_value.configure(text=" - ")
        self.pill_face.set_state(False, "FACE: NONE")
        self.pill_lock.set_state(False, "LOCK: STANDBY")
        self.pill_blink.set_state(False, "BLINK:  - ")
        self.pill_smile.set_state(False, "SMILE:  - ")
        self.pill_talk.set_state(False, "VOX: idle")
        self.pill_yawn.set_state(False, "YAWN:  - ")
        self.badge_drowsy.set(" - ", "no signal", level=0.0, color=C.UI_TEXT_DIM)
        self.badge_posture.set(" - ", "no face detected",
                               level=0.0, color=C.UI_TEXT_DIM)

    def _update_status_bar(self, fps: float, faces: List[FaceData]) -> None:
        self.status.set("backend", f"backend: {self.tracker.backend}")
        self.status.set("model", "model: face_landmarker")
        self.status.set("latency", f"{self._latency_ms:.0f} ms")
        if self.capture:
            aw, ah = self.capture.actual_resolution
            self.status.set("res", f"{aw}x{ah}")
        if faces:
            self.status.set("landmarks", f"{len(faces[0].landmarks_px)} pts")
        else:
            self.status.set("landmarks", "0 pts")
        self.diag_table.set("Backend", self.tracker.backend)
        if self.capture:
            self.diag_table.set("Resolution",
                                f"{self.capture.actual_resolution[0]}x{self.capture.actual_resolution[1]}")
            if self.frame_for_display is not None:
                self.diag_table.set("Detected res",
                                    f"{self.frame_for_display.shape[1]}x{self.frame_for_display.shape[0]}")
        self.diag_table.set("Latency", f"{self._latency_ms:.1f} ms")
        self.diag_table.set("Landmarks",
                            f"{len(faces[0].landmarks_px) if faces else 0}")
        self.diag_table.set("Smoothing", "on" if self.smoothing.get() else "off")
        self.diag_table.set("Iris refine", "on" if self.iris_refine.get() else "off")
        self.diag_table.set("Auto-record",
                            "armed" if self.auto_record.get() else "off")
        self.diag_table.set("CSV log",
                            "writing" if self.csv_logger else "off")
        self.diag_table.set("Replay buffer",
                            f"{self.replay.seconds_buffered:.1f}s")

    def _on_close(self) -> None:
        try:
            if self.recording and self.video_writer is not None:
                self.video_writer.release()
            if self.capture:
                self.capture.stop()
            if self.csv_logger:
                self.csv_logger.close()
            self.tracker.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-q>", lambda e: self._on_close())
        self.root.bind("<KeyPress-Q>", lambda e: self._on_close())
        self.root.bind("<KeyPress-s>", lambda e: self._snapshot())
        self.root.bind("<KeyPress-S>", lambda e: self._snapshot())
        self.root.bind("<KeyPress-r>", lambda e: self._toggle_recording())
        self.root.bind("<KeyPress-R>", lambda e: self._toggle_recording())
        self.root.bind("<KeyPress-b>", lambda e: self._save_replay())
        self.root.bind("<KeyPress-B>", lambda e: self._save_replay())
        self.root.bind("<KeyPress-m>", lambda e: self._add_bookmark())
        self.root.bind("<KeyPress-M>", lambda e: self._add_bookmark())
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("<F11>", lambda e: self._toggle_panel())

    def run(self) -> None:
        self.root.mainloop()
