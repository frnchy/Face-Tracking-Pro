from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class DrowsinessMonitor:
    blink_window: float = 60.0
    _blink_times: Deque[float] = field(default_factory=deque)
    _last_yawn_count: int = 0
    _last_blink_count: int = 0
    _last_score: float = 0.0
    _last_yawn_at: Optional[float] = None
    _shown_warning_at: float = 0.0

    def update(self, blink_count: int, yawn_count: int,
               head_speed: float, attention_s: float) -> float:
        now = time.time()
        new_blinks = blink_count - self._last_blink_count
        if new_blinks > 0:
            for _ in range(min(new_blinks, 8)):
                self._blink_times.append(now)
            self._last_blink_count = blink_count
        cutoff = now - self.blink_window
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        bpm = len(self._blink_times)
        if yawn_count > self._last_yawn_count:
            self._last_yawn_at = now
            self._last_yawn_count = yawn_count

        blink_signal = min(1.0, max(0.0, bpm - 12) / 20.0)
        yawn_signal = 0.0
        if self._last_yawn_at is not None:
            recency = max(0.0, 1.0 - (now - self._last_yawn_at) / 120.0)
            yawn_signal = min(1.0, recency * 0.6 + self._last_yawn_count * 0.05)
        stillness_signal = max(0.0, 1.0 - head_speed / 60.0) * 0.3

        score = min(1.0, blink_signal * 0.55 + yawn_signal * 0.35
                    + stillness_signal * 0.10)
        self._last_score = 0.85 * self._last_score + 0.15 * score
        return self._last_score

    @property
    def score(self) -> float:
        return self._last_score

    @property
    def label(self) -> str:
        s = self._last_score
        if s < 0.25:
            return "alert"
        if s < 0.5:
            return "fine"
        if s < 0.75:
            return "drowsy"
        return "very drowsy"

    def should_warn(self, threshold: float = 0.7,
                    cooldown_s: float = 120.0) -> bool:
        if self._last_score < threshold:
            return False
        now = time.time()
        if now - self._shown_warning_at < cooldown_s:
            return False
        self._shown_warning_at = now
        return True


@dataclass
class PostureMonitor:
    _hunched_since: Optional[float] = None
    _last_warn_at: float = 0.0
    state: str = "ok"
    detail: str = ""

    def update(self, distance_cm: float, pitch_deg: float) -> None:
        if distance_cm <= 0:
            self.state = "unknown"
            self.detail = " - "
            self._hunched_since = None
            return
        if distance_cm < 35:
            self.state = "too close"
            self.detail = f"{distance_cm:.0f}cm  -  lean back"
        elif distance_cm > 90:
            self.state = "too far"
            self.detail = f"{distance_cm:.0f}cm  -  come closer"
        elif pitch_deg < -22:
            self.state = "hunched"
            self.detail = f"head down {pitch_deg:.0f}  -  sit up"
            if self._hunched_since is None:
                self._hunched_since = time.time()
        else:
            self.state = "ok"
            self.detail = f"{distance_cm:.0f}cm  -  posture ok"
            self._hunched_since = None

    def should_warn(self, cooldown_s: float = 30.0) -> bool:
        now = time.time()
        if self.state in ("too close", "hunched") and (
                now - self._last_warn_at > cooldown_s):
            self._last_warn_at = now
            return True
        return False


@dataclass
class EyeStrainReminder:
    interval_s: float = 20 * 60
    started_at: float = field(default_factory=time.time)
    _last_fired: float = 0.0

    def reset(self) -> None:
        self.started_at = time.time()
        self._last_fired = 0.0

    def seconds_to_next(self) -> float:
        now = time.time()
        elapsed = now - max(self.started_at, self._last_fired)
        return max(0.0, self.interval_s - elapsed)

    def should_fire(self) -> bool:
        if self.seconds_to_next() <= 0:
            self._last_fired = time.time()
            return True
        return False


class FaceHeatmap:
    def __init__(self, w: int = 32, h: int = 18, decay: float = 0.998) -> None:
        self.w = w
        self.h = h
        self.grid = np.zeros((h, w), dtype=np.float32)
        self.decay = decay

    def push(self, norm_x: float, norm_y: float, weight: float = 1.0) -> None:
        if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
            return
        self.grid *= self.decay
        gx = int(min(self.w - 1, max(0, norm_x * self.w)))
        gy = int(min(self.h - 1, max(0, norm_y * self.h)))
        self.grid[gy, gx] += weight
        max_v = float(self.grid.max())
        if max_v > 8.0:
            self.grid *= 0.75

    def coverage_pct(self) -> float:
        return float((self.grid > 0.05).mean() * 100.0)

    def clear(self) -> None:
        self.grid[:] = 0


def battery_status() -> Tuple[Optional[int], Optional[bool]]:
    try:
        import sys
        if not sys.platform.startswith("win"):
            return (None, None)
        import ctypes

        class SYS_POWER(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        sps = SYS_POWER()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)) == 0:
            return (None, None)
        pct = sps.BatteryLifePercent
        if pct == 255:
            return (None, None)
        charging = sps.ACLineStatus == 1
        return (int(pct), bool(charging))
    except Exception:
        return (None, None)
