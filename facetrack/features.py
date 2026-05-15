from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


def apply_filter(frame: np.ndarray, name: str) -> np.ndarray:
    if name == "None":
        return frame
    if name == "Cartoon":
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2,
        )
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    if name == "Sketch":
        try:
            gray, _ = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except cv2.error:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sk = cv2.divide(gray, 255 - blur, scale=256)
            return cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)
    if name == "Edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 150)
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out[:, :, 0] = 0
        return out
    if name == "Sepia":
        k = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(frame, k), 0, 255).astype(np.uint8)
    if name == "Cool":
        out = frame.astype(np.int16)
        out[:, :, 0] += 25; out[:, :, 2] -= 15
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == "Warm":
        out = frame.astype(np.int16)
        out[:, :, 2] += 25; out[:, :, 0] -= 15
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == "Noir":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.7)
        gray = cv2.addWeighted(gray, 1.35, gray, 0, -25)
        return cv2.cvtColor(np.clip(gray, 0, 255), cv2.COLOR_GRAY2BGR)
    if name == "Thermal":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    if name == "Infrared":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return frame


def anonymize(frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
              mode: str, eye_y: Optional[int] = None,
              eye_x_left: Optional[int] = None,
              eye_x_right: Optional[int] = None) -> np.ndarray:
    if mode == "Off":
        return frame
    x, y, w, h = face_bbox
    if w <= 0 or h <= 0:
        return frame
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return frame
    if mode == "Blur face":
        k = max(15, ((min(roi.shape[:2]) // 6) | 1))
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    elif mode == "Pixelate face":
        small = cv2.resize(roi, (max(1, roi.shape[1] // 18),
                                 max(1, roi.shape[0] // 18)),
                           interpolation=cv2.INTER_LINEAR)
        frame[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
    elif mode == "Black bar (eyes)" and eye_y is not None:
        bar_h = max(10, h // 6)
        bar_y0 = max(0, eye_y - bar_h // 2)
        bar_y1 = min(H, eye_y + bar_h // 2)
        bx0 = max(0, (eye_x_left if eye_x_left is not None else x) - 10)
        bx1 = min(W, (eye_x_right if eye_x_right is not None else x + w) + 10)
        cv2.rectangle(frame, (bx0, bar_y0), (bx1, bar_y1), (0, 0, 0), -1)
    return frame


def crop_to_face(frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                 padding: float = 0.4) -> np.ndarray:
    x, y, w, h = face_bbox
    H, W = frame.shape[:2]
    px = int(w * padding)
    py = int(h * padding)
    x0 = max(0, x - px); y0 = max(0, y - py)
    x1 = min(W, x + w + px); y1 = min(H, y + h + py)
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1].copy()


def draw_rule_of_thirds(frame: np.ndarray, color=(60, 60, 60)) -> None:
    h, w = frame.shape[:2]
    for i in (1, 2):
        x = w * i // 3
        cv2.line(frame, (x, 0), (x, h), color, 1)
        y = h * i // 3
        cv2.line(frame, (0, y), (w, y), color, 1)


def draw_centered_crosshair(frame: np.ndarray, color=(80, 80, 80)) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    arm = max(20, min(w, h) // 16)
    gap = 8
    cv2.line(frame, (cx - arm, cy), (cx - gap, cy), color, 1)
    cv2.line(frame, (cx + gap, cy), (cx + arm, cy), color, 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy - gap), color, 1)
    cv2.line(frame, (cx, cy + gap), (cx, cy + arm), color, 1)
    cv2.circle(frame, (cx, cy), 3, color, 1)


def draw_grid_4x4(frame: np.ndarray, color=(36, 36, 36)) -> None:
    h, w = frame.shape[:2]
    for i in range(1, 4):
        x = w * i // 4
        cv2.line(frame, (x, 0), (x, h), color, 1)
        y = h * i // 4
        cv2.line(frame, (0, y), (w, y), color, 1)


def draw_center_dot(frame: np.ndarray, color=(120, 120, 120)) -> None:
    h, w = frame.shape[:2]
    cv2.circle(frame, (w // 2, h // 2), 4, color, -1)


def draw_grid(frame: np.ndarray, mode: str) -> None:
    if mode == "Rule of thirds":
        draw_rule_of_thirds(frame)
    elif mode == "Crosshair":
        draw_centered_crosshair(frame)
    elif mode == "Grid 4x4":
        draw_grid_4x4(frame)
    elif mode == "Center dot":
        draw_center_dot(frame)


def apply_scanlines(frame: np.ndarray, strength: float = 0.06) -> np.ndarray:
    if strength <= 0.001:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    overlay[::2, :, :] = (overlay[::2, :, :] * 0.55).astype(np.uint8)
    return cv2.addWeighted(frame, 1.0 - strength, overlay, strength, 0)


def draw_tactical_lock(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                       label: str, *, color=(23, 160, 212),
                       hot: bool = False) -> None:
    x, y, w, h = bbox
    pad = 12
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = x + w + pad, y + h + pad

    corner = 22
    for (cx, cy, dx, dy) in [
        (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, 2)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, 2)
        cv2.line(frame, (cx + dx * 4, cy + dy * 4),
                 (cx + dx * 9, cy + dy * 4), color, 1)
        cv2.line(frame, (cx + dx * 4, cy + dy * 4),
                 (cx + dx * 4, cy + dy * 9), color, 1)

    if hot:
        mid_x = (x0 + x1) // 2
        cv2.line(frame, (mid_x - 6, y0 - 1), (mid_x + 6, y0 - 1), color, 2)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.42
        (tw, th), _ = cv2.getTextSize(label, font, size, 1)
        ty = max(0, y0 - 7)
        cv2.rectangle(frame, (x0, ty - th - 4),
                      (x0 + tw + 10, ty + 4), color, -1)
        cv2.putText(frame, label, (x0 + 5, ty - 1),
                    font, size, (4, 4, 4), 1, cv2.LINE_AA)


def draw_target_reticle(frame: np.ndarray, center: Tuple[int, int],
                        color=(23, 160, 212)) -> None:
    cx, cy = int(center[0]), int(center[1])
    cl = 14
    cv2.line(frame, (cx - cl, cy), (cx - 5, cy), color, 1)
    cv2.line(frame, (cx + 5, cy), (cx + cl, cy), color, 1)
    cv2.line(frame, (cx, cy - cl), (cx, cy - 5), color, 1)
    cv2.line(frame, (cx, cy + 5), (cx, cy + cl), color, 1)
    cv2.circle(frame, (cx, cy), 3, color, 1)


def draw_velocity_arrow(frame: np.ndarray, start: Tuple[int, int],
                        velocity: Tuple[float, float],
                        color=(23, 160, 212)) -> None:
    sx, sy = int(start[0]), int(start[1])
    vx, vy = velocity
    mag = (vx * vx + vy * vy) ** 0.5
    if mag < 1.0:
        return
    scale = min(80.0, mag * 3.0)
    nx, ny = vx / mag, vy / mag
    ex = int(sx + nx * scale)
    ey = int(sy + ny * scale)
    cv2.arrowedLine(frame, (sx, sy), (ex, ey), color, 1,
                    line_type=cv2.LINE_AA, tipLength=0.25)


def rgb_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    if frame.size == 0:
        return np.zeros((3, bins), dtype=np.float32)
    h, w = frame.shape[:2]
    sub = frame
    if h * w > 320 * 240:
        scale = (320 * 240 / (h * w)) ** 0.5
        sub = cv2.resize(frame, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    out = np.empty((3, bins), dtype=np.float32)
    for i in range(3):
        h_ = cv2.calcHist([sub], [i], None, [bins], [0, 256]).flatten()
        m = h_.max()
        out[i] = (h_ / m) if m > 0 else h_
    out = out[[2, 1, 0]]
    return out


def frame_difference(prev: Optional[np.ndarray],
                     curr: np.ndarray) -> Optional[np.ndarray]:
    if prev is None or prev.shape != curr.shape:
        return None
    diff = cv2.absdiff(prev, curr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


@dataclass
class ReplayBuffer:
    duration_s: float = 8.0
    fps_est: float = 25.0
    max_dim: int = 720
    _frames: Deque[Tuple[float, np.ndarray]] = field(default_factory=deque)

    def push(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        h, w = frame.shape[:2]
        if max(h, w) > self.max_dim:
            scale = self.max_dim / float(max(h, w))
            small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            self._frames.append((now, small))
        else:
            self._frames.append((now, frame.copy()))
        cutoff = now - self.duration_s
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def export_mp4(self, out_path: str, fps: float = 24.0) -> bool:
        if not self._frames:
            return False
        f0 = self._frames[0][1]
        h, w = f0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not vw.isOpened():
            return False
        for _, f in self._frames:
            if f.shape[:2] == (h, w):
                vw.write(f)
        vw.release()
        return True

    @property
    def seconds_buffered(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1][0] - self._frames[0][0]

    def clear(self) -> None:
        self._frames.clear()
