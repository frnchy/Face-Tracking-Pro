from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


class OneEuroLandmarkFilter:
    def __init__(
        self,
        mincutoff: float = 1.7,
        beta: float = 0.08,
        dcutoff: float = 1.0,
    ) -> None:
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def configure(self, *, mincutoff: Optional[float] = None,
                  beta: Optional[float] = None) -> None:
        if mincutoff is not None:
            self.mincutoff = float(mincutoff)
        if beta is not None:
            self.beta = float(beta)

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, freq: float) -> np.ndarray | float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = time.monotonic()
        if (self._x_prev is None
                or self._x_prev.shape != x.shape
                or self._t_prev is None):
            self._x_prev = x.astype(np.float32).copy()
            self._dx_prev = np.zeros_like(self._x_prev)
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)
        freq = 1.0 / dt
        self._t_prev = t

        x32 = x.astype(np.float32)
        dx = (x32 - self._x_prev) * freq
        a_d = self._alpha(self.dcutoff, freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        speed = np.linalg.norm(dx_hat.reshape(dx_hat.shape[0], -1),
                               axis=-1, keepdims=True)
        speed = speed.reshape(dx_hat.shape[0], *(1 for _ in dx_hat.shape[1:]))

        cutoff = self.mincutoff + self.beta * speed
        a = self._alpha(cutoff, freq)
        x_hat = a * x32 + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


@dataclass
class PreprocessConfig:
    clahe: bool = True
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    denoise: bool = False
    denoise_strength: int = 5
    sharpen: bool = False
    sharpen_amount: float = 0.6
    auto_gamma: bool = False
    gamma: float = 1.0
    upscale_small: bool = True


def _apply_clahe(bgr: np.ndarray, clip: float, grid: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _apply_denoise(bgr: np.ndarray, strength: int) -> np.ndarray:
    s = max(1, int(strength))
    return cv2.bilateralFilter(bgr, d=2 * s + 1, sigmaColor=20 * s, sigmaSpace=20 * s)


def _apply_sharpen(bgr: np.ndarray, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.5)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01:
        return bgr
    inv = 1.0 / max(0.05, gamma)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                     dtype=np.uint8)
    return cv2.LUT(bgr, table)


def _auto_gamma_value(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean()) / 255.0
    target = 0.5
    if mean < 0.05:
        return 1.0
    return float(np.clip(math.log(target) / math.log(mean), 0.4, 2.5))


def preprocess(frame_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    out = frame_bgr

    if cfg.upscale_small:
        h, w = out.shape[:2]
        target = 720
        if min(h, w) < target:
            scale = target / float(min(h, w))
            out = cv2.resize(out, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_LINEAR)

    if cfg.clahe:
        out = _apply_clahe(out, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.denoise:
        out = _apply_denoise(out, cfg.denoise_strength)
    if cfg.sharpen:
        out = _apply_sharpen(out, cfg.sharpen_amount)
    if cfg.auto_gamma:
        out = _apply_gamma(out, _auto_gamma_value(out))
    elif abs(cfg.gamma - 1.0) > 0.01:
        out = _apply_gamma(out, cfg.gamma)

    return out
