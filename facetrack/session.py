from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SNAPSHOTS_DIR = Path(os.path.expanduser("~")) / "FaceTrackerSnapshots"
PROFILES_DIR = Path(os.path.expanduser("~")) / ".facetrack" / "profiles"


def _ensure_dirs() -> None:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Bookmark:
    timestamp: float
    label: str
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None


@dataclass
class SessionStats:
    started_at: float = field(default_factory=time.time)
    frame_count: int = 0
    detected_frames: int = 0
    blink_total: int = 0
    last_face_seen_at: float = 0.0
    attention_time_s: float = 0.0
    look_away_count: int = 0
    yawn_count: int = 0
    talking_time_s: float = 0.0
    peak_fps: float = 0.0

    @property
    def runtime_s(self) -> float:
        return time.time() - self.started_at

    @property
    def detection_rate(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.detected_frames / float(self.frame_count)


class CsvLogger:
    HEADER = [
        "iso_time", "epoch", "frame", "fps", "face_present",
        "blink_count", "is_blinking", "ear_avg",
        "smile", "mouth_open", "eyebrow_raise",
        "pitch", "yaw", "roll",
        "gaze_x", "gaze_y", "gaze_label",
        "distance_cm", "symmetry",
        "face_shape", "shape_conf",
        "emotion", "emotion_conf",
        "stability", "is_talking", "is_yawning",
        "attention_s", "looking_away",
    ]

    def __init__(self, path: Path, interval_s: float = 1.0) -> None:
        self.path = path
        self.interval = interval_s
        self._last_write = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self._fp = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        if new:
            self._writer.writerow(self.HEADER)
            self._fp.flush()

    def maybe_write(self, frame_idx: int, fps: float, face_present: bool,
                    metrics, extras: Dict[str, Any]) -> bool:
        now = time.time()
        if now - self._last_write < self.interval:
            return False
        self._last_write = now
        m = metrics
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            f"{now:.3f}",
            frame_idx, f"{fps:.2f}", int(face_present),
            m.blink_count if m else 0,
            int(getattr(m, "is_blinking", False)) if m else 0,
            f"{(m.ear_left + m.ear_right) / 2:.4f}" if m else "",
            f"{m.smile:.3f}" if m else "",
            f"{m.mouth_open:.3f}" if m else "",
            f"{m.eyebrow_raise:.3f}" if m else "",
            f"{m.pitch:.2f}" if m else "",
            f"{m.yaw:.2f}" if m else "",
            f"{m.roll:.2f}" if m else "",
            f"{m.gaze_x:.3f}" if m else "",
            f"{m.gaze_y:.3f}" if m else "",
            m.gaze_label if m else "",
            f"{m.distance_cm:.1f}" if m else "",
            f"{m.symmetry:.3f}" if m else "",
            m.face_shape if m else "",
            f"{m.face_shape_conf:.3f}" if m else "",
            m.emotion if m else "",
            f"{m.emotion_conf:.3f}" if m else "",
            f"{extras.get('stability', 0.0):.3f}",
            int(extras.get("is_talking", False)),
            int(extras.get("is_yawning", False)),
            f"{extras.get('attention_s', 0.0):.1f}",
            int(extras.get("looking_away", False)),
        ]
        self._writer.writerow(row)
        self._fp.flush()
        return True

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def export_metrics_json(out_path: Path, metrics, extras: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if metrics is not None:
        payload.update(asdict(metrics))
    payload["extras"] = extras
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


@dataclass
class Profile:
    name: str
    smile_baseline: Optional[float] = None
    eyebrow_baseline: Optional[float] = None
    blink_threshold: float = 0.21
    accent_color: str = "#d4a017"
    preprocess: Dict[str, Any] = field(default_factory=dict)
    overlays: Dict[str, bool] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def save(self) -> Path:
        _ensure_dirs()
        path = PROFILES_DIR / f"{self.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, name: str) -> Optional["Profile"]:
        path = PROFILES_DIR / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @staticmethod
    def list_all() -> List[str]:
        _ensure_dirs()
        return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def new_session_id() -> str:
    return f"OPS-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
