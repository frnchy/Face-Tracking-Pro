from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

import cv2


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int

    @property
    def label(self) -> str:
        res = f"{self.width}×{self.height}" if self.width > 0 else "?"
        return f"{self.name}  ·  {res}"


def _list_dshow_names() -> List[str]:
    if not sys.platform.startswith("win"):
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        graph = FilterGraph()
        return list(graph.get_input_devices())
    except Exception:
        return []


def _probe_one(index: int, timeout_frames: int = 3) -> Optional[CameraInfo]:
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok = False
    for _ in range(timeout_frames):
        ok, _f = cap.read()
        if ok:
            break
    cap.release()
    if not ok:
        return None
    return CameraInfo(index=index, name=f"Camera {index}", width=w, height=h)


def list_cameras(max_probe: int = 6) -> List[CameraInfo]:
    names = _list_dshow_names()
    found: List[CameraInfo] = []
    for i in range(max_probe):
        info = _probe_one(i)
        if info is None:
            continue
        if i < len(names) and names[i]:
            info.name = names[i]
        found.append(info)
    return found


def best_resolution_for(index: int) -> tuple[int, int]:
    candidates = [
        (1920, 1080), (1600, 900), (1280, 720),
        (960, 540), (640, 480), (320, 240),
    ]
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        if cap:
            cap.release()
        return (640, 480)
    best = (640, 480)
    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if aw >= w * 0.95 and ah >= h * 0.95:
            best = (aw, ah)
            break
    cap.release()
    return best
