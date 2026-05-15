from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import constants as C
from .filtering import OneEuroLandmarkFilter
from .model import ensure_model


_API: Optional[str] = None
_API_ERROR: Optional[str] = None
try:
    from mediapipe.tasks.python.vision import (  # type: ignore
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
        FaceLandmarksConnections,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
    import mediapipe as _mp  # type: ignore
    _API = "tasks"
except Exception as _e_tasks:
    _API_ERROR = f"Tasks API not available: {_e_tasks}"
    try:
        from mediapipe.python.solutions import face_mesh as _mp_face_mesh  # type: ignore
        from mediapipe.python.solutions import drawing_utils as _mp_drawing  # type: ignore
        from mediapipe.python.solutions import drawing_styles as _mp_drawing_styles  # type: ignore
        _API = "solutions"
        _API_ERROR = None
    except Exception as _e_sol:
        _API_ERROR = (
            f"Neither MediaPipe Tasks nor Solutions API is importable.\n"
            f"Tasks error:     {_e_tasks}\n"
            f"Solutions error: {_e_sol}"
        )


def api_in_use() -> str:
    if _API is None:
        raise ImportError(_API_ERROR or "No MediaPipe API available")
    return _API


@dataclass
class FaceData:
    landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    bbox: Tuple[int, int, int, int]
    image_shape: Tuple[int, int]
    blendshapes: dict = field(default_factory=dict)
    mp_landmarks: object = field(default=None, repr=False)


def _draw_connections(
    frame: np.ndarray,
    pts_px: np.ndarray,
    connections,
    color: Tuple[int, int, int] = (120, 200, 255),
    thickness: int = 1,
) -> None:
    n = len(pts_px)
    for conn in connections:
        try:
            if hasattr(conn, "start") and hasattr(conn, "end"):
                a, b = int(conn.start), int(conn.end)
            else:
                a, b = int(conn[0]), int(conn[1])
        except Exception:
            continue
        if 0 <= a < n and 0 <= b < n:
            p1 = tuple(int(v) for v in pts_px[a])
            p2 = tuple(int(v) for v in pts_px[b])
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


class _TasksTracker:
    def __init__(
        self,
        model_path: str,
        max_faces: int = 2,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.30,
        )
        self._lm = FaceLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._smoothers: dict = {}
        self.smoothing_enabled = True

    def process(self, frame_bgr: np.ndarray) -> List[FaceData]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        try:
            result = self._lm.detect_for_video(mp_image, ts_ms)
        except Exception:
            return []
        faces: List[FaceData] = []
        if not result.face_landmarks:
            return faces
        for i, lm_list in enumerate(result.face_landmarks):
            pts_norm = np.array(
                [[lm.x, lm.y, lm.z if hasattr(lm, "z") else 0.0] for lm in lm_list],
                dtype=np.float32,
            )
            if self.smoothing_enabled:
                if i not in self._smoothers:
                    self._smoothers[i] = OneEuroLandmarkFilter()
                pts_norm = self._smoothers[i].filter(pts_norm)
            pts_px = np.stack(
                [pts_norm[:, 0] * w, pts_norm[:, 1] * h], axis=1
            ).astype(np.int32)
            xs, ys = pts_px[:, 0], pts_px[:, 1]
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            blendshapes: dict = {}
            if getattr(result, "face_blendshapes", None) and i < len(result.face_blendshapes):
                for cat in result.face_blendshapes[i]:
                    blendshapes[cat.category_name] = float(cat.score)
            faces.append(
                FaceData(
                    landmarks_px=pts_px,
                    landmarks_norm=pts_norm,
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    image_shape=(h, w),
                    blendshapes=blendshapes,
                )
            )
        return faces

    def draw_mesh(self, frame, face):
        _draw_connections(
            frame, face.landmarks_px,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            color=(70, 90, 130), thickness=1,
        )

    def draw_contours(self, frame, face):
        conn_groups = [
            (FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,    (255, 240, 200)),
            (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,     (120, 220, 255)),
            (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,    (120, 220, 255)),
            (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW, (200, 160, 255)),
            (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,(200, 160, 255)),
            (FaceLandmarksConnections.FACE_LANDMARKS_NOSE,         (220, 220, 220)),
            (FaceLandmarksConnections.FACE_LANDMARKS_LIPS,         (140, 230, 180)),
        ]
        for conn, color in conn_groups:
            _draw_connections(frame, face.landmarks_px, conn, color=color, thickness=1)

    def draw_irises(self, frame, face):
        for conn in [FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                     FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS]:
            _draw_connections(frame, face.landmarks_px, conn,
                              color=(0, 255, 220), thickness=2)

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


class _SolutionsTracker:
    def __init__(self, max_faces: int = 2, refine_landmarks: bool = True, **_) -> None:
        self._mesh = _mp_face_mesh.FaceMesh(  # noqa
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smoothers: dict = {}
        self.smoothing_enabled = True

    def process(self, frame_bgr: np.ndarray) -> List[FaceData]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)
        rgb.flags.writeable = True
        faces: List[FaceData] = []
        if not results.multi_face_landmarks:
            return faces
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            pts_norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            if self.smoothing_enabled:
                if i not in self._smoothers:
                    self._smoothers[i] = OneEuroLandmarkFilter()
                pts_norm = self._smoothers[i].filter(pts_norm)
            pts_px = np.stack(
                [pts_norm[:, 0] * w, pts_norm[:, 1] * h], axis=1
            ).astype(np.int32)
            xs, ys = pts_px[:, 0], pts_px[:, 1]
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            faces.append(
                FaceData(
                    landmarks_px=pts_px,
                    landmarks_norm=pts_norm,
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    image_shape=(h, w),
                    blendshapes={},
                    mp_landmarks=face_landmarks,
                )
            )
        return faces

    def draw_mesh(self, frame, face):
        _mp_drawing.draw_landmarks(  # noqa
            image=frame, landmark_list=face.mp_landmarks,
            connections=_mp_face_mesh.FACEMESH_TESSELATION,  # noqa
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_drawing_styles  # noqa
                .get_default_face_mesh_tesselation_style(),
        )

    def draw_contours(self, frame, face):
        _mp_drawing.draw_landmarks(  # noqa
            image=frame, landmark_list=face.mp_landmarks,
            connections=_mp_face_mesh.FACEMESH_CONTOURS,  # noqa
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_drawing_styles  # noqa
                .get_default_face_mesh_contours_style(),
        )

    def draw_irises(self, frame, face):
        _mp_drawing.draw_landmarks(  # noqa
            image=frame, landmark_list=face.mp_landmarks,
            connections=_mp_face_mesh.FACEMESH_IRISES,  # noqa
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_drawing_styles  # noqa
                .get_default_face_mesh_iris_connections_style(),
        )

    def close(self):
        try:
            self._mesh.close()
        except Exception:
            pass


class FaceTracker:
    def __init__(
        self,
        max_faces: int = 2,
        refine_landmarks: bool = True,
        model_path: Optional[str] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        if _API is None:
            raise ImportError(_API_ERROR or "No MediaPipe API available")
        self._api = _API
        if _API == "tasks":
            if model_path is None:
                model_path = str(ensure_model(progress_cb=progress_cb))
            self._impl: object = _TasksTracker(
                model_path=model_path, max_faces=max_faces, progress_cb=progress_cb,
            )
        else:
            self._impl = _SolutionsTracker(
                max_faces=max_faces, refine_landmarks=refine_landmarks,
            )

    @property
    def backend(self) -> str:
        return self._api

    def set_smoothing(self, enabled: bool) -> None:
        try:
            self._impl.smoothing_enabled = enabled  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def configure_smoothing(self, *, mincutoff: Optional[float] = None,
                            beta: Optional[float] = None) -> None:
        smoothers = getattr(self._impl, "_smoothers", {})
        for s in smoothers.values():
            s.configure(mincutoff=mincutoff, beta=beta)

    def process(self, frame_bgr: np.ndarray) -> List[FaceData]:
        return self._impl.process(frame_bgr)  # type: ignore[attr-defined]

    def draw_mesh(self, frame: np.ndarray, face: FaceData) -> None:
        self._impl.draw_mesh(frame, face)  # type: ignore[attr-defined]

    def draw_contours(self, frame: np.ndarray, face: FaceData) -> None:
        self._impl.draw_contours(frame, face)  # type: ignore[attr-defined]

    def draw_irises(self, frame: np.ndarray, face: FaceData) -> None:
        self._impl.draw_irises(frame, face)  # type: ignore[attr-defined]

    def draw_bbox(
        self, frame: np.ndarray, face: FaceData, label: Optional[str] = None
    ) -> None:
        x, y, w, h = face.bbox
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = x + w + pad, y + h + pad
        cv2.rectangle(frame, (x0, y0), (x1, y1), C.COLOR_BOX, 2)
        corner = 18
        for (cx, cy, dx, dy) in [
            (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), C.COLOR_BOX, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), C.COLOR_BOX, 3)
        if label:
            cv2.putText(
                frame, label, (x0, max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C.COLOR_TEXT, 1, cv2.LINE_AA,
            )

    def close(self) -> None:
        self._impl.close()  # type: ignore[attr-defined]
