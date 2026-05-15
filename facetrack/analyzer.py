from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from . import constants as C
from .tracker import FaceData


@dataclass
class FaceMetrics:
    ear_left: float = 0.0
    ear_right: float = 0.0
    is_blinking: bool = False
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    mouth_open: float = 0.0
    smile: float = 0.0
    eyebrow_raise: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    gaze_label: str = "Center"
    symmetry: float = 1.0
    distance_cm: float = 0.0
    face_shape: str = "Unknown"
    face_shape_conf: float = 0.0
    shape_scores: Dict[str, float] = field(default_factory=dict)
    emotion: str = "Neutral"
    emotion_conf: float = 0.0

    face_area_pct: float = 0.0
    stability: float = 1.0
    is_talking: bool = False
    is_yawning: bool = False
    yawn_count: int = 0
    looking_away: bool = False
    attention_s: float = 0.0
    look_away_count: int = 0
    glasses_likely: bool = False
    iris_color_left: Tuple[int, int, int] = (0, 0, 0)
    iris_color_right: Tuple[int, int, int] = (0, 0, 0)
    velocity_px_per_s: Tuple[float, float] = (0.0, 0.0)
    head_speed: float = 0.0


class FaceAnalyzer:
    AVG_INTEROCULAR_MM = 63.0
    FOCAL_LEN_PX = 650.0

    def __init__(self, blink_threshold: float = 0.21, blink_consec: int = 2) -> None:
        self.blink_threshold = blink_threshold
        self.blink_consec = blink_consec
        self._below_count = 0
        self._blink_count = 0
        self._blink_timestamps: Deque[float] = deque(maxlen=120)
        self._smile_baseline: Optional[float] = None
        self._eyebrow_baseline: Optional[float] = None
        self._ear_baseline_samples: Deque[float] = deque(maxlen=240)
        self._pose_ema: Optional[np.ndarray] = None
        self._pose_alpha = 0.4

        self._shape_history: Deque[Dict[str, float]] = deque(maxlen=12)
        self._emotion_history: Deque[Dict[str, float]] = deque(maxlen=10)
        self._mouth_history: Deque[float] = deque(maxlen=30)
        self._nose_history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=40)
        self._yawn_active = False
        self._yawn_start_t: Optional[float] = None
        self._yawn_count = 0
        self._look_away_active = False
        self._look_away_count = 0
        self._look_away_start_t: Optional[float] = None
        self._attention_total = 0.0
        self._attention_last_t: Optional[float] = None
        self._neutral_start_t: Optional[float] = None
        self._auto_calibrate_after = 4.0

    def reset_blinks(self) -> None:
        self._blink_count = 0
        self._below_count = 0
        self._blink_timestamps.clear()

    def calibrate(self) -> None:
        self._smile_baseline = None
        self._eyebrow_baseline = None
        self._ear_baseline_samples.clear()

    def analyze(self, face: FaceData) -> FaceMetrics:
        m = FaceMetrics()
        pts = face.landmarks_px.astype(np.float32)
        h, w = face.image_shape
        bs = getattr(face, "blendshapes", None) or {}

        self._compute_eyes(pts, m)
        if "eyeBlinkLeft" in bs:
            self._update_blink_state_from_blendshapes(bs, m)
        else:
            self._update_blink_state(m)

        if "jawOpen" in bs:
            m.mouth_open = float(np.clip(bs.get("jawOpen", 0.0), 0.0, 1.0))
        else:
            self._compute_mouth(pts, m)

        if "mouthSmileLeft" in bs:
            sL = float(bs.get("mouthSmileLeft", 0.0))
            sR = float(bs.get("mouthSmileRight", 0.0))
            m.smile = float(np.clip((sL + sR) * 0.5 * 1.4, 0.0, 1.0))
        else:
            self._compute_smile(pts, m)

        if "browInnerUp" in bs:
            up = (bs.get("browInnerUp", 0.0)
                  + bs.get("browOuterUpLeft", 0.0)
                  + bs.get("browOuterUpRight", 0.0)) / 3.0
            down = (bs.get("browDownLeft", 0.0)
                    + bs.get("browDownRight", 0.0)) / 2.0
            m.eyebrow_raise = float(np.clip(up - down, -1.0, 1.0))
        else:
            self._compute_eyebrow(pts, m)

        self._compute_head_pose(pts, (h, w), m)
        self._compute_gaze(pts, m)
        self._compute_symmetry(pts, m)
        self._compute_distance(pts, m)
        self._compute_face_shape(pts, m)
        self._smooth_face_shape(m)
        self._compute_emotion(m)
        self._smooth_emotion(m)

        self._compute_face_area(face, m)
        self._compute_head_kinematics(pts, m)
        self._compute_stability(m)
        self._compute_talking_and_yawn(m)
        self._compute_look_away_attention(m)
        self._maybe_auto_calibrate(m)

        return m

    def _update_blink_state_from_blendshapes(self, bs: dict, m: FaceMetrics) -> None:
        eL = float(bs.get("eyeBlinkLeft", 0.0))
        eR = float(bs.get("eyeBlinkRight", 0.0))
        score = max(eL, eR)
        m.ear_left = 1.0 - eL
        m.ear_right = 1.0 - eR
        if score > 0.5:
            self._below_count += 1
            m.is_blinking = True
        else:
            if self._below_count >= self.blink_consec:
                self._blink_count += 1
                self._blink_timestamps.append(time.time())
            self._below_count = 0
            m.is_blinking = False
        m.blink_count = self._blink_count
        cutoff = time.time() - 60.0
        recent = [t for t in self._blink_timestamps if t >= cutoff]
        m.blink_rate_per_min = float(len(recent))

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray, idx) -> float:
        p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        if h < 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def _compute_eyes(self, pts: np.ndarray, m: FaceMetrics) -> None:
        m.ear_left = self._eye_aspect_ratio(pts, C.LEFT_EYE_EAR)
        m.ear_right = self._eye_aspect_ratio(pts, C.RIGHT_EYE_EAR)

    def _update_blink_state(self, m: FaceMetrics) -> None:
        ear = (m.ear_left + m.ear_right) / 2.0
        if ear < self.blink_threshold:
            self._below_count += 1
            m.is_blinking = True
        else:
            if self._below_count >= self.blink_consec:
                self._blink_count += 1
                self._blink_timestamps.append(time.time())
            self._below_count = 0
            m.is_blinking = False
        m.blink_count = self._blink_count
        cutoff = time.time() - 60.0
        recent = [t for t in self._blink_timestamps if t >= cutoff]
        m.blink_rate_per_min = float(len(recent))

    def _compute_mouth(self, pts: np.ndarray, m: FaceMetrics) -> None:
        top = pts[C.MOUTH_TOP]
        bot = pts[C.MOUTH_BOTTOM]
        left = pts[C.MOUTH_LEFT]
        right = pts[C.MOUTH_RIGHT]
        h = np.linalg.norm(top - bot)
        w = np.linalg.norm(left - right)
        if w < 1e-6:
            m.mouth_open = 0.0
        else:
            m.mouth_open = float(np.clip(h / w, 0.0, 1.0))

    def _compute_smile(self, pts: np.ndarray, m: FaceMetrics) -> None:
        mouth_w = np.linalg.norm(pts[C.SMILE_LEFT_CORNER] - pts[C.SMILE_RIGHT_CORNER])
        eye_w = np.linalg.norm(pts[33] - pts[263])
        if eye_w < 1e-6:
            ratio = 0.0
        else:
            ratio = float(mouth_w / eye_w)
        if self._smile_baseline is None:
            self._smile_baseline = ratio
        diff = ratio - self._smile_baseline
        m.smile = float(np.clip(diff / 0.15, 0.0, 1.0))

    def _compute_eyebrow(self, pts: np.ndarray, m: FaceMetrics) -> None:
        left_dist = np.linalg.norm(pts[C.LEFT_EYEBROW_TOP] - pts[33])
        right_dist = np.linalg.norm(pts[C.RIGHT_EYEBROW_TOP] - pts[263])
        face_h = np.linalg.norm(pts[C.FOREHEAD_TOP] - pts[C.CHIN])
        if face_h < 1e-6:
            ratio = 0.0
        else:
            ratio = (left_dist + right_dist) / (2.0 * face_h)
        if self._eyebrow_baseline is None:
            self._eyebrow_baseline = ratio
        diff = ratio - self._eyebrow_baseline
        m.eyebrow_raise = float(np.clip(diff / 0.03, -1.0, 1.0))

    def _compute_head_pose(
        self, pts: np.ndarray, shape: Tuple[int, int], m: FaceMetrics
    ) -> None:
        h, w = shape
        image_pts = np.array(
            [pts[C.POSE_LANDMARKS[k]] for k in
             ["nose_tip", "chin", "left_eye_corner", "right_eye_corner",
              "left_mouth", "right_mouth"]],
            dtype=np.float64,
        )
        model_pts = np.array(C.POSE_3D_MODEL, dtype=np.float64)
        focal = float(w)
        cam = np.array(
            [[focal, 0, w / 2.0],
             [0, focal, h / 2.0],
             [0, 0, 1.0]], dtype=np.float64,
        )
        dist = np.zeros((4, 1))
        ok, rvec, tvec = cv2.solvePnP(
            model_pts, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
            roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        else:
            pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            yaw = 0.0
            roll = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        vec = np.array([pitch, yaw, roll], dtype=np.float32)
        if self._pose_ema is None:
            self._pose_ema = vec
        else:
            self._pose_ema = (1 - self._pose_alpha) * self._pose_ema + self._pose_alpha * vec
        m.pitch = float(self._pose_ema[0])
        m.yaw = float(self._pose_ema[1])
        m.roll = float(self._pose_ema[2])

    def _compute_gaze(self, pts: np.ndarray, m: FaceMetrics) -> None:
        l_eye_l, l_eye_r = pts[33], pts[133]
        r_eye_l, r_eye_r = pts[362], pts[263]
        l_iris = pts[C.LEFT_IRIS_CENTER]
        r_iris = pts[C.RIGHT_IRIS_CENTER]

        l_mid = (l_eye_l + l_eye_r) / 2
        r_mid = (r_eye_l + r_eye_r) / 2
        l_w = np.linalg.norm(l_eye_l - l_eye_r) + 1e-6
        r_w = np.linalg.norm(r_eye_l - r_eye_r) + 1e-6

        l_dx = (l_iris[0] - l_mid[0]) / (l_w / 2)
        r_dx = (r_iris[0] - r_mid[0]) / (r_w / 2)
        l_dy = (l_iris[1] - l_mid[1]) / (l_w / 2)
        r_dy = (r_iris[1] - r_mid[1]) / (r_w / 2)

        gx = float(np.clip((l_dx + r_dx) / 2.0, -1.0, 1.0))
        gy = float(np.clip((l_dy + r_dy) / 2.0, -1.0, 1.0))
        m.gaze_x = gx
        m.gaze_y = gy

        thresh = 0.25
        parts = []
        if gy < -thresh:
            parts.append("Up")
        elif gy > thresh:
            parts.append("Down")
        if gx < -thresh:
            parts.append("Left")
        elif gx > thresh:
            parts.append("Right")
        m.gaze_label = " ".join(parts) if parts else "Center"

    def _compute_symmetry(self, pts: np.ndarray, m: FaceMetrics) -> None:
        nose = pts[C.NOSE_TIP]
        top = pts[C.FOREHEAD_TOP]
        axis = top - nose
        n = np.linalg.norm(axis)
        if n < 1e-6:
            m.symmetry = 1.0
            return
        axis /= n
        normal = np.array([-axis[1], axis[0]])

        pairs = [
            (33, 263), (61, 291), (234, 454), (172, 397),
            (50, 280), (127, 356), (93, 323), (132, 361),
        ]
        diffs = []
        for l, r in pairs:
            pl = pts[l] - nose
            pr = pts[r] - nose
            dl = pl @ normal
            dr = pr @ normal
            al = pl @ axis
            ar = pr @ axis
            diffs.append(abs(dl + dr))
            diffs.append(abs(al - ar))
        face_size = np.linalg.norm(pts[C.LEFT_TEMPLE] - pts[C.RIGHT_TEMPLE]) + 1e-6
        err = float(np.mean(diffs)) / face_size
        m.symmetry = float(np.clip(1.0 - err * 4.0, 0.0, 1.0))

    def _compute_distance(self, pts: np.ndarray, m: FaceMetrics) -> None:
        eye_dist_px = np.linalg.norm(pts[33] - pts[263])
        if eye_dist_px < 1.0:
            m.distance_cm = 0.0
            return
        dist_mm = (self.FOCAL_LEN_PX * self.AVG_INTEROCULAR_MM) / eye_dist_px
        m.distance_cm = float(dist_mm / 10.0)

    def _compute_face_shape(self, pts: np.ndarray, m: FaceMetrics) -> None:
        face_len = float(np.linalg.norm(pts[C.FOREHEAD_TOP] - pts[C.CHIN]))
        forehead_w = float(np.linalg.norm(pts[103] - pts[332]))
        cheek_w = float(np.linalg.norm(pts[C.LEFT_TEMPLE] - pts[C.RIGHT_TEMPLE]))
        jaw_w = float(np.linalg.norm(pts[C.LEFT_JAW] - pts[C.RIGHT_JAW]))

        if cheek_w < 1.0 or face_len < 1.0:
            m.face_shape = "Unknown"
            m.face_shape_conf = 0.0
            return

        len_to_cheek = face_len / cheek_w
        jaw_to_cheek = jaw_w / cheek_w
        forehead_to_cheek = forehead_w / cheek_w
        jaw_to_forehead = jaw_w / max(forehead_w, 1.0)

        def gauss(x, mu, sigma):
            return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        scores: Dict[str, float] = {}
        scores["Oval"] = (
            gauss(len_to_cheek, 1.5, 0.12) *
            gauss(jaw_to_cheek, 0.78, 0.10) *
            gauss(forehead_to_cheek, 0.78, 0.10)
        )
        scores["Round"] = (
            gauss(len_to_cheek, 1.0, 0.10) *
            gauss(jaw_to_cheek, 0.90, 0.08)
        )
        scores["Square"] = (
            gauss(len_to_cheek, 1.05, 0.10) *
            gauss(jaw_to_cheek, 0.98, 0.08) *
            gauss(forehead_to_cheek, 0.98, 0.10)
        )
        scores["Heart"] = (
            gauss(len_to_cheek, 1.3, 0.15) *
            gauss(forehead_to_cheek, 0.95, 0.10) *
            gauss(jaw_to_forehead, 0.65, 0.12)
        )
        scores["Diamond"] = (
            gauss(len_to_cheek, 1.4, 0.15) *
            gauss(forehead_to_cheek, 0.75, 0.10) *
            gauss(jaw_to_cheek, 0.72, 0.10)
        )
        scores["Oblong"] = (
            gauss(len_to_cheek, 1.7, 0.12) *
            gauss(jaw_to_cheek, 0.88, 0.12) *
            gauss(forehead_to_cheek, 0.88, 0.12)
        )
        scores["Triangle"] = (
            gauss(len_to_cheek, 1.3, 0.15) *
            gauss(forehead_to_cheek, 0.72, 0.10) *
            gauss(jaw_to_forehead, 1.20, 0.15)
        )

        total = sum(scores.values()) + 1e-9
        scores = {k: v / total for k, v in scores.items()}
        best = max(scores.items(), key=lambda kv: kv[1])
        m.shape_scores = scores
        m.face_shape = best[0]
        m.face_shape_conf = float(best[1])

    def _compute_emotion(self, m: FaceMetrics) -> None:
        smile = m.smile
        brows = m.eyebrow_raise
        mouth = m.mouth_open
        ear = (m.ear_left + m.ear_right) / 2.0

        scores = {
            "Happy": smile * 1.2,
            "Surprised": max(0.0, brows) * 0.7 + mouth * 0.9,
            "Angry": max(0.0, -brows) * 0.9 + (1.0 - mouth) * 0.2,
            "Sad": max(0.0, -brows) * 0.4 + max(0.0, 0.05 - smile) * 4.0,
            "Focused": max(0.0, 0.25 - ear) * 2.0 if ear < 0.25 else 0.0,
            "Neutral": 0.35,
        }
        if smile < 0.15 and abs(brows) < 0.2 and mouth < 0.08:
            scores["Neutral"] += 0.4
        total = sum(scores.values()) + 1e-9
        scores = {k: v / total for k, v in scores.items()}
        best = max(scores.items(), key=lambda kv: kv[1])
        m.emotion = best[0]
        m.emotion_conf = float(best[1])

    @staticmethod
    def draw_pose_axes(frame: np.ndarray, face: FaceData, m: FaceMetrics) -> None:
        h, w = face.image_shape
        nose = tuple(face.landmarks_px[C.NOSE_TIP].astype(int))
        length = 70

        def rot(ang_x, ang_y, ang_z, vec):
            x = math.radians(ang_x)
            y = math.radians(ang_y)
            z = math.radians(ang_z)
            cx, sx = math.cos(x), math.sin(x)
            cy, sy = math.cos(y), math.sin(y)
            cz, sz = math.cos(z), math.sin(z)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            return rz @ ry @ rx @ np.array(vec)

        axes = {
            (0, 0, 255): [length, 0, 0],
            (0, 255, 0): [0, -length, 0],
            (255, 0, 0): [0, 0, -length],
        }
        for color, vec in axes.items():
            p = rot(m.pitch, m.yaw, m.roll, vec)
            end = (int(nose[0] + p[0]), int(nose[1] + p[1]))
            cv2.line(frame, nose, end, color, 2, cv2.LINE_AA)

    def _compute_face_area(self, face: FaceData, m: FaceMetrics) -> None:
        h, w = face.image_shape
        if h <= 0 or w <= 0:
            return
        _, _, bw, bh = face.bbox
        m.face_area_pct = float(np.clip((bw * bh) / float(h * w) * 100.0, 0.0, 100.0))

    def _compute_head_kinematics(self, pts: np.ndarray, m: FaceMetrics) -> None:
        now = time.time()
        nose = pts[C.NOSE_TIP].copy()
        self._nose_history.append((now, nose))
        if len(self._nose_history) < 2:
            return
        t0, p0 = self._nose_history[-2]
        dt = now - t0
        if dt <= 1e-6:
            return
        vx = (nose[0] - p0[0]) / dt
        vy = (nose[1] - p0[1]) / dt
        m.velocity_px_per_s = (float(vx), float(vy))
        m.head_speed = float(math.hypot(vx, vy))

    def _compute_stability(self, m: FaceMetrics) -> None:
        if len(self._nose_history) < 10:
            m.stability = 1.0
            return
        recent = np.array([p for _, p in list(self._nose_history)[-20:]])
        spread = float(recent.std(axis=0).mean())
        m.stability = float(np.clip(1.0 - spread / 12.0, 0.0, 1.0))

    def _compute_talking_and_yawn(self, m: FaceMetrics) -> None:
        self._mouth_history.append(m.mouth_open)
        if len(self._mouth_history) >= 8:
            arr = np.array(self._mouth_history)
            variance = float(arr.var())
            m.is_talking = bool(variance > 0.001 and arr.mean() > 0.04 and not m.is_yawning)
        else:
            m.is_talking = False

        if m.mouth_open > 0.55:
            if not self._yawn_active:
                self._yawn_active = True
                self._yawn_start_t = time.time()
            elif self._yawn_start_t is not None:
                duration = time.time() - self._yawn_start_t
                if duration > 1.2 and not m.is_yawning:
                    m.is_yawning = True
                    self._yawn_count += 1
        else:
            if self._yawn_active and self._yawn_start_t is not None:
                if (time.time() - self._yawn_start_t) > 1.2:
                    pass
            self._yawn_active = False
            self._yawn_start_t = None
            m.is_yawning = False
        m.yawn_count = self._yawn_count

    def _compute_look_away_attention(self, m: FaceMetrics) -> None:
        now = time.time()
        away = abs(m.yaw) > 25.0 or abs(m.pitch) > 22.0 or m.gaze_label not in ("Center", "")
        if away:
            if not self._look_away_active:
                self._look_away_active = True
                self._look_away_count += 1
                self._look_away_start_t = now
        else:
            if self._look_away_active:
                self._look_away_active = False
                self._look_away_start_t = None
            if self._attention_last_t is not None:
                self._attention_total += max(0.0, now - self._attention_last_t)
        self._attention_last_t = now if not away else None
        m.looking_away = self._look_away_active
        m.look_away_count = self._look_away_count
        m.attention_s = self._attention_total

    def _maybe_auto_calibrate(self, m: FaceMetrics) -> None:
        if self._smile_baseline is not None and self._eyebrow_baseline is not None:
            return
        looks_neutral = (
            abs(m.smile) < 0.18 and abs(m.eyebrow_raise) < 0.15
            and m.mouth_open < 0.08 and m.head_speed < 60.0
        )
        if looks_neutral:
            if self._neutral_start_t is None:
                self._neutral_start_t = time.time()
            elif (time.time() - self._neutral_start_t) >= self._auto_calibrate_after:
                pass
        else:
            self._neutral_start_t = None

    def _smooth_face_shape(self, m: FaceMetrics) -> None:
        if not m.shape_scores:
            return
        self._shape_history.append(m.shape_scores)
        agg: Dict[str, float] = {}
        for snap in self._shape_history:
            for k, v in snap.items():
                agg[k] = agg.get(k, 0.0) + v
        for k in agg:
            agg[k] /= len(self._shape_history)
        m.shape_scores = agg
        best = max(agg.items(), key=lambda kv: kv[1])
        m.face_shape = best[0]
        m.face_shape_conf = float(best[1])

    def _smooth_emotion(self, m: FaceMetrics) -> None:
        snap = {m.emotion: m.emotion_conf}
        self._emotion_history.append(snap)
        counts: Dict[str, float] = {}
        for s in self._emotion_history:
            for k, v in s.items():
                counts[k] = counts.get(k, 0.0) + v
        if counts:
            best = max(counts.items(), key=lambda kv: kv[1])
            m.emotion = best[0]
            m.emotion_conf = float(best[1] / max(1, len(self._emotion_history)))

    def refine_iris_centers(self, gray_frame: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
        if gray_frame is None or gray_frame.size == 0:
            return pts_px
        try:
            corners = np.array(
                [pts_px[C.LEFT_IRIS_CENTER], pts_px[C.RIGHT_IRIS_CENTER]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)
            cv2.cornerSubPix(gray_frame, corners, (5, 5), (-1, -1), criteria)
            refined = corners.reshape(-1, 2).astype(np.int32)
            pts_px = pts_px.copy()
            pts_px[C.LEFT_IRIS_CENTER] = refined[0]
            pts_px[C.RIGHT_IRIS_CENTER] = refined[1]
        except Exception:
            pass
        return pts_px

    def sample_iris_colors(self, bgr_frame: np.ndarray,
                           pts_px: np.ndarray, m: FaceMetrics) -> None:
        if bgr_frame is None or bgr_frame.size == 0:
            return
        h, w = bgr_frame.shape[:2]

        def avg_color(cx: int, cy: int, r: int = 3) -> Tuple[int, int, int]:
            x0 = max(0, cx - r); y0 = max(0, cy - r)
            x1 = min(w, cx + r + 1); y1 = min(h, cy + r + 1)
            roi = bgr_frame[y0:y1, x0:x1]
            if roi.size == 0:
                return (0, 0, 0)
            b, g, r_ = roi.mean(axis=(0, 1))
            return (int(r_), int(g), int(b))

        lx, ly = pts_px[C.LEFT_IRIS_CENTER]
        rx, ry = pts_px[C.RIGHT_IRIS_CENTER]
        m.iris_color_left = avg_color(int(lx), int(ly))
        m.iris_color_right = avg_color(int(rx), int(ry))

    def detect_glasses(self, gray_frame: np.ndarray, pts_px: np.ndarray,
                       m: FaceMetrics) -> None:
        if gray_frame is None or gray_frame.size == 0:
            return
        try:
            xs = pts_px[[33, 263, 130, 359, 70, 300], 0]
            ys = pts_px[[33, 263, 130, 359, 70, 300], 1]
            x0, x1 = max(0, int(xs.min()) - 10), min(gray_frame.shape[1], int(xs.max()) + 10)
            y0, y1 = max(0, int(ys.min()) - 18), min(gray_frame.shape[0], int(ys.max()) + 10)
            roi = gray_frame[y0:y1, x0:x1]
            if roi.size < 100:
                return
            edges = cv2.Canny(roi, 70, 160)
            edge_pct = float(edges.mean()) / 255.0
            m.glasses_likely = edge_pct > 0.10
        except Exception:
            return
