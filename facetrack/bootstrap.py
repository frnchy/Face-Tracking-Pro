from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


_DEPS: List[Tuple[str, str, str]] = [
    ("Python tkinter",       "tkinter",                                    ""),
    ("NumPy",                "numpy",                                      "numpy"),
    ("OpenCV",               "cv2",                                        "opencv-python"),
    ("Pillow",               "PIL",                                        "Pillow"),
    ("CustomTkinter",        "customtkinter",                              "customtkinter"),
    ("MediaPipe",            "mediapipe",                                  "mediapipe"),
]


@dataclass
class DependencyStatus:
    name: str
    module: str
    pip_name: str
    found: bool = False
    version: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CheckReport:
    python_version: str
    platform: str
    statuses: List[DependencyStatus] = field(default_factory=list)
    mediapipe_api: Optional[str] = None
    mediapipe_api_error: Optional[str] = None
    facemesh_loadable: Optional[bool] = None
    facemesh_error: Optional[str] = None
    model_path: Optional[str] = None
    webcam_available: Optional[bool] = None
    webcam_message: str = ""

    @property
    def missing(self) -> List[DependencyStatus]:
        return [s for s in self.statuses if not s.found]

    @property
    def all_required_ok(self) -> bool:
        return not self.missing and self.mediapipe_api is not None

    def missing_pip_names(self) -> List[str]:
        names = []
        seen = set()
        for s in self.missing:
            if s.pip_name and s.pip_name not in seen:
                seen.add(s.pip_name)
                names.append(s.pip_name)
        return names

    def install_hint(self) -> str:
        names = self.missing_pip_names()
        if not names:
            return ""
        return f"python -m pip install {' '.join(names)}"


class DependencyChecker:
    def check_all(
        self, progress: Optional[Callable[[float, str], None]] = None
    ) -> CheckReport:
        report = CheckReport(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}."
                           f"{sys.version_info.micro}",
            platform=sys.platform,
        )
        total = len(_DEPS) + 1
        for i, (display, mod, pip_name) in enumerate(_DEPS):
            if progress:
                progress(i / total, f"Checking {display}...")
            s = self._check_one(display, mod, pip_name)
            report.statuses.append(s)

        if progress:
            progress(len(_DEPS) / total, "Probing MediaPipe API...")
        api, err = self._probe_mediapipe()
        report.mediapipe_api = api
        report.mediapipe_api_error = err

        if progress:
            progress(1.0, "Done.")
        return report

    @staticmethod
    def _probe_mediapipe() -> Tuple[Optional[str], Optional[str]]:
        tasks_err = None
        try:
            import mediapipe.tasks.python.vision  # noqa: F401
            import mediapipe.tasks.python.core.base_options  # noqa: F401
            return "tasks", None
        except Exception as e:
            tasks_err = f"{type(e).__name__}: {e}"
        sol_err = None
        try:
            import mediapipe.python.solutions.face_mesh  # noqa: F401
            return "solutions", None
        except Exception as e:
            sol_err = f"{type(e).__name__}: {e}"
        return None, (
            f"Neither the Tasks API nor the Solutions API of MediaPipe is "
            f"importable.\n  tasks API:     {tasks_err}\n"
            f"  solutions API: {sol_err}\n"
            f"This usually means the installed mediapipe package is broken "
            f"or built for a different Python version."
        )

    @staticmethod
    def _check_one(display: str, module: str, pip_name: str) -> DependencyStatus:
        s = DependencyStatus(name=display, module=module, pip_name=pip_name)
        try:
            m = importlib.import_module(module)
            s.found = True
            s.version = getattr(m, "__version__", None)
        except ImportError as e:
            s.error = f"ImportError: {e}"
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
        return s

    @staticmethod
    def try_load_facemesh(
        report: CheckReport,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        if report.mediapipe_api is None:
            report.facemesh_loadable = False
            report.facemesh_error = "MediaPipe API not available"
            return
        try:
            if report.mediapipe_api == "tasks":
                if progress:
                    progress(0.1, "Locating face landmarker model...")
                from .model import ensure_model
                model_path = ensure_model(progress_cb=progress)
                report.model_path = str(model_path)
                if progress:
                    progress(0.9, "Constructing FaceLandmarker...")
                from mediapipe.tasks.python.vision import (
                    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
                )
                from mediapipe.tasks.python.core.base_options import BaseOptions
                opts = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.IMAGE, num_faces=1,
                    output_face_blendshapes=False,
                )
                lm = FaceLandmarker.create_from_options(opts)
                lm.close()
            else:
                if progress:
                    progress(0.5, "Constructing FaceMesh...")
                from mediapipe.python.solutions.face_mesh import FaceMesh
                fm = FaceMesh(max_num_faces=1, refine_landmarks=True)
                fm.close()
            report.facemesh_loadable = True
            if progress:
                progress(1.0, "Face landmarker loaded")
        except Exception as e:
            report.facemesh_loadable = False
            report.facemesh_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def try_open_webcam(
        report: CheckReport,
        progress: Optional[Callable[[float, str], None]] = None,
        max_index: int = 6,
    ) -> None:
        if progress:
            progress(0.0, "Scanning webcams 0-5...")
        try:
            import cv2
            tried: List[str] = []
            for idx in range(max_index):
                if progress:
                    progress(idx / max(1, max_index),
                             f"Probing camera #{idx}...")
                cap = None
                try:
                    if sys.platform.startswith("win"):
                        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(idx)
                    if not cap or not cap.isOpened():
                        tried.append(f"#{idx}: not openable")
                        if cap:
                            cap.release()
                        continue
                    ok = False
                    for _ in range(10):
                        ok, _frame = cap.read()
                        if ok:
                            break
                        time.sleep(0.08)
                    cap.release()
                    if ok:
                        report.webcam_available = True
                        report.webcam_message = f"Camera #{idx} ready"
                        if progress:
                            progress(1.0, report.webcam_message)
                        return
                    tried.append(f"#{idx}: no frames")
                except Exception as e:
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    tried.append(f"#{idx}: {type(e).__name__}")
                    continue
            report.webcam_available = False
            report.webcam_message = (
                f"No working webcam found. Tried: {', '.join(tried) or 'nothing'}. "
                f"Close any app holding the camera (Zoom / Teams / OBS / Discord) "
                f"and click Retry, or click Skip to continue anyway."
            )
        except Exception as e:
            report.webcam_available = False
            report.webcam_message = f"{type(e).__name__}: {e}"


def _print_diagnostic(report: CheckReport) -> None:
    print("=" * 64)
    print(" Face Tracker Pro - Dependency Diagnostic")
    print("=" * 64)
    print(f" Python   : {report.python_version}")
    print(f" Platform : {report.platform}")
    print("-" * 64)
    print(f" {'Dependency':<24} {'Status':<10} {'Version'}")
    print("-" * 64)
    for s in report.statuses:
        status = "OK" if s.found else "MISSING"
        ver = s.version or "-"
        print(f" {s.name:<24} {status:<10} {ver}")
        if s.error:
            print(f"     -> {s.error}")
    print("-" * 64)
    if report.mediapipe_api:
        print(f" MediaPipe API in use : {report.mediapipe_api}")
    elif report.mediapipe_api_error:
        print(" MediaPipe API in use : NONE")
        for line in report.mediapipe_api_error.splitlines():
            print(f"     {line}")
    if report.facemesh_loadable is True:
        print(" Face landmarker load : OK")
    elif report.facemesh_loadable is False:
        print(f" Face landmarker load : FAILED ({report.facemesh_error})")
    if report.model_path:
        print(f" Model file           : {report.model_path}")
    if report.webcam_available is True:
        print(f" Webcam               : OK ({report.webcam_message})")
    elif report.webcam_available is False:
        print(f" Webcam               : MISSING ({report.webcam_message})")
    print("=" * 64)
    if report.missing:
        print(" To install only what's missing, run:")
        print(f"    {report.install_hint()}")
        print("=" * 64)


def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="facetrack.bootstrap")
    p.add_argument(
        "--check", action="store_true",
        help="Silent check. Exit 0 if all deps present, 1 if any missing.",
    )
    p.add_argument(
        "--diagnose", action="store_true",
        help="Print a full diagnostic report. Exit 0 only if all deps present.",
    )
    p.add_argument(
        "--list-missing-pip", action="store_true",
        help="Print missing pip package names (one per line). Always exits 0.",
    )
    p.add_argument(
        "--with-facemesh", action="store_true",
        help="Also try to actually load the MediaPipe FaceMesh model.",
    )
    p.add_argument(
        "--with-webcam", action="store_true",
        help="Also check if webcam #0 is available.",
    )
    args = p.parse_args(argv)

    checker = DependencyChecker()
    report = checker.check_all()
    if args.with_facemesh and report.all_required_ok:
        checker.try_load_facemesh(report)
    if args.with_webcam:
        checker.try_open_webcam(report)

    if args.list_missing_pip:
        for name in report.missing_pip_names():
            print(name)
        return 0

    if args.diagnose:
        _print_diagnostic(report)
        if not report.all_required_ok:
            return 1
        if report.facemesh_loadable is False:
            return 2
        return 0

    if args.check:
        return 0 if report.all_required_ok else 1

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
