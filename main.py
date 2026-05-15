from __future__ import annotations

import sys


def _show_console_error(title: str, msg: str) -> None:
    sys.stderr.write(f"\n[ {title} ]\n{msg}\n")


def _show_messagebox_error(title: str, msg: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
        existing = tk._default_root  # type: ignore[attr-defined]
        if existing is not None:
            messagebox.showerror(title, msg, parent=existing)
            return
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, msg)
        root.destroy()
    except Exception:
        _show_console_error(title, msg)


def _preflight() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception as e:
        _show_console_error(
            "Face Tracker Pro - tkinter missing",
            f"Python's tkinter module is not available: {e}\n\n"
            "On Windows: reinstall Python and ensure 'tcl/tk and IDLE' is "
            "checked in the installer.\n"
            "On Linux:   sudo apt install python3-tk",
        )
        return False
    try:
        import customtkinter  # noqa: F401
    except Exception as e:
        _show_messagebox_error(
            "Face Tracker Pro - customtkinter missing",
            f"customtkinter is not installed.\n\n"
            f"Run this command, then re-launch:\n\n"
            f"    python -m pip install customtkinter\n\n"
            f"Details: {e}",
        )
        return False
    return True


def _build_boot_steps(splash) -> None:
    from facetrack.bootstrap import DependencyChecker, CheckReport

    def step_check_deps(progress_cb, shared):
        progress_cb(0.0, "Importing modules...")
        checker = DependencyChecker()
        report = checker.check_all(progress=progress_cb)
        shared["dep_report"] = report
        if report.missing:
            missing = ", ".join(s.name for s in report.missing)
            raise RuntimeError(
                f"Missing dependencies: {missing}\n\n"
                f"Fix with:\n{report.install_hint()}"
            )
        if report.mediapipe_api is None:
            raise RuntimeError(report.mediapipe_api_error or "MediaPipe API not available")
        progress_cb(1.0, f"All deps OK. MediaPipe API: {report.mediapipe_api}")
        return report

    def step_fetch_model(progress_cb, shared):
        report: CheckReport = shared.get("dep_report")
        if report and report.mediapipe_api == "solutions":
            progress_cb(1.0, "Using bundled FaceMesh model (no download needed)")
            return None
        progress_cb(0.0, "Locating face landmarker model...")
        from facetrack.model import existing_model_path, download_model, ensure_model
        existing = existing_model_path()
        if existing is not None:
            progress_cb(1.0, f"Using cached model at {existing}")
            return str(existing)
        progress_cb(0.0, "Downloading face landmarker model (~5 MB)...")
        path = ensure_model(progress_cb=progress_cb)
        return str(path)

    def step_check_camera(progress_cb, shared):
        progress_cb(0.0, "Scanning camera indices 0-5...")
        report: CheckReport = shared.get("dep_report")
        if report is None:
            from facetrack.bootstrap import CheckReport as _CR
            report = _CR(python_version="", platform="")
            shared["dep_report"] = report
        try:
            DependencyChecker.try_open_webcam(report, progress=progress_cb)
        except Exception as e:
            raise RuntimeError(
                f"Camera probe crashed: {type(e).__name__}: {e}\n\n"
                "You can still continue without a camera by clicking Skip."
            )
        if not report.webcam_available:
            raise RuntimeError(report.webcam_message
                               or "No working webcam detected.")
        progress_cb(1.0, report.webcam_message)
        return True

    def step_load_facemesh(progress_cb, shared):
        progress_cb(0.1, "Loading face landmarker...")
        from facetrack.tracker import FaceTracker
        model_path = shared.get("model")
        tracker = FaceTracker(
            max_faces=2, refine_landmarks=True,
            model_path=model_path, progress_cb=progress_cb,
        )
        progress_cb(1.0, f"Loaded ({tracker.backend} backend)")
        return tracker

    def step_build_analyzer(progress_cb, shared):
        progress_cb(0.2, "Building geometry analyzer...")
        from facetrack.analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        progress_cb(1.0, "Analyzer ready")
        return analyzer

    splash.add_step("deps",     "Checking dependencies",      step_check_deps,    weight=1.0)
    splash.add_step("model",    "Fetching face landmarker model", step_fetch_model, weight=2.0)
    splash.add_step("camera",   "Detecting camera",           step_check_camera,  weight=1.0,
                    skippable=True)
    splash.add_step("facemesh", "Loading face landmarker",    step_load_facemesh, weight=2.5)
    splash.add_step("analyzer", "Preparing analyzer",         step_build_analyzer, weight=0.5)


def main() -> int:
    if not _preflight():
        return 1

    from facetrack.splash import SplashScreen
    splash = SplashScreen(title="Face Tracker Pro")
    _build_boot_steps(splash)
    result = splash.run()

    if result.cancelled:
        return 0
    if not result.ok:
        sys.stderr.write(
            f"[ Face Tracker Pro - Startup failed ]\n"
            f"Step: {result.failed_step}\n{result.error_message}\n"
        )
        return 1

    from facetrack.app import FaceTrackerApp
    tracker = result.shared.get("facemesh")
    analyzer = result.shared.get("analyzer")
    if tracker is None or analyzer is None:
        sys.stderr.write(
            "[ Face Tracker Pro ] WARNING: a boot step returned None - "
            "the main app will reinitialize the missing component (this is "
            "wasted work). Skipped steps: "
            f"{result.skipped_steps}\n"
        )
    app = FaceTrackerApp(tracker=tracker, analyzer=analyzer)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
