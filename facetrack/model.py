from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

MODEL_FILENAME = "face_landmarker.task"

MODEL_URLS: List[str] = [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task",
]

MIN_MODEL_SIZE_BYTES = 100_000


def _candidate_paths() -> List[Path]:
    paths: List[Path] = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        paths.append(Path(sys._MEIPASS) / MODEL_FILENAME)
    try:
        exe_dir = Path(sys.argv[0]).resolve().parent
        paths.append(exe_dir / MODEL_FILENAME)
        paths.append(exe_dir.parent / MODEL_FILENAME)
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    paths.append(here / MODEL_FILENAME)
    paths.append(here.parent / MODEL_FILENAME)
    paths.append(cache_dir() / MODEL_FILENAME)
    seen, deduped = set(), []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            deduped.append(p)
    return deduped


def cache_dir() -> Path:
    return Path.home() / ".facetrack"


def existing_model_path() -> Optional[Path]:
    for p in _candidate_paths():
        try:
            if p.exists() and p.is_file() and p.stat().st_size >= MIN_MODEL_SIZE_BYTES:
                return p
        except OSError:
            continue
    return None


def download_model(
    progress_cb: Optional[Callable[[float, str], None]] = None,
    dest: Optional[Path] = None,
) -> Path:
    if dest is None:
        dest = cache_dir() / MODEL_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[Exception] = None

    for url in MODEL_URLS:
        if progress_cb:
            progress_cb(0.0, f"Connecting to model server...")
        try:
            def _hook(blocks: int, blocksize: int, total: int) -> None:
                if not progress_cb:
                    return
                if total > 0:
                    pct = min(1.0, (blocks * blocksize) / total)
                    mb_done = (blocks * blocksize) / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    progress_cb(pct, f"Downloading face model ({mb_done:.1f} / {mb_total:.1f} MB)")
                else:
                    progress_cb(0.0, f"Downloading face model... ({blocks * blocksize / 1024:.0f} KB)")

            urllib.request.urlretrieve(url, tmp, reporthook=_hook)
            if tmp.stat().st_size < MIN_MODEL_SIZE_BYTES:
                tmp.unlink(missing_ok=True)
                raise IOError(
                    f"Downloaded model is too small ({tmp.stat().st_size} bytes); "
                    f"server may have returned an error page."
                )
            if dest.exists():
                dest.unlink()
            tmp.rename(dest)
            if progress_cb:
                progress_cb(1.0, f"Model cached to {dest}")
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, OSError) as e:
            last_err = e
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if progress_cb:
                progress_cb(0.0, f"Mirror failed, trying next...")
            continue

    raise RuntimeError(
        f"Could not download face landmarker model from any mirror.\n"
        f"Last error: {last_err}\n\n"
        f"Manual fix: download this file with a browser:\n"
        f"  {MODEL_URLS[0]}\n"
        f"and save it as:\n"
        f"  {dest}"
    )


def ensure_model(progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
    found = existing_model_path()
    if found is not None:
        if progress_cb:
            progress_cb(1.0, f"Using cached model at {found}")
        return found
    if progress_cb:
        progress_cb(0.0, "Model not cached - downloading (~5 MB)")
    return download_model(progress_cb=progress_cb)
