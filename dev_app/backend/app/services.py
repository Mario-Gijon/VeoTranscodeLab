"""Filesystem, media, rendering, and authoritative geometry adapters."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import load_veo_metadata


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def repository_calibrations() -> list[dict[str, Any]]:
    return [calibration_summary(path, "repository") for path in sorted((ROOT / "data").glob("camera_alignment*.veo"))]


def calibration_summary(path: Path, source: str) -> dict[str, Any]:
    metadata = load_veo_metadata(path)
    return {
        "id": str(path.resolve()),
        "name": path.name,
        "path": str(path.resolve()),
        "source": source,
        "field_length": metadata.field_length,
        "field_width": metadata.field_width,
    }


def validate_calibration_path(value: str | Path, source: str = "path") -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".veo":
        raise ValueError("Calibration must be a readable .veo file.")
    return calibration_summary(path, source)


def infer_media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Supported media are JPG, PNG, WebP, MP4, MOV, MKV, AVI, and WebM.")


def inspect_media(path: Path, forced_kind: str | None = None) -> dict[str, Any]:
    kind = forced_kind or infer_media_kind(path)
    if kind in {"image", "frame"}:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The image could not be decoded by OpenCV.")
        height, width = image.shape[:2]
        return {"kind": kind, "width": int(width), "height": int(height), "fps": None, "duration_s": None}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The video could not be opened by OpenCV.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "kind": "clip" if kind == "clip" else "video",
        "width": width,
        "height": height,
        "fps": fps or None,
        "duration_s": frame_count / fps if fps else None,
        "frame_count": frame_count,
    }


def media_record(path: Path, *, name: str | None = None, kind: str | None = None, source: str = "path") -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError("The selected media path does not exist or is not a file.")
    return {
        "id": "",  # Assigned by the store-facing route.
        "name": name or path.name,
        "path": str(path),
        "source": source,
        **inspect_media(path, kind),
    }


def save_upload(upload, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        shutil.copyfileobj(upload.file, target)
    return destination


def extract_frame(video_path: Path, timestamp_s: float, output_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The video could not be opened for frame extraction.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    target = int(round(timestamp_s * fps)) if fps else 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    success, frame = capture.read()
    actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    capture.release()
    if not success or frame is None:
        raise ValueError("Could not extract a frame at the requested time.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError("Could not write the extracted frame.")
    height, width = frame.shape[:2]
    return {"width": width, "height": height, "fps": fps or None, "actual_frame": actual, "actual_time_s": actual / fps if fps else timestamp_s}


def load_stacked_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The selected image could not be decoded by OpenCV.")
    if frame.shape[:2] != (2048, 2048):
        raise ValueError(f"A panorama render needs a 2048×2048 stacked frame; received {frame.shape[1]}×{frame.shape[0]}.")
    return frame
