"""Local Flask UI for validating Veo panorama pixels against the pitch plane."""

from __future__ import annotations

import os
import io
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.panorama_ground import ground_intersection_for_pixel
from analysis.reproduce_veo_panorama import render
from analysis.veo_metadata import load_veo_metadata


RENDER_CACHE: dict[str, dict] = {}
RESOLUTIONS = {"1920x1080": (1920, 1080), "3840x2160": (3840, 2160)}

app = Flask(__name__)
app.config.update(SECRET_KEY=os.urandom(32), MAX_CONTENT_LENGTH=32 * 1024 * 1024)


def available_calibrations() -> list[Path]:
    return sorted((ROOT / "data").glob("camera_alignment*.veo"))


def session_id() -> str:
    if "validation_session_id" not in session:
        session["validation_session_id"] = uuid.uuid4().hex
    return session["validation_session_id"]


def load_uploaded_calibration(upload) -> tuple[object, str]:
    """Pass an uploaded .veo file through the canonical metadata reader."""
    with tempfile.NamedTemporaryFile(suffix=".veo", delete=False) as temporary_file:
        temporary_file.write(upload.read())
        temporary_path = Path(temporary_file.name)
    try:
        return load_veo_metadata(temporary_path), secure_filename(upload.filename)
    finally:
        temporary_path.unlink(missing_ok=True)


def decode_frame(upload) -> np.ndarray:
    encoded = np.frombuffer(upload.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The uploaded frame is not a readable JPG or PNG.")
    if frame.shape[:2] != (2048, 2048):
        raise ValueError(f"Expected a 2048x2048 stacked Veo frame, got {frame.shape[1]}x{frame.shape[0]}.")
    return frame


def selected_metadata() -> tuple[object, str]:
    source = request.form.get("calibration_source", "repository")
    if source == "upload":
        upload = request.files.get("calibration_upload")
        if upload is None or not upload.filename:
            raise ValueError("Choose a .veo calibration upload.")
        return load_uploaded_calibration(upload)

    selected = request.form.get("calibration_file", "")
    path = ROOT / "data" / selected
    if path not in available_calibrations():
        raise ValueError("Choose one of the repository calibration files.")
    return load_veo_metadata(path), path.name


def current_render() -> dict | None:
    render_id = session.get("render_id")
    return RENDER_CACHE.get(render_id) if render_id else None


@app.get("/")
def index():
    cached = current_render()
    context = None
    if cached is not None:
        context = {
            "render_id": cached["render_id"],
            "width": cached["width"],
            "height": cached["height"],
            "field_length": cached["metadata"].field_length,
            "field_width": cached["metadata"].field_width,
            "calibration_name": cached["calibration_name"],
            "frame_name": cached["frame_name"],
        }
    return render_template(
        "index.html",
        calibrations=available_calibrations(),
        resolutions=RESOLUTIONS,
        context=context,
        error=request.args.get("error"),
    )


@app.post("/render")
def create_render():
    frame_upload = request.files.get("frame")
    if frame_upload is None or not frame_upload.filename:
        return index_with_error("Upload one 2048x2048 stacked Veo frame.")
    try:
        frame = decode_frame(frame_upload)
        metadata, calibration_name = selected_metadata()
        resolution_name = request.form.get("resolution", "1920x1080")
        width, height = RESOLUTIONS[resolution_name]
        result = render(frame, metadata, width, height)
        success, encoded = cv2.imencode(".png", result["panorama"])
        if not success:
            raise RuntimeError("Could not encode the rendered panorama.")
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        return index_with_error(str(exc))

    render_id = uuid.uuid4().hex
    RENDER_CACHE[render_id] = {
        "render_id": render_id,
        "png": encoded.tobytes(),
        "width": width,
        "height": height,
        "metadata": metadata,
        "calibration_name": calibration_name,
        "frame_name": secure_filename(frame_upload.filename),
    }
    session["render_id"] = render_id
    return index()


def index_with_error(error: str):
    return render_template(
        "index.html",
        calibrations=available_calibrations(),
        resolutions=RESOLUTIONS,
        context=None,
        error=error,
    ), 400


@app.get("/panorama/<render_id>.png")
def panorama_image(render_id: str):
    cached = current_render()
    if cached is None or cached["render_id"] != render_id:
        return "No current panorama render.", 404
    return send_file(
        io.BytesIO(cached["png"]),
        mimetype="image/png",
        download_name="veo_panorama_validation.png",
    )


@app.post("/api/click")
def click_to_ground():
    cached = current_render()
    if cached is None:
        return jsonify(error="Render a panorama before selecting points."), 409
    payload = request.get_json(silent=True) or {}
    try:
        pixel_x = int(payload["x"])
        pixel_y = int(payload["y"])
        point = ground_intersection_for_pixel(
            cached["metadata"], pixel_x, pixel_y, cached["width"], cached["height"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(point.to_dict())


if __name__ == "__main__":
    app.run(debug=True)
