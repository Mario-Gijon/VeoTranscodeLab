"""FastAPI service for the local Veo panorama-to-pitch validation app.

The backend intentionally owns rendering and every camera/ground calculation.
The browser receives results and visualizes them; it never recreates geometry.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .schemas import CreateMatchRequest, ExtractFrameRequest, GroundPointRequest, PathImportRequest, RenderRequest
from .services import (
    ROOT,
    calibration_summary,
    extract_frame,
    load_stacked_frame,
    media_record,
    repository_calibrations,
    save_upload,
    validate_calibration_path,
)
from .storage import WorkspaceStore

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.panorama_ground import ground_intersection_for_pixel
from analysis.panorama_landmarks import field_landmarks, landmark_validation
from analysis.reproduce_veo_panorama import render
from analysis.veo_metadata import load_veo_metadata


RUNTIME_ROOT = ROOT / "dev_app" / "runtime"
store = WorkspaceStore(RUNTIME_ROOT)
render_cache: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Veo Panorama Geometry Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def api_error(status: int, detail: str, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"message": detail, "code": code})


def require_match(match_id: str) -> dict[str, Any]:
    match = store.match(match_id)
    if match is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return match


def require_media(media_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result = store.media(media_id)
    if result is None:
        raise api_error(404, "Media item not found.", "media_not_found")
    return result


def with_calibration(match: dict[str, Any]):
    calibration = match.get("calibration")
    if not calibration:
        raise api_error(409, "Select a calibration before rendering.", "calibration_required")
    try:
        return load_veo_metadata(calibration["path"])
    except (OSError, KeyError, ValueError) as exc:
        raise api_error(422, f"Could not read the workspace calibration: {exc}", "invalid_calibration") from exc


def match_summary(match: dict[str, Any]) -> dict[str, Any]:
    return {**match, "media_count": len(match["media"])}


def public_media(media: dict[str, Any]) -> dict[str, Any]:
    return {**media, "url": f"/api/media/{media['id']}/content"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/calibrations")
def calibrations() -> list[dict[str, Any]]:
    return repository_calibrations()


@app.get("/api/matches")
def list_matches() -> list[dict[str, Any]]:
    return [match_summary(match) for match in store.matches()]


@app.post("/api/matches", status_code=201)
def create_match(payload: CreateMatchRequest) -> dict[str, Any]:
    calibration = None
    if payload.calibration_id:
        try:
            calibration = validate_calibration_path(payload.calibration_id, "repository")
        except (OSError, ValueError) as exc:
            raise api_error(422, str(exc), "invalid_calibration") from exc
    return match_summary(store.create_match(payload.name.strip(), payload.description.strip(), calibration))


@app.get("/api/matches/{match_id}")
def get_match(match_id: str) -> dict[str, Any]:
    match = require_match(match_id)
    return {**match_summary(match), "media": [public_media(item) for item in match["media"]]}


@app.post("/api/matches/{match_id}/calibration/path")
def import_calibration_path(match_id: str, payload: PathImportRequest) -> dict[str, Any]:
    try:
        calibration = validate_calibration_path(payload.path)
    except (OSError, ValueError) as exc:
        raise api_error(422, str(exc), "invalid_calibration") from exc
    match = store.update_match(match_id, lambda item: item.update(calibration=calibration))
    if match is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return match_summary(match)


@app.post("/api/matches/{match_id}/calibration/upload")
def upload_calibration(match_id: str, file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    require_match(match_id)
    if not file.filename or Path(file.filename).suffix.lower() != ".veo":
        raise api_error(422, "Upload a .veo calibration file.", "invalid_calibration")
    destination = RUNTIME_ROOT / "media" / match_id / f"calibration_{uuid.uuid4().hex[:8]}.veo"
    save_upload(file, destination)
    try:
        calibration = calibration_summary(destination, "upload")
    except (OSError, ValueError, KeyError) as exc:
        destination.unlink(missing_ok=True)
        raise api_error(422, f"Could not read the uploaded calibration: {exc}", "invalid_calibration") from exc
    match = store.update_match(match_id, lambda item: item.update(calibration=calibration))
    return match_summary(match) if match else require_match(match_id)


def add_media(match_id: str, record: dict[str, Any]) -> dict[str, Any]:
    record["id"] = store.new_id("media")
    match = store.update_match(match_id, lambda item: item["media"].append(record))
    if match is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return public_media(record)


@app.post("/api/matches/{match_id}/media/path", status_code=201)
def import_media_path(match_id: str, payload: PathImportRequest) -> dict[str, Any]:
    require_match(match_id)
    try:
        record = media_record(Path(payload.path), source="path")
    except (OSError, ValueError) as exc:
        raise api_error(422, str(exc), "invalid_media") from exc
    return add_media(match_id, record)


@app.post("/api/matches/{match_id}/media/upload", status_code=201)
def upload_media(match_id: str, file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    require_match(match_id)
    if not file.filename:
        raise api_error(422, "Choose a media file to upload.", "invalid_media")
    suffix = Path(file.filename).suffix.lower()
    destination = RUNTIME_ROOT / "media" / match_id / f"{uuid.uuid4().hex[:8]}{suffix}"
    save_upload(file, destination)
    try:
        record = media_record(destination, name=Path(file.filename).name, source="upload")
    except (OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise api_error(422, str(exc), "invalid_media") from exc
    return add_media(match_id, record)


@app.get("/api/media/{media_id}/content")
def media_content(media_id: str):
    _, media = require_media(media_id)
    path = Path(media["path"])
    if not path.is_file():
        raise api_error(404, "The referenced media file is no longer available.", "media_missing")
    return FileResponse(path, filename=media["name"])


@app.post("/api/media/frame", status_code=201)
def frame_from_video(payload: ExtractFrameRequest) -> dict[str, Any]:
    match, media = require_media(payload.media_id)
    if media["kind"] not in {"video", "clip"}:
        raise api_error(422, "Frame extraction requires video or clip media.", "not_video")
    destination = RUNTIME_ROOT / "frames" / match["id"] / f"frame_{uuid.uuid4().hex[:8]}.jpg"
    try:
        details = extract_frame(Path(media["path"]), payload.timestamp_s, destination)
        record = media_record(destination, name=f"{media['name']} @ {details['actual_time_s']:.3f}s", kind="frame", source="extracted")
        record["extracted_from"] = media["id"]
        record["timestamp_s"] = details["actual_time_s"]
    except (OSError, RuntimeError, ValueError) as exc:
        raise api_error(422, str(exc), "frame_extraction_failed") from exc
    return add_media(match["id"], record)


@app.post("/api/panorama/render", status_code=201)
def render_panorama(payload: RenderRequest) -> dict[str, Any]:
    match = require_match(payload.match_id)
    media_match, media = require_media(payload.media_id)
    if media_match["id"] != match["id"]:
        raise api_error(422, "Selected media belongs to another workspace.", "media_workspace_mismatch")
    if media["kind"] not in {"image", "frame"}:
        raise api_error(422, "Select a stacked image or extracted frame to render.", "render_requires_image")
    metadata = with_calibration(match)
    try:
        stacked = load_stacked_frame(Path(media["path"]))
        result = render(stacked, metadata, payload.width, payload.height)
        encoded_ok, encoded = cv2.imencode(".png", result["panorama"])
        if not encoded_ok:
            raise RuntimeError("Could not encode the panorama.")
    except (OSError, RuntimeError, ValueError) as exc:
        raise api_error(422, str(exc), "render_failed") from exc
    render_id = f"render_{uuid.uuid4().hex[:12]}"
    image_path = RUNTIME_ROOT / "cache" / f"{render_id}.png"
    image_path.write_bytes(encoded.tobytes())
    render_cache[render_id] = {
        "path": image_path,
        "metadata": metadata,
        "width": payload.width,
        "height": payload.height,
        "match_id": match["id"],
        "media_id": media["id"],
    }
    return {
        "id": render_id,
        "url": f"/api/renders/{render_id}/image",
        "width": payload.width,
        "height": payload.height,
        "field_length": metadata.field_length,
        "field_width": metadata.field_width,
        "landmarks": [value.to_dict() for value in field_landmarks(metadata.field_length, metadata.field_width).values()],
    }


@app.get("/api/renders/{render_id}/image")
def render_image(render_id: str):
    cached = render_cache.get(render_id)
    if cached is None or not cached["path"].is_file():
        raise api_error(404, "Rendered panorama is not available. Render it again.", "render_not_found")
    return FileResponse(cached["path"], media_type="image/png", filename="veo_panorama.png")


@app.post("/api/ground-point")
def ground_point(payload: GroundPointRequest) -> dict[str, Any]:
    cached = render_cache.get(payload.render_id)
    if cached is None:
        raise api_error(409, "Render a panorama before selecting points.", "render_required")
    if payload.x >= cached["width"] or payload.y >= cached["height"]:
        raise api_error(422, "Panorama pixel lies outside the active render.", "pixel_out_of_range")
    point = ground_intersection_for_pixel(cached["metadata"], payload.x, payload.y, cached["width"], cached["height"])
    response = point.to_dict()
    if payload.landmark_id:
        landmarks = field_landmarks(cached["metadata"].field_length, cached["metadata"].field_width)
        landmark = landmarks.get(payload.landmark_id)
        if landmark is None:
            raise api_error(422, "Unknown field landmark.", "unknown_landmark")
        response.update(landmark_validation(landmark, point))
    else:
        response.update({"landmark": None, "landmark_name": None, "expected_x": None, "expected_z": None,
                         "measured_x": point.world_x, "measured_z": point.world_z, "delta_x": None,
                         "delta_z": None, "field_error_m": None})
    return response
