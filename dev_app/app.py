"""Minimal Streamlit development UI for Veo lens rectification tuning."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import VeoMetadata, load_veo_metadata
from tools.tune_veo_rectification import (
    default_settings,
    mapped_intrinsics,
    rectify_image,
)


OUTPUT_DIR = ROOT / "output" / "dev_app"
SETTINGS_PREFIX = "lens_configuration_"


def decode_uploaded_image(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> np.ndarray:
    """Decode an uploaded JPG or PNG into OpenCV's BGR image format."""
    encoded = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {uploaded_file.name}")
    return image


def split_stacked_image(stacked: np.ndarray) -> dict[str, np.ndarray]:
    """Split a LEFT-over-RIGHT stacked frame without assuming its dimensions."""
    height, _ = stacked.shape[:2]
    if height % 2:
        raise ValueError(f"Stacked frame height must be even, got {height}")
    half_height = height // 2
    return {"left": stacked[:half_height].copy(), "right": stacked[half_height:].copy()}


def load_stacked_path(path_text: str) -> tuple[dict[str, np.ndarray] | None, str | None, str | None]:
    """Load a repository-local stacked image path and return a readable error."""
    if not path_text.strip():
        return None, None, None
    source_path = Path(path_text).expanduser()
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None, f"Could not read stacked frame: {source_path}"
    try:
        return split_stacked_image(image), str(source_path), None
    except ValueError as exc:
        return None, None, str(exc)


def load_uploaded_calibration(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> VeoMetadata:
    """Use the retained metadata reader for an uploaded temporary .veo file."""
    with tempfile.NamedTemporaryFile(suffix=".veo", delete=False) as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        temporary_path = Path(temporary_file.name)
    try:
        return load_veo_metadata(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_selected_calibration(
    source: str,
    local_path: Path | None,
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None,
) -> tuple[VeoMetadata | None, str | None, str | None]:
    try:
        if source == "Repository file":
            if local_path is None:
                return None, None, "No .veo files were found under data/."
            return load_veo_metadata(local_path), str(local_path), None
        if uploaded_file is None:
            return None, None, "Upload a .veo calibration file."
        return load_uploaded_calibration(uploaded_file), uploaded_file.name, None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, None, f"Could not read calibration: {exc}"


def settings_key(name: str) -> str:
    return f"{SETTINGS_PREFIX}{name}"


def reset_settings(defaults: dict) -> None:
    for name, value in defaults.items():
        st.session_state[settings_key(name)] = float(value)


def current_settings() -> dict:
    return {
        name.removeprefix(SETTINGS_PREFIX): value
        for name, value in st.session_state.items()
        if name.startswith(SETTINGS_PREFIX)
    }


def source_token(frame_name: str, calibration_name: str) -> str:
    return f"{frame_name}|{calibration_name}"


def save_configuration(
    frame_name: str,
    calibration_name: str,
    lens_name: str,
    settings: dict,
    effective_intrinsic: np.ndarray,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_frame": frame_name,
        "calibration": calibration_name,
        "selected_lens": lens_name,
        "settings": {name: float(value) for name, value in settings.items()},
        "effective_intrinsic": effective_intrinsic.tolist(),
    }
    destination = OUTPUT_DIR / "lens_configuration.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def save_rectified_image(rectified: np.ndarray, lens_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"rectified_{lens_name}.png"
    if not cv2.imwrite(str(destination), rectified):
        raise RuntimeError(f"Could not write {destination}")
    return destination


def show_controls(defaults: dict, lens_name: str) -> dict:
    """Render only controls that are meaningful for the currently selected lens."""
    st.sidebar.subheader("RAW → calibration mapping")
    st.sidebar.number_input("scale_x", min_value=0.001, max_value=3.0, step=0.001, format="%.6f", key=settings_key("raw_to_calib_scale_x"))
    st.sidebar.number_input("scale_y", min_value=0.001, max_value=3.0, step=0.001, format="%.6f", key=settings_key("raw_to_calib_scale_y"))
    st.sidebar.number_input("offset_x", min_value=-500.0, max_value=500.0, step=1.0, format="%.3f", key=settings_key("raw_to_calib_offset_x"))
    st.sidebar.number_input("offset_y", min_value=-500.0, max_value=500.0, step=1.0, format="%.3f", key=settings_key("raw_to_calib_offset_y"))

    st.sidebar.subheader("Lens adjustment")
    st.sidebar.slider("distortion_gain", min_value=0.0, max_value=3.0, step=0.01, key=settings_key("distortion_gain"))
    st.sidebar.number_input("cx_shift", min_value=-250.0, max_value=250.0, step=0.5, format="%.3f", key=settings_key(f"principal_delta_{lens_name}_x"))
    st.sidebar.number_input("cy_shift", min_value=-250.0, max_value=250.0, step=0.5, format="%.3f", key=settings_key(f"principal_delta_{lens_name}_y"))

    st.sidebar.subheader("Virtual view")
    st.sidebar.slider("HFOV (degrees)", min_value=30.0, max_value=160.0, step=0.1, key=settings_key("rectified_hfov_deg"))

    if st.sidebar.button("Reset", use_container_width=True):
        reset_settings(defaults)
        st.rerun()
    return current_settings()


def main() -> None:
    st.set_page_config(page_title="Veo Calibration Lab", layout="wide")
    st.title("Veo Calibration Lab")
    st.caption("Lens configuration — a local development view for Veo RAW rectification.")

    st.sidebar.header("Inputs")
    local_frame_path = st.sidebar.text_input("Repository stacked-frame path", value="samples/match2_stacked_frame.jpg")
    uploaded_frame = st.sidebar.file_uploader("Or upload stacked JPG/PNG", type=("jpg", "jpeg", "png"))

    if uploaded_frame is not None:
        try:
            images = split_stacked_image(decode_uploaded_image(uploaded_frame))
            frame_name = uploaded_frame.name
            frame_error = None
        except ValueError as exc:
            images, frame_name, frame_error = None, None, str(exc)
    else:
        images, frame_name, frame_error = load_stacked_path(local_frame_path)

    calibration_paths = sorted((ROOT / "data").glob("*.veo"))
    calibration_source = st.sidebar.radio("Veo calibration", ("Repository file", "Upload .veo"), horizontal=True)
    selected_calibration = None
    uploaded_calibration = None
    if calibration_source == "Repository file":
        if calibration_paths:
            selected_calibration = st.sidebar.selectbox("Repository calibration", calibration_paths, format_func=lambda path: path.name)
    else:
        uploaded_calibration = st.sidebar.file_uploader("Upload .veo", type=("veo",))
    metadata, calibration_name, calibration_error = load_selected_calibration(
        calibration_source,
        selected_calibration,
        uploaded_calibration,
    )

    lens_label = st.sidebar.selectbox("Lens", ("LEFT", "RIGHT"))
    lens_name = lens_label.lower()

    if frame_error:
        st.error(frame_error)
    if calibration_error:
        st.error(calibration_error)
    if images is None or metadata is None or frame_name is None or calibration_name is None:
        st.info("Choose a valid stacked frame and Veo calibration to start tuning.")
        return

    defaults = default_settings(metadata, images)
    token = source_token(frame_name, calibration_name)
    if st.session_state.get("lens_configuration_source") != token:
        reset_settings(defaults)
        st.session_state["lens_configuration_source"] = token

    settings = show_controls(defaults, lens_name)
    lens = metadata.lens(lens_name)
    rectified, _, _ = rectify_image(images[lens_name], lens, settings, lens_name)
    effective_intrinsic, _ = mapped_intrinsics(lens, settings, lens_name)

    raw_column, rectified_column = st.columns(2, gap="medium")
    with raw_column:
        st.subheader(f"{lens_label} RAW")
        st.image(cv2.cvtColor(images[lens_name], cv2.COLOR_BGR2RGB), use_container_width=True)
    with rectified_column:
        st.subheader(f"{lens_label} RECTIFIED")
        st.image(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB), use_container_width=True)

    save_column, image_column = st.columns(2)
    with save_column:
        if st.button("Save configuration", type="primary"):
            destination = save_configuration(frame_name, calibration_name, lens_name, settings, effective_intrinsic)
            st.success(f"Saved {destination.relative_to(ROOT)}")
    with image_column:
        if st.button("Save rectified image"):
            try:
                destination = save_rectified_image(rectified, lens_name)
                st.success(f"Saved {destination.relative_to(ROOT)}")
            except RuntimeError as exc:
                st.error(str(exc))

    with st.expander("Development information"):
        st.write(f"RAW lens dimensions: {images[lens_name].shape[1]} × {images[lens_name].shape[0]} px")
        st.write(f"Calibration: {calibration_name}")
        st.write(f"Calibration version: {metadata.raw_json.get('calibration_version', 'not stated')}")
        st.write("Effective K")
        st.code(np.array2string(effective_intrinsic, precision=6), language=None)
        st.write("Original Veo distortion vector")
        st.code(np.array2string(lens.distortion, precision=8), language=None)
        st.write("Current tuning parameters")
        st.json({name: float(value) for name, value in settings.items()})


if __name__ == "__main__":
    main()
