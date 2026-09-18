#!/usr/bin/env python3
"""Interactive manual tuning of Veo RAW rectification parameters."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import LensMetadata, VeoMetadata, load_veo_metadata


WINDOW_PREVIEW = "Veo rectification preview"
WINDOW_CONTROLS = "Veo rectification controls"
WINDOW_STATUS = "Veo rectification status"
CALIBRATION_WIDTH = 3840.0
CALIBRATION_HEIGHT = 1920.0
TRACKBAR_SCALE = 1000
OFFSET_RANGE = 500
PP_RANGE = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked", required=True, type=Path, help="LEFT-over-RIGHT stacked RAW frame")
    parser.add_argument("--calibration", required=True, type=Path, help="Veo .veo calibration file")
    parser.add_argument("--annotations", type=Path, default=None, help="Optional read-only annotation JSON")
    parser.add_argument("--out", type=Path, default=Path("output/tune_veo_rectification"), help="Output directory")
    parser.add_argument("--view", choices=("left", "right", "both"), default="both")
    return parser.parse_args()


def split_stacked(path: Path) -> dict[str, np.ndarray]:
    stacked = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if stacked is None:
        raise RuntimeError(f"Could not read stacked frame: {path}")
    height, _ = stacked.shape[:2]
    if height % 2:
        raise ValueError(f"Stacked frame height must be even, got {height}")
    split = height // 2
    return {"left": stacked[:split].copy(), "right": stacked[split:].copy()}


def load_annotations(path: Path | None) -> dict[str, list[dict]] | None:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as source_file:
            document = json.load(source_file)
        lines = document.get("annotations", {}).get("lines", {})
        result = {"left": [], "right": []}
        for lens in ("left", "right"):
            for label, points in lines.get(lens, {}).items():
                points = np.asarray(points, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                    continue
                result[lens].append({"label": label, "points": points})
        return result
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        print(f"Annotation warning: could not load {path}: {exc}")
        return None


def default_settings(metadata: VeoMetadata, images: dict[str, np.ndarray]) -> dict:
    raw_height, raw_width = images["left"].shape[:2]
    scale_x = CALIBRATION_WIDTH / raw_width
    scale_y = CALIBRATION_HEIGHT / raw_height
    fx_raw = metadata.left.intrinsic[0, 0] / scale_x
    hfov = math.degrees(2.0 * math.atan(raw_width / (2.0 * fx_raw)))
    # The 3840x1920 hypothesis is the current physically plausible starting
    # point for an exported 2048x1024 lens image; all values remain tunable.
    return {
        "raw_to_calib_scale_x": scale_x,
        "raw_to_calib_scale_y": scale_y,
        "raw_to_calib_offset_x": 0.0,
        "raw_to_calib_offset_y": 0.0,
        "rectified_hfov_deg": float(np.clip(hfov, 60.0, 140.0)),
        "distortion_gain": 1.0,
        "principal_delta_left_x": 0.0,
        "principal_delta_left_y": 0.0,
        "principal_delta_right_x": 0.0,
        "principal_delta_right_y": 0.0,
    }


def apply_distortion_gain(distortion: np.ndarray, gain: float) -> np.ndarray:
    result = np.asarray(distortion, dtype=np.float64).copy()
    # OpenCV's first two and fifth-or-later radial terms are scaled. Tangential
    # terms p1/p2 at indices 2/3 remain exactly as supplied by Veo.
    for index in (0, 1, 4, 5, 6):
        if index < result.size:
            result[index] *= gain
    return result


def mapped_intrinsics(
    lens: LensMetadata,
    settings: dict,
    lens_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = lens.intrinsic.astype(np.float64).copy()
    camera_matrix[0, 2] += settings[f"principal_delta_{lens_name}_x"]
    camera_matrix[1, 2] += settings[f"principal_delta_{lens_name}_y"]
    distortion = apply_distortion_gain(lens.distortion, settings["distortion_gain"])
    return camera_matrix, distortion


def rectified_intrinsics(raw_size: tuple[int, int], hfov_deg: float) -> np.ndarray:
    raw_width, raw_height = raw_size
    focal = raw_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    # Keep the output image dimensions and principal point fixed while tuning.
    return np.array(
        [[focal, 0.0, raw_width / 2.0], [0.0, focal, raw_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def raw_to_calibration(points: np.ndarray, settings: dict) -> np.ndarray:
    mapped = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    mapped[:, 0] = mapped[:, 0] * settings["raw_to_calib_scale_x"] + settings["raw_to_calib_offset_x"]
    mapped[:, 1] = mapped[:, 1] * settings["raw_to_calib_scale_y"] + settings["raw_to_calib_offset_y"]
    return mapped


def rectification_maps(
    image: np.ndarray,
    lens: LensMetadata,
    settings: dict,
    lens_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    source_intrinsic, distortion = mapped_intrinsics(lens, settings, lens_name)
    output_intrinsic = rectified_intrinsics((width, height), settings["rectified_hfov_deg"])
    calibration_x, calibration_y = cv2.initUndistortRectifyMap(
        source_intrinsic,
        distortion,
        np.eye(3, dtype=np.float64),
        output_intrinsic,
        (width, height),
        cv2.CV_32FC1,
    )
    raw_x = (calibration_x - settings["raw_to_calib_offset_x"]) / settings["raw_to_calib_scale_x"]
    raw_y = (calibration_y - settings["raw_to_calib_offset_y"]) / settings["raw_to_calib_scale_y"]
    valid = (raw_x >= 0) & (raw_x < width - 1) & (raw_y >= 0) & (raw_y < height - 1)
    return raw_x.astype(np.float32), raw_y.astype(np.float32), valid, output_intrinsic


def rectify_image(image: np.ndarray, lens: LensMetadata, settings: dict, lens_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map_x, map_y, valid, output_intrinsic = rectification_maps(image, lens, settings, lens_name)
    rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified[~valid] = 0
    return rectified, valid, output_intrinsic


def rectify_points(points: np.ndarray, lens: LensMetadata, settings: dict, lens_name: str, output_intrinsic: np.ndarray) -> np.ndarray:
    source_intrinsic, distortion = mapped_intrinsics(lens, settings, lens_name)
    calibration_points = raw_to_calibration(points, settings)
    return cv2.undistortPoints(
        calibration_points.reshape(-1, 1, 2),
        source_intrinsic,
        distortion,
        P=output_intrinsic,
    ).reshape(-1, 2)


def line_score(points: np.ndarray) -> dict:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    residuals = centered @ normal
    span = float(np.ptp(centered @ direction))
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    return {
        "count": int(len(points)),
        "rms_px": rms,
        "normalized_rms": rms / max(span, 1.0),
        "span_px": span,
        "max_abs_px": float(np.max(np.abs(residuals))),
    }


def score_annotations(
    annotations: dict[str, list[dict]] | None,
    lens_name: str,
    lens: LensMetadata,
    settings: dict,
    output_intrinsic: np.ndarray,
) -> tuple[dict[str, dict], float | None, dict[str, np.ndarray]]:
    if annotations is None:
        return {}, None, {}
    metrics = {}
    transformed = {}
    for item in annotations.get(lens_name, []):
        points = rectify_points(item["points"], lens, settings, lens_name, output_intrinsic)
        transformed[item["label"]] = points
        metrics[item["label"]] = line_score(points)
    if not metrics:
        return {}, None, transformed
    return metrics, float(np.mean([item["normalized_rms"] for item in metrics.values()])), transformed


def draw_line_overlays(image: np.ndarray, lines: dict[str, np.ndarray], color_offset: int = 0) -> np.ndarray:
    overlay = image.copy()
    colors = ((0, 220, 255), (0, 255, 0), (255, 180, 0), (255, 0, 255), (0, 130, 255))
    for index, (label, points) in enumerate(lines.items()):
        color = colors[(index + color_offset) % len(colors)]
        for point in points:
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 4, color, -1, cv2.LINE_AA)
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        direction = vh[0]
        projection = (points - center) @ direction
        start = center + direction * projection.min()
        end = center + direction * projection.max()
        cv2.line(overlay, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), color, 2, cv2.LINE_AA)
        cv2.putText(overlay, label, tuple(np.round(start).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


def draw_raw_overlays(image: np.ndarray, annotations: dict[str, list[dict]] | None, lens_name: str) -> np.ndarray:
    if annotations is None:
        return image.copy()
    lines = {item["label"]: item["points"] for item in annotations.get(lens_name, [])}
    return draw_line_overlays(image, lines)


def make_status(settings: dict, metrics: dict[str, dict], aggregate: float | None, warnings: list[str]) -> np.ndarray:
    status = np.full((320, 980, 3), 28, dtype=np.uint8)
    rows = [
        f"scale RAW->calib: x={settings['raw_to_calib_scale_x']:.5f} y={settings['raw_to_calib_scale_y']:.5f}",
        f"offset RAW->calib: x={settings['raw_to_calib_offset_x']:.1f} y={settings['raw_to_calib_offset_y']:.1f}",
        f"rectified HFOV: {settings['rectified_hfov_deg']:.2f} deg | distortion gain: {settings['distortion_gain']:.3f}",
        f"principal delta LEFT: ({settings['principal_delta_left_x']:.1f}, {settings['principal_delta_left_y']:.1f})",
        f"principal delta RIGHT: ({settings['principal_delta_right_x']:.1f}, {settings['principal_delta_right_y']:.1f})",
        f"aggregate normalized RMS: {aggregate:.6f}" if aggregate is not None else "aggregate normalized RMS: unavailable",
    ]
    for row, text in enumerate(rows):
        cv2.putText(status, text, (12, 24 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    y = 202
    for label, item in metrics.items():
        cv2.putText(status, f"{label}: rms={item['rms_px']:.2f}px norm={item['normalized_rms']:.6f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        y += 20
        if y > 310:
            break
    for warning in warnings[:2]:
        cv2.putText(status, warning[:105], (410, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
    return status


def compose_preview(images: dict[str, np.ndarray], rectified: dict[str, np.ndarray], view: str) -> np.ndarray:
    lenses = ("left", "right") if view == "both" else (view,)
    tiles = []
    for lens_name in lenses:
        raw = images[lens_name].copy()
        corrected = rectified[lens_name].copy()
        cv2.putText(raw, f"{lens_name.upper()} RAW", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(corrected, f"{lens_name.upper()} RECTIFIED", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.hstack((raw, corrected)))
    return np.vstack(tiles) if len(tiles) == 2 else tiles[0]


def settings_snapshot(settings: dict, args: argparse.Namespace, metadata: VeoMetadata) -> dict:
    return {
        "stacked": str(args.stacked),
        "calibration": str(args.calibration),
        "annotations": str(args.annotations) if args.annotations else None,
        "field": {"length": metadata.field_length, "width": metadata.field_width},
        "settings": {key: float(value) for key, value in settings.items()},
    }


def save_outputs(
    out_dir: Path,
    settings: dict,
    args: argparse.Namespace,
    metadata: VeoMetadata,
    rectified: dict[str, np.ndarray],
    metrics: dict[str, dict[str, dict]],
    aggregates: dict[str, float | None],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "current_settings.json").write_text(json.dumps(settings_snapshot(settings, args, metadata), indent=2) + "\n", encoding="utf-8")
    for lens_name, image in rectified.items():
        cv2.imwrite(str(out_dir / f"{lens_name}_rectified.jpg"), image)
    if args.annotations is not None:
        summary = {"annotations": str(args.annotations), "metrics": metrics, "aggregate_normalized_rms": aggregates}
        (out_dir / "annotation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_metrics(metrics, aggregates)


def print_metrics(metrics: dict[str, dict[str, dict]], aggregates: dict[str, float | None]) -> None:
    for lens_name in ("left", "right"):
        if lens_name not in metrics:
            continue
        print(f"{lens_name.upper()} annotation scores:")
        for label, item in metrics[lens_name].items():
            print(f"  {label}: RMS={item['rms_px']:.3f}px normalized={item['normalized_rms']:.6f}")
        print(f"  aggregate normalized RMS: {aggregates[lens_name]}")


def make_trackbars(settings: dict) -> None:
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("scale x", WINDOW_CONTROLS, int(round(settings["raw_to_calib_scale_x"] * TRACKBAR_SCALE)), 3000, lambda _: None)
    cv2.createTrackbar("scale y", WINDOW_CONTROLS, int(round(settings["raw_to_calib_scale_y"] * TRACKBAR_SCALE)), 3000, lambda _: None)
    cv2.createTrackbar("offset x", WINDOW_CONTROLS, int(round(settings["raw_to_calib_offset_x"] + OFFSET_RANGE)), OFFSET_RANGE * 2, lambda _: None)
    cv2.createTrackbar("offset y", WINDOW_CONTROLS, int(round(settings["raw_to_calib_offset_y"] + OFFSET_RANGE)), OFFSET_RANGE * 2, lambda _: None)
    cv2.createTrackbar("HFOV deg", WINDOW_CONTROLS, int(round(settings["rectified_hfov_deg"])), 160, lambda _: None)
    cv2.createTrackbar("dist gain x100", WINDOW_CONTROLS, int(round(settings["distortion_gain"] * 100)), 300, lambda _: None)
    for lens_name in ("left", "right"):
        cv2.createTrackbar(f"{lens_name} pp dx", WINDOW_CONTROLS, int(round(settings[f"principal_delta_{lens_name}_x"] + PP_RANGE)), PP_RANGE * 2, lambda _: None)
        cv2.createTrackbar(f"{lens_name} pp dy", WINDOW_CONTROLS, int(round(settings[f"principal_delta_{lens_name}_y"] + PP_RANGE)), PP_RANGE * 2, lambda _: None)


def read_trackbars() -> dict:
    return {
        "raw_to_calib_scale_x": cv2.getTrackbarPos("scale x", WINDOW_CONTROLS) / TRACKBAR_SCALE,
        "raw_to_calib_scale_y": cv2.getTrackbarPos("scale y", WINDOW_CONTROLS) / TRACKBAR_SCALE,
        "raw_to_calib_offset_x": cv2.getTrackbarPos("offset x", WINDOW_CONTROLS) - OFFSET_RANGE,
        "raw_to_calib_offset_y": cv2.getTrackbarPos("offset y", WINDOW_CONTROLS) - OFFSET_RANGE,
        "rectified_hfov_deg": max(30.0, float(cv2.getTrackbarPos("HFOV deg", WINDOW_CONTROLS))),
        "distortion_gain": cv2.getTrackbarPos("dist gain x100", WINDOW_CONTROLS) / 100.0,
        "principal_delta_left_x": cv2.getTrackbarPos("left pp dx", WINDOW_CONTROLS) - PP_RANGE,
        "principal_delta_left_y": cv2.getTrackbarPos("left pp dy", WINDOW_CONTROLS) - PP_RANGE,
        "principal_delta_right_x": cv2.getTrackbarPos("right pp dx", WINDOW_CONTROLS) - PP_RANGE,
        "principal_delta_right_y": cv2.getTrackbarPos("right pp dy", WINDOW_CONTROLS) - PP_RANGE,
    }


def reset_trackbars(settings: dict) -> None:
    values = {
        "scale x": round(settings["raw_to_calib_scale_x"] * TRACKBAR_SCALE),
        "scale y": round(settings["raw_to_calib_scale_y"] * TRACKBAR_SCALE),
        "offset x": round(settings["raw_to_calib_offset_x"] + OFFSET_RANGE),
        "offset y": round(settings["raw_to_calib_offset_y"] + OFFSET_RANGE),
        "HFOV deg": round(settings["rectified_hfov_deg"]),
        "dist gain x100": round(settings["distortion_gain"] * 100),
        "left pp dx": round(settings["principal_delta_left_x"] + PP_RANGE),
        "left pp dy": round(settings["principal_delta_left_y"] + PP_RANGE),
        "right pp dx": round(settings["principal_delta_right_x"] + PP_RANGE),
        "right pp dy": round(settings["principal_delta_right_y"] + PP_RANGE),
    }
    for name, value in values.items():
        cv2.setTrackbarPos(name, WINDOW_CONTROLS, int(value))


def main() -> None:
    args = parse_args()
    images = split_stacked(args.stacked)
    metadata = load_veo_metadata(args.calibration)
    annotations = load_annotations(args.annotations)
    settings = default_settings(metadata, images)
    make_trackbars(settings)
    cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_STATUS, cv2.WINDOW_NORMAL)
    print("Interactive tuning started. Press S to save, W to write previews, R to reset, Q/Escape to quit.")
    last_settings = None
    print_next_metrics = True
    cached = None
    while True:
        current = read_trackbars()
        if last_settings != current:
            rectified = {}
            raw_overlays = {}
            rectified_overlays = {}
            metrics = {}
            aggregates = {}
            for lens_name in ("left", "right"):
                lens = metadata.lens(lens_name)
                corrected, _, output_intrinsic = rectify_image(images[lens_name], lens, current, lens_name)
                line_metrics, aggregate, transformed = score_annotations(annotations, lens_name, lens, current, output_intrinsic)
                rectified[lens_name] = corrected
                raw_overlays[lens_name] = draw_raw_overlays(images[lens_name], annotations, lens_name)
                rectified_overlays[lens_name] = draw_line_overlays(corrected, transformed)
                metrics[lens_name] = line_metrics
                aggregates[lens_name] = aggregate
            display_rectified = {lens: rectified_overlays[lens] for lens in rectified}
            preview_images = {lens: raw_overlays[lens] for lens in raw_overlays}
            preview = compose_preview(preview_images, display_rectified, args.view)
            if args.view != "both":
                status_metrics = metrics.get(args.view, {})
            else:
                status_metrics = {
                    f"LEFT/{label}": item
                    for label, item in metrics.get("left", {}).items()
                }
                status_metrics.update(
                    {
                        f"RIGHT/{label}": item
                        for label, item in metrics.get("right", {}).items()
                    }
                )
            aggregate_values = [value for value in aggregates.values() if value is not None]
            status = make_status(current, status_metrics, float(np.mean(aggregate_values)) if aggregate_values else None, [])
            cached = (current, rectified, metrics, aggregates, preview, status)
            last_settings = current.copy()
            if print_next_metrics:
                # Print the initial score and scores after a reset without
                # flooding the terminal while a trackbar is being dragged.
                print_metrics(metrics, aggregates)
                print_next_metrics = False
            cv2.imshow(WINDOW_PREVIEW, preview)
            cv2.imshow(WINDOW_STATUS, status)

        key = cv2.waitKeyEx(30)
        low = key & 0xFF
        if low in (ord("q"), ord("Q"), 27):
            break
        if low in (ord("r"), ord("R")):
            reset_trackbars(settings)
            last_settings = None
            print_next_metrics = True
        elif low in (ord("w"), ord("W")) and cached is not None:
            _, rectified, _, _, _, _ = cached
            args.out.mkdir(parents=True, exist_ok=True)
            for lens_name, image in rectified.items():
                cv2.imwrite(str(args.out / f"{lens_name}_rectified.jpg"), image)
            print(f"Saved rectified previews to {args.out}")
        elif low in (ord("s"), ord("S")) and cached is not None:
            current, rectified, metrics, aggregates, _, _ = cached
            save_outputs(args.out, current, args, metadata, rectified, metrics, aggregates)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
