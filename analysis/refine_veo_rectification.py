#!/usr/bin/env python3
"""Search RAW-to-Veo calibration mappings with fixed Veo distortion.

This is an evaluation-only analysis.  It maps annotated RAW line samples
through each candidate and scores their rectified straightness; annotations
never define a homography, warp, or fitted line transform for the image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import LensMetadata, load_veo_metadata


DEFAULT_CALIBRATION_WIDTH = 3840.0
DEFAULT_CALIBRATION_HEIGHT = 1920.0
MODELS = ("uniform", "anisotropic", "crop_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked", required=True, type=Path, help="LEFT-over-RIGHT RAW frame")
    parser.add_argument("--calibration", required=True, type=Path, help="Veo .veo metadata")
    parser.add_argument("--annotations", required=True, type=Path, help="Manual annotation JSON")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--calibration-width", type=float, default=DEFAULT_CALIBRATION_WIDTH)
    parser.add_argument("--calibration-height", type=float, default=DEFAULT_CALIBRATION_HEIGHT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--rectified-focal-scale", type=float, default=1.0)
    return parser.parse_args()


def load_lenses(path: Path) -> dict[str, np.ndarray]:
    stacked = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if stacked is None:
        raise RuntimeError(f"Could not read stacked frame: {path}")
    height, _ = stacked.shape[:2]
    if height % 2:
        raise ValueError("Stacked frame height must be divisible by two")
    split = height // 2
    return {"left": stacked[:split].copy(), "right": stacked[split:].copy()}


def load_line_annotations(path: Path) -> dict[str, list[dict]]:
    with path.open("r", encoding="utf-8") as source_file:
        document = json.load(source_file)
    result: dict[str, list[dict]] = {"left": [], "right": []}
    for lens in ("left", "right"):
        for label, points in document.get("annotations", {}).get("lines", {}).get(lens, {}).items():
            if len(points) >= 3:
                result[lens].append({"label": label, "points": np.asarray(points, dtype=np.float64)})
    if not result["left"] and not result["right"]:
        raise ValueError("No annotated straight line with at least three points was found")
    return result


def mapping(scale_x: float, scale_y: float, offset_x: float, offset_y: float, model: str, **extra) -> dict:
    return {
        "model": model,
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
        **extra,
    }


def uniform_candidates(raw_width: int, raw_height: int, calibration_width: float, calibration_height: float) -> list[dict]:
    nominal = (calibration_width / raw_width + calibration_height / raw_height) / 2.0
    # The bounds cover a moderate crop/resize mismatch without allowing the
    # distortion model to be repurposed into an arbitrary image warp.
    scales = np.linspace(nominal - 0.35, nominal + 0.35, 7)
    offsets = (-160.0, -80.0, 0.0, 80.0, 160.0)
    return [mapping(scale, scale, ox, oy, "uniform") for scale in scales for ox in offsets for oy in offsets]


def anisotropic_candidates(raw_width: int, raw_height: int, calibration_width: float, calibration_height: float) -> list[dict]:
    nominal_x, nominal_y = calibration_width / raw_width, calibration_height / raw_height
    scales_x = np.linspace(nominal_x - 0.25, nominal_x + 0.25, 5)
    scales_y = np.linspace(nominal_y - 0.25, nominal_y + 0.25, 5)
    offsets = (-120.0, 0.0, 120.0)
    return [
        mapping(sx, sy, ox, oy, "anisotropic")
        for sx in scales_x for sy in scales_y for ox in offsets for oy in offsets
    ]


def crop_aware_candidates(raw_width: int, raw_height: int, calibration_width: float, calibration_height: float) -> list[dict]:
    """Test centered-crop hypotheses from larger exported source canvases."""
    source_canvases = ((raw_width, raw_height), (2304, 1152), (2560, 1280))
    candidates: list[dict] = []
    for source_width, source_height in source_canvases:
        scale_x, scale_y = calibration_width / source_width, calibration_height / source_height
        base_crop_x = (source_width - raw_width) / 2.0
        base_crop_y = (source_height - raw_height) / 2.0
        for shift_x in (-64.0, 0.0, 64.0):
            for shift_y in (-64.0, 0.0, 64.0):
                candidates.append(mapping(
                    scale_x,
                    scale_y,
                    scale_x * (base_crop_x + shift_x),
                    scale_y * (base_crop_y + shift_y),
                    "crop_aware",
                    source_canvas=[source_width, source_height],
                    crop_shift_raw=[shift_x, shift_y],
                ))
    return candidates


def local_candidates(best: dict) -> list[dict]:
    """Small deterministic refinement around a coarse uniform/anisotropic result."""
    scales = (-0.025, 0.0, 0.025)
    offsets = (-20.0, 0.0, 20.0)
    candidates = []
    if best["model"] == "uniform":
        for delta_scale in scales:
            for delta_x in offsets:
                for delta_y in offsets:
                    candidates.append(mapping(
                        best["scale_x"] + delta_scale,
                        best["scale_y"] + delta_scale,
                        best["offset_x"] + delta_x,
                        best["offset_y"] + delta_y,
                        "uniform",
                    ))
    elif best["model"] == "anisotropic":
        for delta_x_scale in scales:
            for delta_y_scale in scales:
                for delta_x in offsets:
                    for delta_y in offsets:
                        candidates.append(mapping(
                            best["scale_x"] + delta_x_scale,
                            best["scale_y"] + delta_y_scale,
                            best["offset_x"] + delta_x,
                            best["offset_y"] + delta_y,
                            "anisotropic",
                        ))
    return candidates


def calibration_points(raw_points: np.ndarray, candidate: dict) -> np.ndarray:
    points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2).copy()
    points[:, 0] = candidate["scale_x"] * points[:, 0] + candidate["offset_x"]
    points[:, 1] = candidate["scale_y"] * points[:, 1] + candidate["offset_y"]
    return points


def rectified_intrinsics(lens: LensMetadata, raw_size: tuple[int, int], calibration_size: tuple[float, float], focal_scale: float) -> np.ndarray:
    raw_width, raw_height = raw_size
    calibration_width, calibration_height = calibration_size
    scale = min(raw_width / calibration_width, raw_height / calibration_height)
    focal = float(np.sqrt(lens.intrinsic[0, 0] * lens.intrinsic[1, 1])) * scale * focal_scale
    return np.array([[focal, 0.0, raw_width / 2.0], [0.0, focal, raw_height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rectify_points(raw_points: np.ndarray, candidate: dict, lens: LensMetadata, new_intrinsic: np.ndarray) -> np.ndarray:
    points = calibration_points(raw_points, candidate)
    return cv2.undistortPoints(points.reshape(-1, 1, 2), lens.intrinsic, lens.distortion, P=new_intrinsic).reshape(-1, 2)


def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    direction = vectors[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    residuals = (points - center) @ normal
    span = float(np.ptp((points - center) @ direction))
    return center, direction, residuals, span


def line_metrics(points: np.ndarray) -> dict:
    center, direction, residuals, span = fit_line(points)
    rms_px = float(np.sqrt(np.mean(residuals * residuals)))
    return {
        "count": int(len(points)),
        "rms_px": rms_px,
        "normalized_rms": rms_px / max(span, 1.0),
        "span_px": span,
        "max_abs_px": float(np.max(np.abs(residuals))),
        "center": center.tolist(),
        "direction": direction.tolist(),
    }


def evaluate(candidate: dict, lines: list[dict], lens: LensMetadata, new_intrinsic: np.ndarray) -> dict:
    per_line = {}
    scores = []
    transformed = {}
    for line in lines:
        points = rectify_points(line["points"], candidate, lens, new_intrinsic)
        metrics = line_metrics(points)
        per_line[line["label"]] = metrics
        transformed[line["label"]] = points
        scores.append(metrics["normalized_rms"])
    return {
        **candidate,
        "aggregate_score": float(np.mean(scores)),
        "per_line": per_line,
        "transformed": transformed,
    }


def serializable(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "transformed"}


def refine_family(candidates: list[dict], lines: list[dict], lens: LensMetadata, new_intrinsic: np.ndarray) -> list[dict]:
    results = [evaluate(candidate, lines, lens, new_intrinsic) for candidate in candidates]
    results.sort(key=lambda item: item["aggregate_score"])
    if results[0]["model"] in ("uniform", "anisotropic"):
        results.extend(evaluate(candidate, lines, lens, new_intrinsic) for candidate in local_candidates(results[0]))
        results.sort(key=lambda item: item["aggregate_score"])
    return results


def build_rectification_map(raw_shape: tuple[int, int], candidate: dict, lens: LensMetadata, new_intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_height, raw_width = raw_shape
    calibration_map_x, calibration_map_y = cv2.initUndistortRectifyMap(
        lens.intrinsic, lens.distortion, np.eye(3, dtype=np.float64), new_intrinsic,
        (raw_width, raw_height), cv2.CV_32FC1,
    )
    raw_map_x = (calibration_map_x - candidate["offset_x"]) / candidate["scale_x"]
    raw_map_y = (calibration_map_y - candidate["offset_y"]) / candidate["scale_y"]
    valid = ((raw_map_x >= 0) & (raw_map_x < raw_width - 1) & (raw_map_y >= 0) & (raw_map_y < raw_height - 1))
    return raw_map_x.astype(np.float32), raw_map_y.astype(np.float32), valid


def rectify_image(image: np.ndarray, candidate: dict, lens: LensMetadata, new_intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_x, map_y, valid = build_rectification_map(image.shape[:2], candidate, lens, new_intrinsic)
    rectified = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified[~valid] = 0
    return rectified, valid


def draw_overlay(image: np.ndarray, result: dict) -> np.ndarray:
    overlay = image.copy()
    colors = ((0, 220, 255), (0, 255, 0), (255, 180, 0), (255, 0, 255), (0, 120, 255))
    for index, (label, points) in enumerate(result["transformed"].items()):
        color = colors[index % len(colors)]
        center, direction, _, _ = fit_line(points)
        longitudinal = (points - center) @ direction
        start = center + direction * np.min(longitudinal)
        end = center + direction * np.max(longitudinal)
        cv2.line(overlay, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 4, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, label, tuple(np.round(start).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


def comparison_panel(images: dict[str, np.ndarray], family_bests: dict[str, dict[str, dict]]) -> np.ndarray:
    tiles = []
    for lens in ("left", "right"):
        row = []
        for model in MODELS:
            result = family_bests[lens].get(model)
            if result is None:
                tile = np.full((240, 480, 3), 20, dtype=np.uint8)
                cv2.putText(tile, f"{lens} / {model}: not tested", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                tile = cv2.resize(images[lens], (480, 240), interpolation=cv2.INTER_AREA)
                cv2.putText(tile, f"{lens.upper()} | {model} | score={result['aggregate_score']:.6f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            row.append(tile)
        tiles.append(np.hstack(row))
    return np.vstack(tiles)


def write_csv(path: Path, rankings: dict[str, list[dict]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["lens", "rank", "model", "aggregate_score", "scale_x", "scale_y", "offset_x", "offset_y"])
        writer.writeheader()
        for lens, results in rankings.items():
            for rank, result in enumerate(results, start=1):
                writer.writerow({"lens": lens, "rank": rank, **{key: result[key] for key in writer.fieldnames if key not in ("lens", "rank")}})


def main() -> None:
    args = parse_args()
    images = load_lenses(args.stacked)
    annotations = load_line_annotations(args.annotations)
    metadata = load_veo_metadata(args.calibration)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_height, raw_width = images["left"].shape[:2]
    if images["right"].shape[:2] != (raw_height, raw_width):
        raise ValueError("LEFT and RIGHT lens images must have equal dimensions")

    candidate_builders = {
        "uniform": uniform_candidates,
        "anisotropic": anisotropic_candidates,
        "crop_aware": crop_aware_candidates,
    }
    calibration_size = (args.calibration_width, args.calibration_height)
    rankings: dict[str, list[dict]] = {}
    family_bests: dict[str, dict[str, dict]] = {"left": {}, "right": {}}
    rectified_overlays: dict[str, np.ndarray] = {}
    report_lenses = {}

    for lens_name in ("left", "right"):
        lens = metadata.lens(lens_name)
        new_intrinsic = rectified_intrinsics(lens, (raw_width, raw_height), calibration_size, args.rectified_focal_scale)
        all_results = []
        for model in args.models:
            candidates = candidate_builders[model](raw_width, raw_height, *calibration_size)
            family_results = refine_family(candidates, annotations[lens_name], lens, new_intrinsic)
            family_bests[lens_name][model] = family_results[0]
            all_results.extend(family_results)
        all_results.sort(key=lambda item: item["aggregate_score"])
        rankings[lens_name] = all_results
        best = all_results[0]
        rectified, valid = rectify_image(images[lens_name], best, lens, new_intrinsic)
        overlay = draw_overlay(rectified, best)
        rectified_overlays[lens_name] = overlay
        cv2.imwrite(str(args.out_dir / f"{lens_name}_rectified_best.jpg"), rectified)
        cv2.imwrite(str(args.out_dir / f"{lens_name}_rectified_best_overlay.jpg"), overlay)
        cv2.imwrite(str(args.out_dir / f"{lens_name}_rectified_valid_mask.png"), (valid.astype(np.uint8) * 255))
        report_lenses[lens_name] = {
            "best": serializable(best),
            "rectified_intrinsic": new_intrinsic.tolist(),
            "tested_models": {
                model: {"tested_count": sum(1 for item in all_results if item["model"] == model), "best": serializable(family_bests[lens_name][model])}
                for model in args.models
            },
            "ranking_top_20": [serializable(item) for item in all_results[:20]],
        }
        print(f"{lens_name.upper()}: {best['model']} score={best['aggregate_score']:.8f} params=({best['scale_x']:.6f}, {best['scale_y']:.6f}, {best['offset_x']:.2f}, {best['offset_y']:.2f})")

    panel = comparison_panel(rectified_overlays, family_bests)
    cv2.imwrite(str(args.out_dir / "hypothesis_comparison.jpg"), panel)
    write_csv(args.out_dir / "ranking.csv", rankings)
    report = {
        "purpose": "Annotation-scored RAW-to-Veo mapping search with fixed Veo distortion; annotations do not define the warp.",
        "inputs": {"stacked": str(args.stacked), "calibration": str(args.calibration), "annotations": str(args.annotations)},
        "raw_lens_size": [raw_width, raw_height],
        "calibration_canvas_hypothesis": [args.calibration_width, args.calibration_height],
        "lenses": report_lenses,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved rectification analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
