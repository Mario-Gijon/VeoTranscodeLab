#!/usr/bin/env python3
"""
Analyze Veo optical-model consistency from automatic LEFT/RIGHT correspondences.

This diagnostic is intentionally independent from camera2world for its primary
metric. It asks:

    "Do pixels that depict the same physical scene point produce LEFT/RIGHT
     optical rays that lie in the same stereo epipolar plane?"

Pipeline
--------
1. Split a stacked 2048x2048 Veo frame:
       LEFT  = top    2048x1024
       RIGHT = bottom 2048x1024

2. Detect SIFT features in the expected overlap:
       LEFT  -> right side of image
       RIGHT -> left side of image

3. Match descriptors with:
       Lowe ratio test
       mutual nearest-neighbour agreement

4. Optionally apply a VERY LOOSE empirical fundamental-matrix RANSAC only to
   remove obvious descriptor mismatches. This filter is not used as the optical
   model being evaluated.

5. Evaluate several optical interpretations with the .veo intrinsics,
   distortion and lens2camera matrices.

Primary score
-------------
Symmetric angular epipolar residual in degrees, computed entirely in the common
CAMERA coordinate system before camera2world and before the Y=0 ground plane.

Interpretation
--------------
- Low residual across the image:
      RAW mapping + intrinsics + distortion + lens2camera are mutually coherent.

- Residual varies strongly by image position:
      suspect RAW->calibration mapping and/or distortion interpretation.

- Canonical model clearly beats alternatives:
      our current optical interpretation is probably directionally correct,
      even if some residual remains.

- An alternative model clearly beats canonical:
      investigate that specific assumption before touching camera2world.

This script NEVER changes calibration and does not fit camera2world.

Example
-------
python analysis/analyze_optical_model.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --stacked samples/match2_calibration_frame.jpg \
  --out-dir output/optical_model_match2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analysis.veo_geometry as vg  # noqa: E402


EPS = 1e-12


@dataclass(frozen=True)
class MatchPair:
    index: int
    left_pixel: np.ndarray
    right_pixel: np.ndarray
    descriptor_distance: float


@dataclass(frozen=True)
class CameraRay:
    origin: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class OpticalHypothesis:
    name: str
    description: str
    calibration_width: float
    calibration_height: float
    inverse_model: str
    distortion_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically match LEFT/RIGHT Veo features and compare optical "
            "model hypotheses before camera2world."
        )
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Path to the .veo calibration file.",
    )
    parser.add_argument(
        "--stacked",
        required=True,
        type=Path,
        help="2048x2048 stacked frame: LEFT top, RIGHT bottom.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for JSON/CSV/visual diagnostics.",
    )

    parser.add_argument(
        "--left-overlap-start",
        type=float,
        default=0.45,
        help=(
            "Fraction of LEFT image width where feature detection begins. "
            "Default: 0.45."
        ),
    )
    parser.add_argument(
        "--right-overlap-end",
        type=float,
        default=0.55,
        help=(
            "Fraction of RIGHT image width where feature detection ends. "
            "Default: 0.55."
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum SIFT features per lens. Default: 5000.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.72,
        help="Lowe descriptor ratio threshold. Default: 0.72.",
    )

    parser.add_argument(
        "--no-geometric-filter",
        action="store_true",
        help=(
            "Disable the loose empirical fundamental-matrix RANSAC used only "
            "to reject blatant descriptor mismatches."
        ),
    )
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=12.0,
        help=(
            "Loose raw-pixel RANSAC threshold for match cleanup. "
            "Default: 12 px."
        ),
    )
    parser.add_argument(
        "--max-visualized-matches",
        type=int,
        default=80,
        help="Maximum accepted correspondences drawn to matches.jpg.",
    )
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise RuntimeError("Cannot normalize a near-zero vector")
    return vector / norm


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def read_stacked(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if stacked is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = stacked.shape[:2]
    expected = (vg.RAW_WIDTH, vg.RAW_HEIGHT * 2)

    if (w, h) != expected:
        raise ValueError(
            f"Expected stacked frame {expected[0]}x{expected[1]}, "
            f"got {w}x{h}."
        )

    left = stacked[: vg.RAW_HEIGHT].copy()
    right = stacked[vg.RAW_HEIGHT :].copy()

    return stacked, left, right


def build_overlap_masks(
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
    left_overlap_start: float,
    right_overlap_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= left_overlap_start < 1.0):
        raise ValueError("--left-overlap-start must be in [0, 1)")
    if not (0.0 < right_overlap_end <= 1.0):
        raise ValueError("--right-overlap-end must be in (0, 1]")

    lh, lw = left_shape
    rh, rw = right_shape

    left_mask = np.zeros((lh, lw), dtype=np.uint8)
    right_mask = np.zeros((rh, rw), dtype=np.uint8)

    left_x0 = int(round(left_overlap_start * lw))
    right_x1 = int(round(right_overlap_end * rw))

    left_mask[:, left_x0:] = 255
    right_mask[:, :right_x1] = 255

    return left_mask, right_mask


def detect_and_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_overlap_start: float,
    right_overlap_end: float,
    max_features: int,
    ratio: float,
) -> tuple[
    list[cv2.KeyPoint],
    list[cv2.KeyPoint],
    list[cv2.DMatch],
]:
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_mask, right_mask = build_overlap_masks(
        gray_left.shape,
        gray_right.shape,
        left_overlap_start,
        right_overlap_end,
    )

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This diagnostic requires cv2.SIFT_create(), but the current "
            "OpenCV build does not expose SIFT."
        )

    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.01,
        edgeThreshold=15,
    )

    kp_left, desc_left = sift.detectAndCompute(
        gray_left,
        left_mask,
    )
    kp_right, desc_right = sift.detectAndCompute(
        gray_right,
        right_mask,
    )

    if desc_left is None or desc_right is None:
        raise RuntimeError(
            "Could not compute enough SIFT descriptors in the overlap."
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    lr_knn = matcher.knnMatch(
        desc_left,
        desc_right,
        k=2,
    )
    rl_knn = matcher.knnMatch(
        desc_right,
        desc_left,
        k=2,
    )

    lr_good: dict[int, cv2.DMatch] = {}
    for pair in lr_knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            lr_good[m.queryIdx] = m

    rl_good: dict[int, cv2.DMatch] = {}
    for pair in rl_knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            rl_good[m.queryIdx] = m

    mutual: list[cv2.DMatch] = []

    for left_query, match in lr_good.items():
        reverse = rl_good.get(match.trainIdx)
        if reverse is None:
            continue
        if reverse.trainIdx == left_query:
            mutual.append(match)

    mutual.sort(key=lambda m: float(m.distance))

    return kp_left, kp_right, mutual


def loose_geometric_filter(
    keypoints_left: list[cv2.KeyPoint],
    keypoints_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    threshold_px: float,
) -> tuple[list[cv2.DMatch], dict[str, Any]]:
    if len(matches) < 8:
        return matches, {
            "applied": False,
            "reason": "fewer than 8 descriptor matches",
        }

    left_points = np.float32(
        [
            keypoints_left[m.queryIdx].pt
            for m in matches
        ]
    )
    right_points = np.float32(
        [
            keypoints_right[m.trainIdx].pt
            for m in matches
        ]
    )

    method = getattr(
        cv2,
        "USAC_MAGSAC",
        cv2.FM_RANSAC,
    )

    try:
        fundamental, mask = cv2.findFundamentalMat(
            left_points,
            right_points,
            method,
            threshold_px,
            0.999,
            10000,
        )
    except TypeError:
        fundamental, mask = cv2.findFundamentalMat(
            left_points,
            right_points,
            cv2.FM_RANSAC,
            threshold_px,
            0.999,
        )

    if fundamental is None or mask is None:
        return matches, {
            "applied": False,
            "reason": "fundamental-matrix estimation failed",
        }

    inlier_mask = mask.reshape(-1).astype(bool)

    accepted = [
        match
        for match, keep in zip(matches, inlier_mask)
        if keep
    ]

    return accepted, {
        "applied": True,
        "threshold_px": threshold_px,
        "input_matches": len(matches),
        "accepted_matches": len(accepted),
        "empirical_fundamental_matrix": np.asarray(
            fundamental,
            dtype=np.float64,
        ).tolist(),
    }


def to_match_pairs(
    keypoints_left: list[cv2.KeyPoint],
    keypoints_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> list[MatchPair]:
    result: list[MatchPair] = []

    for index, match in enumerate(matches, start=1):
        left_pixel = np.asarray(
            keypoints_left[match.queryIdx].pt,
            dtype=np.float64,
        )
        right_pixel = np.asarray(
            keypoints_right[match.trainIdx].pt,
            dtype=np.float64,
        )

        result.append(
            MatchPair(
                index=index,
                left_pixel=left_pixel,
                right_pixel=right_pixel,
                descriptor_distance=float(match.distance),
            )
        )

    return result


def calibration_pixel(
    raw_pixel: np.ndarray,
    calibration_width: float,
    calibration_height: float,
) -> np.ndarray:
    u, v = np.asarray(
        raw_pixel,
        dtype=np.float64,
    ).reshape(2)

    return np.asarray(
        [
            u * calibration_width / vg.RAW_WIDTH,
            v * calibration_height / vg.RAW_HEIGHT,
        ],
        dtype=np.float64,
    )


def distortion_for_mode(
    source: np.ndarray,
    mode: str,
) -> np.ndarray:
    source = np.asarray(
        source,
        dtype=np.float64,
    ).reshape(-1)

    if mode == "full":
        return source.copy()

    if mode == "brown5":
        result = np.zeros_like(source)
        keep = min(5, source.size)
        result[:keep] = source[:keep]
        return result

    if mode == "zero":
        return np.zeros_like(source)

    raise ValueError(f"Unknown distortion mode: {mode!r}")


def undistort_opencv(
    distorted_pixel: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    return vg._undistort_point_precise(
        distorted_pixel,
        intrinsic,
        distortion,
    )


def undistort_fisheye4(
    distorted_pixel: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    distortion = np.asarray(
        distortion,
        dtype=np.float64,
    ).reshape(-1)

    fisheye_d = np.zeros((4, 1), dtype=np.float64)
    keep = min(4, distortion.size)
    fisheye_d[:keep, 0] = distortion[:keep]

    normalized = cv2.fisheye.undistortPoints(
        np.asarray(
            distorted_pixel,
            dtype=np.float64,
        ).reshape(1, 1, 2),
        intrinsic,
        fisheye_d,
    )

    return normalized.reshape(2)


def camera_ray_for_hypothesis(
    calibration: vg.VeoCalibration,
    lens_name: vg.LensName,
    raw_pixel: np.ndarray,
    hypothesis: OpticalHypothesis,
) -> CameraRay:
    lens = calibration.lens(lens_name)

    pixel_cal = calibration_pixel(
        raw_pixel,
        hypothesis.calibration_width,
        hypothesis.calibration_height,
    )

    distortion = distortion_for_mode(
        lens.distortion,
        hypothesis.distortion_mode,
    )

    if hypothesis.inverse_model == "opencv":
        normalized = undistort_opencv(
            pixel_cal,
            lens.intrinsic,
            distortion,
        )
    elif hypothesis.inverse_model == "fisheye":
        normalized = undistort_fisheye4(
            pixel_cal,
            lens.intrinsic,
            distortion,
        )
    else:
        raise ValueError(
            f"Unknown inverse model: {hypothesis.inverse_model!r}"
        )

    ray_lens = np.asarray(
        [
            normalized[0],
            normalized[1],
            1.0,
        ],
        dtype=np.float64,
    )

    origin_camera = lens.lens2camera[:3, 3].astype(
        np.float64,
        copy=True,
    )

    direction_camera = (
        lens.lens2camera[:3, :3] @ ray_lens
    )

    return CameraRay(
        origin=origin_camera,
        direction=normalize(direction_camera),
    )


def epipolar_residual_deg(
    left_ray: CameraRay,
    right_ray: CameraRay,
) -> float:
    baseline = right_ray.origin - left_ray.origin
    baseline_hat = normalize(baseline)

    left_plane_normal = np.cross(
        baseline_hat,
        left_ray.direction,
    )
    right_plane_normal = np.cross(
        baseline_hat,
        right_ray.direction,
    )

    left_norm = float(
        np.linalg.norm(left_plane_normal)
    )
    right_norm = float(
        np.linalg.norm(right_plane_normal)
    )

    if left_norm <= EPS or right_norm <= EPS:
        return float("nan")

    left_plane_normal /= left_norm
    right_plane_normal /= right_norm

    right_sin = abs(
        float(
            np.dot(
                right_ray.direction,
                left_plane_normal,
            )
        )
    )
    left_sin = abs(
        float(
            np.dot(
                left_ray.direction,
                right_plane_normal,
            )
        )
    )

    right_angle = math.degrees(
        math.asin(
            float(np.clip(right_sin, 0.0, 1.0))
        )
    )
    left_angle = math.degrees(
        math.asin(
            float(np.clip(left_sin, 0.0, 1.0))
        )
    )

    return 0.5 * (
        left_angle + right_angle
    )


def closest_ray_gap_m(
    left_ray: CameraRay,
    right_ray: CameraRay,
) -> float:
    p1 = left_ray.origin
    p2 = right_ray.origin
    d1 = left_ray.direction
    d2 = right_ray.direction

    w0 = p1 - p2

    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))

    denominator = a * c - b * b

    if abs(denominator) <= 1e-15:
        return float("nan")

    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator

    q1 = p1 + s * d1
    q2 = p2 + t * d2

    return float(
        np.linalg.norm(q1 - q2)
    )


def ray_angle_deg(
    left_ray: CameraRay,
    right_ray: CameraRay,
) -> float:
    cosine = float(
        np.clip(
            np.dot(
                left_ray.direction,
                right_ray.direction,
            ),
            -1.0,
            1.0,
        )
    )
    return math.degrees(
        math.acos(cosine)
    )


def hypotheses() -> list[OpticalHypothesis]:
    return [
        OpticalHypothesis(
            name="canonical",
            description=(
                "Current project interpretation: RAW 2048x1024 -> "
                "calibration 3840x1536, full OpenCV distortion vector."
            ),
            calibration_width=3840.0,
            calibration_height=1536.0,
            inverse_model="opencv",
            distortion_mode="full",
        ),
        OpticalHypothesis(
            name="brown5",
            description=(
                "Same 3840x1536 mapping but only k1,k2,p1,p2,k3 "
                "(OpenCV first 5 coefficients)."
            ),
            calibration_width=3840.0,
            calibration_height=1536.0,
            inverse_model="opencv",
            distortion_mode="brown5",
        ),
        OpticalHypothesis(
            name="no_distortion",
            description=(
                "Same 3840x1536 mapping with distortion disabled."
            ),
            calibration_width=3840.0,
            calibration_height=1536.0,
            inverse_model="opencv",
            distortion_mode="zero",
        ),
        OpticalHypothesis(
            name="uniform_from_width_3840x1920",
            description=(
                "Uniform RAW scaling derived from width: 2048x1024 -> "
                "3840x1920, full OpenCV distortion."
            ),
            calibration_width=3840.0,
            calibration_height=1920.0,
            inverse_model="opencv",
            distortion_mode="full",
        ),
        OpticalHypothesis(
            name="uniform_from_height_3072x1536",
            description=(
                "Uniform RAW scaling derived from height: 2048x1024 -> "
                "3072x1536, full OpenCV distortion."
            ),
            calibration_width=3072.0,
            calibration_height=1536.0,
            inverse_model="opencv",
            distortion_mode="full",
        ),
        OpticalHypothesis(
            name="fisheye4_3840x1536",
            description=(
                "Diagnostic only: 3840x1536 mapping with OpenCV fisheye "
                "inverse using the first four stored coefficients."
            ),
            calibration_width=3840.0,
            calibration_height=1536.0,
            inverse_model="fisheye",
            distortion_mode="full",
        ),
    ]


def evaluate_model(
    calibration: vg.VeoCalibration,
    pairs: list[MatchPair],
    hypothesis: OpticalHypothesis,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        row: dict[str, Any] = {
            "match_index": pair.index,
            "left_raw": pair.left_pixel.tolist(),
            "right_raw": pair.right_pixel.tolist(),
            "descriptor_distance": pair.descriptor_distance,
        }

        try:
            left_ray = camera_ray_for_hypothesis(
                calibration,
                "left",
                pair.left_pixel,
                hypothesis,
            )
            right_ray = camera_ray_for_hypothesis(
                calibration,
                "right",
                pair.right_pixel,
                hypothesis,
            )

            residual = epipolar_residual_deg(
                left_ray,
                right_ray,
            )
            gap = closest_ray_gap_m(
                left_ray,
                right_ray,
            )
            angle = ray_angle_deg(
                left_ray,
                right_ray,
            )

            row.update(
                {
                    "valid": True,
                    "epipolar_residual_deg": residual,
                    "closest_ray_gap_m": gap,
                    "ray_angle_deg": angle,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "valid": False,
                    "error": str(exc),
                    "epipolar_residual_deg": None,
                    "closest_ray_gap_m": None,
                    "ray_angle_deg": None,
                }
            )

        rows.append(row)

    residuals = np.asarray(
        [
            row["epipolar_residual_deg"]
            for row in rows
            if row["valid"]
            and row["epipolar_residual_deg"] is not None
            and math.isfinite(
                row["epipolar_residual_deg"]
            )
        ],
        dtype=np.float64,
    )

    gaps = np.asarray(
        [
            row["closest_ray_gap_m"]
            for row in rows
            if row["valid"]
            and row["closest_ray_gap_m"] is not None
            and math.isfinite(
                row["closest_ray_gap_m"]
            )
        ],
        dtype=np.float64,
    )

    summary = {
        "name": hypothesis.name,
        "description": hypothesis.description,
        "valid_matches": int(residuals.size),
        "failed_matches": int(
            len(rows) - residuals.size
        ),
        "epipolar_residual_deg": {
            "mean": (
                float(np.mean(residuals))
                if residuals.size
                else None
            ),
            "median": (
                float(np.median(residuals))
                if residuals.size
                else None
            ),
            "p75": (
                percentile(residuals, 75)
                if residuals.size
                else None
            ),
            "p90": (
                percentile(residuals, 90)
                if residuals.size
                else None
            ),
            "p95": (
                percentile(residuals, 95)
                if residuals.size
                else None
            ),
            "max": (
                float(np.max(residuals))
                if residuals.size
                else None
            ),
        },
        "closest_ray_gap_m": {
            "median": (
                float(np.median(gaps))
                if gaps.size
                else None
            ),
            "p90": (
                percentile(gaps, 90)
                if gaps.size
                else None
            ),
        },
    }

    return rows, summary


def spatial_summary(
    canonical_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [
        row
        for row in canonical_rows
        if row["valid"]
        and row["epipolar_residual_deg"] is not None
    ]

    if not valid:
        return {}

    enriched = []

    for row in valid:
        left = np.asarray(
            row["left_raw"],
            dtype=np.float64,
        )
        right = np.asarray(
            row["right_raw"],
            dtype=np.float64,
        )

        mean_v_fraction = float(
            0.5
            * (
                left[1] / vg.RAW_HEIGHT
                + right[1] / vg.RAW_HEIGHT
            )
        )

        enriched.append(
            (
                mean_v_fraction,
                float(
                    row["epipolar_residual_deg"]
                ),
            )
        )

    enriched.sort(key=lambda item: item[0])

    buckets: dict[str, list[float]] = {
        "top_quarter": [],
        "upper_middle": [],
        "lower_middle": [],
        "bottom_quarter": [],
    }

    for v_fraction, residual in enriched:
        if v_fraction < 0.25:
            key = "top_quarter"
        elif v_fraction < 0.50:
            key = "upper_middle"
        elif v_fraction < 0.75:
            key = "lower_middle"
        else:
            key = "bottom_quarter"

        buckets[key].append(residual)

    result: dict[str, Any] = {}

    for key, values_list in buckets.items():
        values = np.asarray(
            values_list,
            dtype=np.float64,
        )

        result[key] = {
            "count": int(values.size),
            "median_deg": (
                float(np.median(values))
                if values.size
                else None
            ),
            "p90_deg": (
                percentile(values, 90)
                if values.size
                else None
            ),
        }

    return result


def write_matches_visualization(
    left: np.ndarray,
    right: np.ndarray,
    keypoints_left: list[cv2.KeyPoint],
    keypoints_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: Path,
    max_matches: int,
) -> None:
    selected = matches[:max_matches]

    canvas = cv2.drawMatches(
        left,
        keypoints_left,
        right,
        keypoints_right,
        selected,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def write_csv(
    output_path: Path,
    pairs: list[MatchPair],
    model_rows: dict[str, list[dict[str, Any]]],
) -> None:
    model_names = list(model_rows.keys())

    row_lookup: dict[str, dict[int, dict[str, Any]]] = {
        name: {
            int(row["match_index"]): row
            for row in rows
        }
        for name, rows in model_rows.items()
    }

    fieldnames = [
        "match_index",
        "left_u",
        "left_v",
        "right_u",
        "right_v",
        "descriptor_distance",
    ]

    for name in model_names:
        fieldnames.extend(
            [
                f"{name}_epipolar_deg",
                f"{name}_closest_gap_m",
                f"{name}_ray_angle_deg",
            ]
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for pair in pairs:
            row: dict[str, Any] = {
                "match_index": pair.index,
                "left_u": float(pair.left_pixel[0]),
                "left_v": float(pair.left_pixel[1]),
                "right_u": float(pair.right_pixel[0]),
                "right_v": float(pair.right_pixel[1]),
                "descriptor_distance": pair.descriptor_distance,
            }

            for name in model_names:
                model_row = row_lookup[name][
                    pair.index
                ]

                row[f"{name}_epipolar_deg"] = (
                    model_row["epipolar_residual_deg"]
                )
                row[f"{name}_closest_gap_m"] = (
                    model_row["closest_ray_gap_m"]
                )
                row[f"{name}_ray_angle_deg"] = (
                    model_row["ray_angle_deg"]
                )

            writer.writerow(row)


def print_summary(
    descriptor_matches: int,
    accepted_matches: int,
    summaries: list[dict[str, Any]],
    spatial: dict[str, Any],
) -> None:
    print("=" * 96)
    print("Veo optical-model diagnostic")
    print("=" * 96)
    print(
        f"Mutual descriptor matches: {descriptor_matches}"
    )
    print(
        f"Accepted correspondences:   {accepted_matches}"
    )
    print()

    print(
        f"{'model':<34}"
        f"{'valid':>7}"
        f"{'median°':>12}"
        f"{'p90°':>12}"
        f"{'max°':>12}"
        f"{'gap med m':>13}"
    )
    print("-" * 96)

    for summary in summaries:
        epi = summary["epipolar_residual_deg"]
        gap = summary["closest_ray_gap_m"]

        def text(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"{value:.6f}"

        print(
            f"{summary['name']:<34}"
            f"{summary['valid_matches']:>7d}"
            f"{text(epi['median']):>12}"
            f"{text(epi['p90']):>12}"
            f"{text(epi['max']):>12}"
            f"{text(gap['median']):>13}"
        )

    canonical_summary = next(
        (
            summary
            for summary in summaries
            if summary["name"] == "canonical"
        ),
        None,
    )

    if canonical_summary is not None:
        print()
        print("Canonical residual by vertical image region")

        for key, data in spatial.items():
            print(
                f"  {key:<16} "
                f"n={data['count']:>3d}  "
                f"median={data['median_deg']}  "
                f"p90={data['p90_deg']}"
            )

    print()
    print(
        "Important: automatic correspondences are a diagnostic sample, "
        "not ground truth. Inspect matches.jpg before trusting the statistics."
    )


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = vg.VeoCalibration.from_file(
        args.calibration
    )

    _, left, right = read_stacked(
        args.stacked
    )

    kp_left, kp_right, mutual_matches = (
        detect_and_match(
            left,
            right,
            left_overlap_start=args.left_overlap_start,
            right_overlap_end=args.right_overlap_end,
            max_features=args.max_features,
            ratio=args.ratio,
        )
    )

    descriptor_match_count = len(
        mutual_matches
    )

    if args.no_geometric_filter:
        accepted_matches = mutual_matches
        filter_info = {
            "applied": False,
            "reason": "disabled by --no-geometric-filter",
        }
    else:
        accepted_matches, filter_info = (
            loose_geometric_filter(
                kp_left,
                kp_right,
                mutual_matches,
                args.ransac_threshold_px,
            )
        )

    if len(accepted_matches) < 8:
        raise RuntimeError(
            "Too few reliable LEFT/RIGHT correspondences. "
            f"Only {len(accepted_matches)} accepted. "
            "Try a slightly larger overlap region or a looser --ratio."
        )

    accepted_matches.sort(
        key=lambda m: float(m.distance)
    )

    pairs = to_match_pairs(
        kp_left,
        kp_right,
        accepted_matches,
    )

    model_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []

    for hypothesis in hypotheses():
        rows, summary = evaluate_model(
            calibration,
            pairs,
            hypothesis,
        )
        model_rows[hypothesis.name] = rows
        summaries.append(summary)

    canonical_rows = model_rows[
        "canonical"
    ]
    spatial = spatial_summary(
        canonical_rows
    )

    summary_lookup = {
        item["name"]: item
        for item in summaries
    }

    payload = {
        "calibration": str(args.calibration),
        "stacked": str(args.stacked),
        "raw_lens_size": [
            vg.RAW_WIDTH,
            vg.RAW_HEIGHT,
        ],
        "feature_matching": {
            "detector": "SIFT",
            "max_features": args.max_features,
            "left_overlap_start_fraction": (
                args.left_overlap_start
            ),
            "right_overlap_end_fraction": (
                args.right_overlap_end
            ),
            "lowe_ratio": args.ratio,
            "mutual_descriptor_matches": (
                descriptor_match_count
            ),
            "accepted_correspondences": len(
                accepted_matches
            ),
            "geometric_filter": filter_info,
        },
        "models": summaries,
        "canonical_vertical_spatial_summary": spatial,
        "matches": [
            {
                "match_index": pair.index,
                "left_raw": pair.left_pixel.tolist(),
                "right_raw": pair.right_pixel.tolist(),
                "descriptor_distance": (
                    pair.descriptor_distance
                ),
                "models": {
                    name: next(
                        row
                        for row in rows
                        if row["match_index"]
                        == pair.index
                    )
                    for name, rows in model_rows.items()
                },
            }
            for pair in pairs
        ],
    }

    json_path = (
        args.out_dir / "optical_model_report.json"
    )
    csv_path = (
        args.out_dir / "optical_model_matches.csv"
    )
    matches_path = (
        args.out_dir / "matches.jpg"
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        csv_path,
        pairs,
        model_rows,
    )

    write_matches_visualization(
        left,
        right,
        kp_left,
        kp_right,
        accepted_matches,
        matches_path,
        args.max_visualized_matches,
    )

    print_summary(
        descriptor_matches=descriptor_match_count,
        accepted_matches=len(accepted_matches),
        summaries=summaries,
        spatial=spatial,
    )

    print()
    print("Written:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {matches_path}")


if __name__ == "__main__":
    main()
