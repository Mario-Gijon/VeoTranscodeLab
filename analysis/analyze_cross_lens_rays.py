#!/usr/bin/env python3
"""
Cross-lens ray consistency diagnostic for VeoTranscodeLab.

Purpose
-------
Given corresponding RAW pixels for the SAME physical point observed in the
LEFT and RIGHT Veo lenses, inspect the stereo geometry BEFORE the shared
camera2world transform and ground-plane intersection.

This script does NOT modify or fit calibration.

It uses the canonical Veo geometry implementation from:
    analysis/veo_geometry.py

For each LEFT/RIGHT pair it reports:

1. The two optical rays expressed in the common CAMERA coordinate system.
2. Stereo baseline.
3. Angle between the rays.
4. Epipolar/coplanarity residual, independent of camera2world.
5. Closest approach between the two 3D rays.
6. Triangulated midpoint (diagnostic only; depth can be ill-conditioned
   because the Veo lens baseline is small compared with football-pitch
   distances).
7. Existing canonical pixel_to_field() result for each lens and their
   difference.
8. Optional triangulated midpoint transformed to world coordinates, including
   distance from the expected Y=0 ground plane.

Interpretation
--------------
- Large epipolar residual:
    suspect RAW->calibration mapping, distortion interpretation, intrinsics,
    or lens2camera extrinsics.

- Small epipolar residual but different pixel_to_field() ground positions:
    optical LEFT/RIGHT geometry is mutually consistent; investigate the shared
    camera2world / ground-plane model or the amplification caused by shallow
    ground-ray angles.

- Tiny stereo ray angle:
    triangulated depth is numerically sensitive. Do NOT interpret triangulated
    depth as a precise ground-truth position merely because the closest-ray gap
    is small.

Example
-------
python analysis/analyze_cross_lens_rays.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --pair "far_touchline=1923.0,535.7:88.6,580.0" \
  --pair "circle_far=1953.5,598.7:129.1,617.4" \
  --pair "centerish=1971.3,624.2:154.6,633.2" \
  --pair "circle_close=2010.8,670.6:210.8,665.6" \
  --out output/cross_lens_rays_match2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analysis.veo_geometry as vg  # noqa: E402


EPS = 1e-12


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze LEFT/RIGHT Veo ray consistency before camera2world."
        )
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to the .veo calibration file.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help=(
            "Corresponding RAW pixels for the same physical point. "
            "Format: [label=]uL,vL:uR,vR . "
            "Repeat --pair for multiple correspondences."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def parse_xy(text: str) -> np.ndarray:
    pieces = [part.strip() for part in text.split(",")]
    if len(pieces) != 2:
        raise ValueError(f"Expected 'u,v', got {text!r}")
    return np.asarray(
        [float(pieces[0]), float(pieces[1])],
        dtype=np.float64,
    )


def parse_pair(text: str, index: int) -> tuple[str, np.ndarray, np.ndarray]:
    raw = text.strip()

    if "=" in raw:
        label, coordinates = raw.split("=", 1)
        label = label.strip()
        if not label:
            label = f"pair_{index}"
    else:
        label = f"pair_{index}"
        coordinates = raw

    if ":" not in coordinates:
        raise ValueError(
            f"Invalid --pair {text!r}. "
            "Expected [label=]uL,vL:uR,vR"
        )

    left_text, right_text = coordinates.split(":", 1)

    return (
        label,
        parse_xy(left_text),
        parse_xy(right_text),
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise RuntimeError("Cannot normalize a near-zero vector")
    return vector / norm


def raw_pixel_to_camera_ray(
    calibration: vg.VeoCalibration,
    lens_name: vg.LensName,
    raw_pixel: np.ndarray,
) -> Ray:
    """
    Reproduce ONLY the canonical pre-camera2world part of pixel_to_world_ground:

        RAW
        -> Veo calibration pixel
        -> precise inverse distortion
        -> lens ray
        -> lens2camera
        -> common CAMERA-space ray

    camera2world is deliberately NOT applied here.
    """
    lens = calibration.lens(lens_name)

    calibration_pixel = calibration.raw_to_calibration_pixel(
        raw_pixel
    )

    normalized_xy = vg._undistort_point_precise(
        calibration_pixel,
        lens.intrinsic,
        lens.distortion,
    )

    ray_lens = np.asarray(
        [
            normalized_xy[0],
            normalized_xy[1],
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

    return Ray(
        origin=origin_camera,
        direction=normalize(direction_camera),
    )


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def epipolar_plane_residual_deg(
    left_ray: Ray,
    right_ray: Ray,
) -> dict[str, float]:
    """
    Measure whether the two bearing rays and the stereo baseline are coplanar.

    For a perfect stereo correspondence:
        baseline, d_left, d_right
    lie in one epipolar plane.

    We report the angular departure of each ray from the epipolar plane formed
    by the other ray and the baseline. This is independent of camera2world and
    scene depth.
    """
    baseline = right_ray.origin - left_ray.origin
    baseline_hat = normalize(baseline)

    n_from_left = np.cross(
        baseline_hat,
        left_ray.direction,
    )
    n_from_right = np.cross(
        baseline_hat,
        right_ray.direction,
    )

    n_left_norm = float(np.linalg.norm(n_from_left))
    n_right_norm = float(np.linalg.norm(n_from_right))

    if n_left_norm <= EPS or n_right_norm <= EPS:
        return {
            "right_to_left_plane_deg": float("nan"),
            "left_to_right_plane_deg": float("nan"),
            "symmetric_deg": float("nan"),
        }

    n_from_left /= n_left_norm
    n_from_right /= n_right_norm

    right_sin = abs(
        float(np.dot(right_ray.direction, n_from_left))
    )
    left_sin = abs(
        float(np.dot(left_ray.direction, n_from_right))
    )

    right_sin = float(np.clip(right_sin, 0.0, 1.0))
    left_sin = float(np.clip(left_sin, 0.0, 1.0))

    right_residual = math.degrees(math.asin(right_sin))
    left_residual = math.degrees(math.asin(left_sin))

    return {
        "right_to_left_plane_deg": right_residual,
        "left_to_right_plane_deg": left_residual,
        "symmetric_deg": 0.5 * (
            right_residual + left_residual
        ),
    }


def closest_points_between_rays(
    ray_a: Ray,
    ray_b: Ray,
) -> dict[str, Any]:
    """
    Find closest points on two infinite 3D rays/lines.

    Directions are normalized. s and t are therefore distances in meters in
    the common CAMERA coordinate system, assuming the .veo translations are
    metric.

    When the rays are almost parallel, the denominator becomes tiny and stereo
    depth is ill-conditioned. We still report the geometry, but the caller must
    use the ray-angle / condition indicators.
    """
    p1 = ray_a.origin
    p2 = ray_b.origin
    d1 = ray_a.direction
    d2 = ray_b.direction

    w0 = p1 - p2

    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))

    denominator = a * c - b * b

    if abs(denominator) <= 1e-15:
        return {
            "parallel_or_near_parallel": True,
            "denominator": denominator,
            "left_distance_m": None,
            "right_distance_m": None,
            "left_closest_camera": None,
            "right_closest_camera": None,
            "midpoint_camera": None,
            "closest_gap_m": None,
        }

    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator

    q1 = p1 + s * d1
    q2 = p2 + t * d2

    midpoint = 0.5 * (q1 + q2)
    gap = float(np.linalg.norm(q1 - q2))

    return {
        "parallel_or_near_parallel": False,
        "denominator": denominator,
        "left_distance_m": float(s),
        "right_distance_m": float(t),
        "left_closest_camera": q1.tolist(),
        "right_closest_camera": q2.tolist(),
        "midpoint_camera": midpoint.tolist(),
        "closest_gap_m": gap,
        "left_point_in_front": bool(s > 0.0),
        "right_point_in_front": bool(t > 0.0),
    }


def camera_point_to_world(
    calibration: vg.VeoCalibration,
    camera_point: np.ndarray,
) -> np.ndarray:
    camera_point = np.asarray(
        camera_point,
        dtype=np.float64,
    ).reshape(3)

    camera_h = np.asarray(
        [
            camera_point[0],
            camera_point[1],
            camera_point[2],
            1.0,
        ],
        dtype=np.float64,
    )

    world_h = calibration.camera2world @ camera_h

    if abs(float(world_h[3])) <= EPS:
        raise RuntimeError("Invalid homogeneous world point")

    return world_h[:3] / world_h[3]


def analyze_pair(
    calibration: vg.VeoCalibration,
    label: str,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
) -> dict[str, Any]:
    left_ray = raw_pixel_to_camera_ray(
        calibration,
        "left",
        left_pixel,
    )
    right_ray = raw_pixel_to_camera_ray(
        calibration,
        "right",
        right_pixel,
    )

    baseline_vector = (
        right_ray.origin - left_ray.origin
    )
    baseline_m = float(
        np.linalg.norm(baseline_vector)
    )

    stereo_angle_deg = angle_deg(
        left_ray.direction,
        right_ray.direction,
    )

    stereo_angle_rad = math.radians(stereo_angle_deg)
    sin_angle = abs(math.sin(stereo_angle_rad))
    condition_factor = (
        float("inf")
        if sin_angle <= EPS
        else 1.0 / sin_angle
    )

    epipolar = epipolar_plane_residual_deg(
        left_ray,
        right_ray,
    )

    closest = closest_points_between_rays(
        left_ray,
        right_ray,
    )

    field_left = calibration.pixel_to_field(
        "left",
        left_pixel,
    )
    field_right = calibration.pixel_to_field(
        "right",
        right_pixel,
    )

    field_delta = field_right - field_left
    field_disagreement_m = float(
        np.linalg.norm(field_delta)
    )

    triangulated_world = None
    triangulated_ground_height_m = None

    midpoint = closest.get("midpoint_camera")
    if midpoint is not None:
        world = camera_point_to_world(
            calibration,
            np.asarray(midpoint, dtype=np.float64),
        )
        triangulated_world = world.tolist()
        triangulated_ground_height_m = float(world[1])

    return {
        "label": label,
        "raw_pixels": {
            "left": left_pixel.tolist(),
            "right": right_pixel.tolist(),
        },
        "camera_rays": {
            "left": {
                "origin": left_ray.origin.tolist(),
                "direction_unit": left_ray.direction.tolist(),
            },
            "right": {
                "origin": right_ray.origin.tolist(),
                "direction_unit": right_ray.direction.tolist(),
            },
        },
        "stereo": {
            "baseline_vector_camera_m": baseline_vector.tolist(),
            "baseline_m": baseline_m,
            "ray_angle_deg": stereo_angle_deg,
            "depth_condition_factor_1_over_sin_angle": condition_factor,
            "epipolar_residual_deg": epipolar,
            "closest_approach": closest,
        },
        "canonical_ground_intersections": {
            "left_field_xz_m": field_left.tolist(),
            "right_field_xz_m": field_right.tolist(),
            "right_minus_left_xz_m": field_delta.tolist(),
            "left_right_disagreement_m": field_disagreement_m,
        },
        "triangulation_after_camera2world_diagnostic": {
            "midpoint_world_xyz_m": triangulated_world,
            "midpoint_world_ground_height_y_m": (
                triangulated_ground_height_m
            ),
        },
    }


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def print_pair_report(result: dict[str, Any]) -> None:
    stereo = result["stereo"]
    closest = stereo["closest_approach"]
    ground = result["canonical_ground_intersections"]
    epi = stereo["epipolar_residual_deg"]
    tri = result[
        "triangulation_after_camera2world_diagnostic"
    ]

    left_field = np.asarray(
        ground["left_field_xz_m"],
        dtype=np.float64,
    )
    right_field = np.asarray(
        ground["right_field_xz_m"],
        dtype=np.float64,
    )
    delta = np.asarray(
        ground["right_minus_left_xz_m"],
        dtype=np.float64,
    )

    print()
    print("=" * 78)
    print(result["label"])
    print("=" * 78)

    print(
        "LEFT  raw = "
        f"({result['raw_pixels']['left'][0]:.3f}, "
        f"{result['raw_pixels']['left'][1]:.3f})"
    )
    print(
        "RIGHT raw = "
        f"({result['raw_pixels']['right'][0]:.3f}, "
        f"{result['raw_pixels']['right'][1]:.3f})"
    )

    print()
    print("PRE-camera2world stereo geometry")
    print(
        f"  baseline:              "
        f"{stereo['baseline_m']:.6f} m"
    )
    print(
        f"  ray angle:             "
        f"{stereo['ray_angle_deg']:.6f} deg"
    )
    print(
        f"  depth condition 1/sin: "
        f"{fmt(stereo['depth_condition_factor_1_over_sin_angle'], 2)}"
    )
    print(
        f"  epipolar residual:     "
        f"{fmt(epi['symmetric_deg'], 6)} deg"
    )
    print(
        f"  closest ray gap:       "
        f"{fmt(closest['closest_gap_m'], 6)} m"
    )

    if closest["left_distance_m"] is not None:
        print(
            f"  ray distances L/R:     "
            f"{closest['left_distance_m']:.3f} / "
            f"{closest['right_distance_m']:.3f} m"
        )

    print()
    print("Canonical Y=0 ground intersections")
    print(
        f"  LEFT:   X={left_field[0]:+.3f} "
        f"Z={left_field[1]:+.3f} m"
    )
    print(
        f"  RIGHT:  X={right_field[0]:+.3f} "
        f"Z={right_field[1]:+.3f} m"
    )
    print(
        f"  delta:  dX={delta[0]:+.3f} "
        f"dZ={delta[1]:+.3f} m"
    )
    print(
        f"  distance: "
        f"{ground['left_right_disagreement_m']:.3f} m"
    )

    if tri["midpoint_world_xyz_m"] is not None:
        world = tri["midpoint_world_xyz_m"]
        print()
        print("Triangulated midpoint diagnostic")
        print(
            f"  world XYZ: "
            f"({world[0]:+.3f}, "
            f"{world[1]:+.3f}, "
            f"{world[2]:+.3f}) m"
        )
        print(
            f"  height vs Y=0 ground: "
            f"{tri['midpoint_world_ground_height_y_m']:+.3f} m"
        )

    if stereo["ray_angle_deg"] < 0.25:
        print()
        print(
            "  NOTE: tiny stereo angle -> triangulated DEPTH is "
            "highly sensitive to pixel-level error."
        )


def main() -> None:
    args = parse_args()

    calibration = vg.VeoCalibration.from_file(
        args.calibration
    )

    pairs = [
        parse_pair(text, index)
        for index, text in enumerate(
            args.pair,
            start=1,
        )
    ]

    print("=" * 78)
    print("Veo cross-lens ray consistency")
    print("=" * 78)
    print(f"Calibration: {args.calibration}")
    print(
        f"Field:       {calibration.field_length:.3f} x "
        f"{calibration.field_width:.3f} m"
    )
    print(f"Pairs:       {len(pairs)}")
    print()
    print(
        "Primary diagnostic is PRE-camera2world epipolar/ray consistency."
    )
    print(
        "Triangulated depth is secondary because the stereo baseline is small."
    )

    results = []

    for label, left_pixel, right_pixel in pairs:
        result = analyze_pair(
            calibration,
            label,
            left_pixel,
            right_pixel,
        )
        results.append(result)
        print_pair_report(result)

    field_disagreements = np.asarray(
        [
            item["canonical_ground_intersections"][
                "left_right_disagreement_m"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    epipolar_residuals = np.asarray(
        [
            item["stereo"]["epipolar_residual_deg"][
                "symmetric_deg"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    finite_epi = epipolar_residuals[
        np.isfinite(epipolar_residuals)
    ]

    summary = {
        "num_pairs": len(results),
        "ground_disagreement_m": {
            "mean": float(np.mean(field_disagreements)),
            "median": float(np.median(field_disagreements)),
            "max": float(np.max(field_disagreements)),
        },
        "epipolar_residual_deg": (
            {
                "mean": float(np.mean(finite_epi)),
                "median": float(np.median(finite_epi)),
                "max": float(np.max(finite_epi)),
            }
            if finite_epi.size
            else None
        ),
    }

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        "Ground LEFT/RIGHT disagreement: "
        f"mean={summary['ground_disagreement_m']['mean']:.3f} m, "
        f"median={summary['ground_disagreement_m']['median']:.3f} m, "
        f"max={summary['ground_disagreement_m']['max']:.3f} m"
    )

    if summary["epipolar_residual_deg"] is not None:
        epi_summary = summary["epipolar_residual_deg"]
        print(
            "Epipolar residual: "
            f"mean={epi_summary['mean']:.6f} deg, "
            f"median={epi_summary['median']:.6f} deg, "
            f"max={epi_summary['max']:.6f} deg"
        )

    if args.out is not None:
        args.out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "calibration": str(args.calibration),
            "field_length_m": calibration.field_length,
            "field_width_m": calibration.field_width,
            "summary": summary,
            "pairs": results,
        }

        args.out.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
