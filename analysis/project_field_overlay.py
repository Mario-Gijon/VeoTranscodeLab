#!/usr/bin/env python3
"""
Project a regulation football-pitch model onto Veo LEFT/RIGHT RAW images.

This is a visual validation tool for analysis/veo_geometry.py.

It does NOT implement camera geometry itself. Every field -> RAW projection is
delegated to VeoCalibration.field_to_pixel() from analysis.veo_geometry.

Purpose
-------
Given:
- one Veo `.veo` calibration file;
- one stacked 2048x2048 Veo frame, or separate 2048x1024 LEFT/RIGHT frames;

draw the theoretical pitch markings predicted by the `.veo` calibration on top
of the real RAW images.

If the projected model follows the actual painted lines across the pitch, the
`.veo` interpretation is geometrically plausible. Spatial drift patterns are
also easy to inspect visually.

Examples
--------
Stacked frame:

    python analysis/project_field_overlay.py \
        --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
        --stacked samples/match2_stacked_frame.jpg \
        --out-dir output/match2_overlay

Separate frames:

    python analysis/project_field_overlay.py \
        --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
        --left samples/match2_left_frame.jpg \
        --right samples/match2_right_frame.jpg \
        --out-dir output/match2_overlay
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.veo_geometry import (  # noqa: E402
    RAW_HEIGHT,
    RAW_WIDTH,
    VeoCalibration,
    load_veo_calibration,
)


LensName = Literal["left", "right"]

GOAL_WIDTH_M = 7.32
HALF_GOAL_WIDTH_M = GOAL_WIDTH_M / 2.0

CENTER_CIRCLE_RADIUS_M = 9.15
PENALTY_ARC_RADIUS_M = 9.15
PENALTY_SPOT_DISTANCE_M = 11.0

PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = HALF_GOAL_WIDTH_M + 16.5  # 20.16 m

GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = HALF_GOAL_WIDTH_M + 5.5  # 9.16 m


@dataclass(frozen=True)
class FieldCurve:
    name: str
    points: np.ndarray  # shape [N, 2], columns X,Z
    closed: bool = False


def line_points(
    a: tuple[float, float],
    b: tuple[float, float],
    spacing_m: float,
) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    distance = float(np.linalg.norm(b_arr - a_arr))

    steps = max(2, int(math.ceil(distance / spacing_m)) + 1)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
    return a_arr + (b_arr - a_arr) * t


def circle_points(
    center: tuple[float, float],
    radius_m: float,
    spacing_m: float,
) -> np.ndarray:
    circumference = 2.0 * math.pi * radius_m
    count = max(48, int(math.ceil(circumference / spacing_m)))
    theta = np.linspace(
        0.0,
        2.0 * math.pi,
        count,
        endpoint=False,
        dtype=np.float64,
    )

    cx, cz = center
    return np.column_stack(
        [
            cx + radius_m * np.cos(theta),
            cz + radius_m * np.sin(theta),
        ]
    )


def arc_points(
    center: tuple[float, float],
    radius_m: float,
    theta_start: float,
    theta_end: float,
    spacing_m: float,
) -> np.ndarray:
    arc_length = abs(theta_end - theta_start) * radius_m
    count = max(16, int(math.ceil(arc_length / spacing_m)) + 1)
    theta = np.linspace(
        theta_start,
        theta_end,
        count,
        dtype=np.float64,
    )

    cx, cz = center
    return np.column_stack(
        [
            cx + radius_m * np.cos(theta),
            cz + radius_m * np.sin(theta),
        ]
    )


def build_pitch_curves(
    field_length: float,
    field_width: float,
    spacing_m: float,
) -> list[FieldCurve]:
    """
    Build the regulation ground markings in the project's X/Z convention.

    X: pitch length axis
    Z: pitch width axis
    center: (0,0)
    """
    hl = field_length / 2.0
    hw = field_width / 2.0

    curves: list[FieldCurve] = []

    # Outer boundary.
    curves.extend(
        [
            FieldCurve(
                "left_goal_line",
                line_points((-hl, -hw), (-hl, +hw), spacing_m),
            ),
            FieldCurve(
                "right_goal_line",
                line_points((+hl, -hw), (+hl, +hw), spacing_m),
            ),
            FieldCurve(
                "close_touchline",
                line_points((-hl, -hw), (+hl, -hw), spacing_m),
            ),
            FieldCurve(
                "far_touchline",
                line_points((-hl, +hw), (+hl, +hw), spacing_m),
            ),
        ]
    )

    # Halfway line and center circle.
    curves.append(
        FieldCurve(
            "halfway_line",
            line_points((0.0, -hw), (0.0, +hw), spacing_m),
        )
    )
    curves.append(
        FieldCurve(
            "center_circle",
            circle_points(
                (0.0, 0.0),
                CENTER_CIRCLE_RADIUS_M,
                spacing_m,
            ),
            closed=True,
        )
    )

    # Left penalty area.
    left_pa_front_x = -hl + PENALTY_AREA_DEPTH_M
    curves.extend(
        [
            FieldCurve(
                "left_penalty_area_front",
                line_points(
                    (left_pa_front_x, -PENALTY_AREA_HALF_WIDTH_M),
                    (left_pa_front_x, +PENALTY_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "left_penalty_area_close_side",
                line_points(
                    (-hl, -PENALTY_AREA_HALF_WIDTH_M),
                    (left_pa_front_x, -PENALTY_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "left_penalty_area_far_side",
                line_points(
                    (-hl, +PENALTY_AREA_HALF_WIDTH_M),
                    (left_pa_front_x, +PENALTY_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
        ]
    )

    # Right penalty area.
    right_pa_front_x = +hl - PENALTY_AREA_DEPTH_M
    curves.extend(
        [
            FieldCurve(
                "right_penalty_area_front",
                line_points(
                    (right_pa_front_x, -PENALTY_AREA_HALF_WIDTH_M),
                    (right_pa_front_x, +PENALTY_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "right_penalty_area_close_side",
                line_points(
                    (+hl, -PENALTY_AREA_HALF_WIDTH_M),
                    (right_pa_front_x, -PENALTY_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "right_penalty_area_far_side",
                line_points(
                    (+hl, +PENALTY_AREA_HALF_WIDTH_M),
                    (right_pa_front_x, +PENALTY_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
        ]
    )

    # Left goal area.
    left_ga_front_x = -hl + GOAL_AREA_DEPTH_M
    curves.extend(
        [
            FieldCurve(
                "left_goal_area_front",
                line_points(
                    (left_ga_front_x, -GOAL_AREA_HALF_WIDTH_M),
                    (left_ga_front_x, +GOAL_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "left_goal_area_close_side",
                line_points(
                    (-hl, -GOAL_AREA_HALF_WIDTH_M),
                    (left_ga_front_x, -GOAL_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "left_goal_area_far_side",
                line_points(
                    (-hl, +GOAL_AREA_HALF_WIDTH_M),
                    (left_ga_front_x, +GOAL_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
        ]
    )

    # Right goal area.
    right_ga_front_x = +hl - GOAL_AREA_DEPTH_M
    curves.extend(
        [
            FieldCurve(
                "right_goal_area_front",
                line_points(
                    (right_ga_front_x, -GOAL_AREA_HALF_WIDTH_M),
                    (right_ga_front_x, +GOAL_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "right_goal_area_close_side",
                line_points(
                    (+hl, -GOAL_AREA_HALF_WIDTH_M),
                    (right_ga_front_x, -GOAL_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
            FieldCurve(
                "right_goal_area_far_side",
                line_points(
                    (+hl, +GOAL_AREA_HALF_WIDTH_M),
                    (right_ga_front_x, +GOAL_AREA_HALF_WIDTH_M),
                    spacing_m,
                ),
            ),
        ]
    )

    # Penalty arcs: only the part outside the penalty area.
    #
    # Left goal: outside the PA is toward +X.
    # Right goal: outside the PA is toward -X.
    # Equivalent to acos(5.5 / 9.15).
    penalty_half_angle = math.acos(
        (PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DISTANCE_M)
        / PENALTY_ARC_RADIUS_M
    )

    left_penalty_x = -hl + PENALTY_SPOT_DISTANCE_M
    curves.append(
        FieldCurve(
            "left_penalty_arc",
            arc_points(
                (left_penalty_x, 0.0),
                PENALTY_ARC_RADIUS_M,
                -penalty_half_angle,
                +penalty_half_angle,
                spacing_m,
            ),
        )
    )

    right_penalty_x = +hl - PENALTY_SPOT_DISTANCE_M
    curves.append(
        FieldCurve(
            "right_penalty_arc",
            arc_points(
                (right_penalty_x, 0.0),
                PENALTY_ARC_RADIUS_M,
                math.pi - penalty_half_angle,
                math.pi + penalty_half_angle,
                spacing_m,
            ),
        )
    )

    return curves


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def validate_raw_lens_shape(
    image: np.ndarray,
    label: str,
) -> None:
    h, w = image.shape[:2]
    if (w, h) != (RAW_WIDTH, RAW_HEIGHT):
        raise ValueError(
            f"{label} image must be {RAW_WIDTH}x{RAW_HEIGHT}, "
            f"got {w}x{h}. This validator intentionally avoids hidden resizing."
        )


def load_lens_images(
    stacked: Path | None,
    left: Path | None,
    right: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if stacked is not None:
        if left is not None or right is not None:
            raise ValueError(
                "Use either --stacked or --left/--right, not both."
            )

        image = read_image(stacked)
        h, w = image.shape[:2]

        if (w, h) != (RAW_WIDTH, RAW_HEIGHT * 2):
            raise ValueError(
                f"Stacked image must be {RAW_WIDTH}x{RAW_HEIGHT * 2}, "
                f"got {w}x{h}."
            )

        return (
            image[:RAW_HEIGHT].copy(),
            image[RAW_HEIGHT:].copy(),
        )

    if left is None or right is None:
        raise ValueError(
            "Provide --stacked, or provide both --left and --right."
        )

    left_image = read_image(left)
    right_image = read_image(right)

    validate_raw_lens_shape(left_image, "LEFT")
    validate_raw_lens_shape(right_image, "RIGHT")

    return left_image, right_image


def project_curve(
    calibration: VeoCalibration,
    lens_name: LensName,
    curve: FieldCurve,
) -> list[np.ndarray | None]:
    projected: list[np.ndarray | None] = []

    for field_point in curve.points:
        try:
            pixel = calibration.field_to_pixel(
                lens_name,
                field_point,
            )
        except RuntimeError:
            projected.append(None)
            continue

        if not np.all(np.isfinite(pixel)):
            projected.append(None)
            continue

        projected.append(
            np.asarray(pixel, dtype=np.float64).reshape(2)
        )

    if curve.closed and projected:
        projected.append(projected[0])

    return projected


def draw_projected_segments(
    canvas: np.ndarray,
    projected: list[np.ndarray | None],
    color: tuple[int, int, int],
    thickness: int,
) -> int:
    """
    Draw pairwise projected samples using cv2.clipLine, so lines crossing
    the image remain visible even when one endpoint lies outside the frame.
    """
    h, w = canvas.shape[:2]
    rect = (0, 0, w, h)
    drawn = 0

    for p0, p1 in zip(projected[:-1], projected[1:]):
        if p0 is None or p1 is None:
            continue

        # Avoid pathological joins across enormous projection discontinuities.
        if float(np.linalg.norm(p1 - p0)) > 500.0:
            continue

        pt0 = (
            int(round(float(p0[0]))),
            int(round(float(p0[1]))),
        )
        pt1 = (
            int(round(float(p1[0]))),
            int(round(float(p1[1]))),
        )

        visible, clipped0, clipped1 = cv2.clipLine(
            rect,
            pt0,
            pt1,
        )

        if not visible:
            continue

        cv2.line(
            canvas,
            clipped0,
            clipped1,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        drawn += 1

    return drawn


def draw_key_landmarks(
    canvas: np.ndarray,
    calibration: VeoCalibration,
    lens_name: LensName,
    *,
    color: tuple[int, int, int],
    radius: int,
    draw_labels: bool,
) -> None:
    hl = calibration.field_length / 2.0
    hw = calibration.field_width / 2.0

    landmarks = {
        "C": (0.0, 0.0),
        "LP": (-hl + PENALTY_SPOT_DISTANCE_M, 0.0),
        "RP": (+hl - PENALTY_SPOT_DISTANCE_M, 0.0),
        "HF": (0.0, +hw),
        "HC": (0.0, -hw),
    }

    h, w = canvas.shape[:2]

    for label, field_point in landmarks.items():
        try:
            pixel = calibration.field_to_pixel(
                lens_name,
                field_point,
            )
        except RuntimeError:
            continue

        u = int(round(float(pixel[0])))
        v = int(round(float(pixel[1])))

        if not (0 <= u < w and 0 <= v < h):
            continue

        cv2.circle(
            canvas,
            (u, v),
            radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if draw_labels:
            cv2.putText(
                canvas,
                label,
                (u + radius + 3, v - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


def make_overlay(
    raw_image: np.ndarray,
    calibration: VeoCalibration,
    lens_name: LensName,
    curves: Iterable[FieldCurve],
    *,
    alpha: float,
    thickness: int,
    draw_labels: bool,
) -> tuple[np.ndarray, int]:
    """
    Draw projected markings on a transparent drawing layer and blend it with
    the untouched RAW image.
    """
    layer = raw_image.copy()

    # BGR: vivid red for projected pitch lines, yellow for key landmarks.
    line_color = (0, 0, 255)
    landmark_color = (0, 255, 255)

    drawn_segments = 0

    for curve in curves:
        projected = project_curve(
            calibration,
            lens_name,
            curve,
        )
        drawn_segments += draw_projected_segments(
            layer,
            projected,
            line_color,
            thickness,
        )

    draw_key_landmarks(
        layer,
        calibration,
        lens_name,
        color=landmark_color,
        radius=max(3, thickness + 1),
        draw_labels=draw_labels,
    )

    overlay = cv2.addWeighted(
        layer,
        alpha,
        raw_image,
        1.0 - alpha,
        0.0,
    )

    return overlay, drawn_segments


def print_key_landmarks(
    calibration: VeoCalibration,
    lens_name: LensName,
) -> None:
    hl = calibration.field_length / 2.0
    hw = calibration.field_width / 2.0

    landmarks = [
        ("CENTER", (0.0, 0.0)),
        ("LEFT_PENALTY", (-hl + PENALTY_SPOT_DISTANCE_M, 0.0)),
        ("RIGHT_PENALTY", (+hl - PENALTY_SPOT_DISTANCE_M, 0.0)),
        ("HALFWAY_FAR", (0.0, +hw)),
        ("HALFWAY_CLOSE", (0.0, -hw)),
    ]

    print(f"{lens_name.upper()} predicted landmarks")
    print("-" * (len(lens_name) + 20))

    for name, field_point in landmarks:
        try:
            pixel = calibration.field_to_pixel(
                lens_name,
                field_point,
            )
        except RuntimeError as exc:
            print(f"{name:18s}: not projectable ({exc})")
            continue

        print(
            f"{name:18s}: "
            f"RAW=({pixel[0]:.3f}, {pixel[1]:.3f})"
        )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a theoretical football pitch onto Veo LEFT/RIGHT RAW "
            "images using analysis/veo_geometry.py."
        )
    )

    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Veo .veo calibration file.",
    )

    input_group = parser.add_argument_group("input images")
    input_group.add_argument(
        "--stacked",
        type=Path,
        default=None,
        help="2048x2048 stacked Veo frame: LEFT on top, RIGHT below.",
    )
    input_group.add_argument(
        "--left",
        type=Path,
        default=None,
        help="2048x1024 LEFT RAW image.",
    )
    input_group.add_argument(
        "--right",
        type=Path,
        default=None,
        help="2048x1024 RIGHT RAW image.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/field_overlay"),
        help="Output directory (default: output/field_overlay).",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=0.25,
        help="Sampling spacing along pitch markings in meters (default: 0.25).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.82,
        help="Overlay opacity in [0,1] (default: 0.82).",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Projected line thickness in RAW pixels (default: 2).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not label key predicted landmarks.",
    )

    args = parser.parse_args()

    if not args.calibration.is_file():
        parser.error(
            f"Calibration file does not exist: {args.calibration}"
        )

    if args.stacked is not None:
        if not args.stacked.is_file():
            parser.error(
                f"Stacked image does not exist: {args.stacked}"
            )
    else:
        if args.left is None or args.right is None:
            parser.error(
                "Provide --stacked, or provide both --left and --right."
            )
        if not args.left.is_file():
            parser.error(f"LEFT image does not exist: {args.left}")
        if not args.right.is_file():
            parser.error(f"RIGHT image does not exist: {args.right}")

    if args.spacing_m <= 0:
        parser.error("--spacing-m must be > 0")

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")

    if args.thickness < 1:
        parser.error("--thickness must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    calibration = load_veo_calibration(
        args.calibration
    )

    left_raw, right_raw = load_lens_images(
        args.stacked,
        args.left,
        args.right,
    )

    curves = build_pitch_curves(
        calibration.field_length,
        calibration.field_width,
        args.spacing_m,
    )

    left_overlay, left_segments = make_overlay(
        left_raw,
        calibration,
        "left",
        curves,
        alpha=args.alpha,
        thickness=args.thickness,
        draw_labels=not args.no_labels,
    )

    right_overlay, right_segments = make_overlay(
        right_raw,
        calibration,
        "right",
        curves,
        alpha=args.alpha,
        thickness=args.thickness,
        draw_labels=not args.no_labels,
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    left_path = args.out_dir / "left_field_overlay.jpg"
    right_path = args.out_dir / "right_field_overlay.jpg"
    stacked_path = args.out_dir / "stacked_field_overlay.jpg"

    if not cv2.imwrite(
        str(left_path),
        left_overlay,
    ):
        raise RuntimeError(
            f"Could not write {left_path}"
        )

    if not cv2.imwrite(
        str(right_path),
        right_overlay,
    ):
        raise RuntimeError(
            f"Could not write {right_path}"
        )

    stacked_overlay = np.vstack(
        [
            left_overlay,
            right_overlay,
        ]
    )

    if not cv2.imwrite(
        str(stacked_path),
        stacked_overlay,
    ):
        raise RuntimeError(
            f"Could not write {stacked_path}"
        )

    print(
        f"Field: {calibration.field_length:.3f} x "
        f"{calibration.field_width:.3f} m"
    )
    print(
        f"Curves: {len(curves)} | "
        f"sampling: {args.spacing_m:.3f} m"
    )
    print(
        f"LEFT drawn segments:  {left_segments}"
    )
    print(
        f"RIGHT drawn segments: {right_segments}"
    )
    print()

    print_key_landmarks(
        calibration,
        "left",
    )
    print_key_landmarks(
        calibration,
        "right",
    )

    print("Written:")
    print(f"  {left_path}")
    print(f"  {right_path}")
    print(f"  {stacked_path}")


if __name__ == "__main__":
    main()
