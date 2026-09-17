#!/usr/bin/env python3
"""
Inspect Veo selected corners versus the canonical projected field corners.

Version 2 fixes the selected-corner interpretation used by the first script.

IMPORTANT:
`selected_corners.*.*.stacked` in the `.veo` file are normalized coordinates
in the 2048x2048 stacked frame, not pixels in the 3840x1536 calibration space.

This version deliberately avoids reimplementing that conversion and instead
uses the canonical helper:

    VeoCalibration.selected_corner_observations()

which already returns correct LOCAL RAW 2048x1024 lens coordinates.

Visualization:
- YELLOW circle: selected corner stored by Veo
- RED cross:     same physical field corner projected by our canonical model
- CYAN segment:  residual between Veo observation and model prediction

No geometry or calibration is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.veo_geometry import (  # noqa: E402
    RAW_HEIGHT,
    RAW_WIDTH,
    load_veo_calibration,
)


YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Veo selected RAW corners versus canonical projected "
            "field corners."
        )
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Path to the .veo calibration file.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stacked",
        type=Path,
        help="2048x2048 stacked frame: LEFT top, RIGHT bottom.",
    )
    group.add_argument(
        "--left",
        type=Path,
        help="2048x1024 LEFT image. Requires --right.",
    )

    parser.add_argument(
        "--right",
        type=Path,
        default=None,
        help="2048x1024 RIGHT image.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--crop-radius",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--crop-scale",
        type=int,
        default=5,
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def load_lenses(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.stacked is not None:
        image = read_image(args.stacked)
        h, w = image.shape[:2]

        if (w, h) != (RAW_WIDTH, RAW_HEIGHT * 2):
            raise ValueError(
                f"Expected stacked image {RAW_WIDTH}x{RAW_HEIGHT * 2}, "
                f"got {w}x{h}"
            )

        return {
            "left": image[:RAW_HEIGHT].copy(),
            "right": image[RAW_HEIGHT:].copy(),
        }

    if args.right is None:
        raise ValueError("--left requires --right")

    left = read_image(args.left)
    right = read_image(args.right)

    for label, image in (("LEFT", left), ("RIGHT", right)):
        h, w = image.shape[:2]
        if (w, h) != (RAW_WIDTH, RAW_HEIGHT):
            raise ValueError(
                f"{label} must be {RAW_WIDTH}x{RAW_HEIGHT}, got {w}x{h}"
            )

    return {
        "left": left,
        "right": right,
    }


def draw_cross(
    image: np.ndarray,
    point: np.ndarray,
    color: tuple[int, int, int],
    *,
    size: int = 10,
    thickness: int = 2,
) -> None:
    u = int(round(float(point[0])))
    v = int(round(float(point[1])))

    cv2.line(
        image,
        (u - size, v),
        (u + size, v),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (u, v - size),
        (u, v + size),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_circle(
    image: np.ndarray,
    point: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int = 6,
) -> None:
    center = (
        int(round(float(point[0]))),
        int(round(float(point[1]))),
    )
    cv2.circle(
        image,
        center,
        radius,
        color,
        -1,
        cv2.LINE_AA,
    )


def draw_label(
    image: np.ndarray,
    point: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    *,
    dy: int,
) -> None:
    h, w = image.shape[:2]

    u = int(round(float(point[0]))) + 8
    v = int(round(float(point[1]))) + dy

    u = max(4, min(w - 180, u))
    v = max(18, min(h - 5, v))

    cv2.putText(
        image,
        text,
        (u, v),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def save_crop(
    raw_image: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    *,
    radius: int,
    scale: int,
    title: str,
) -> None:
    midpoint = 0.5 * (observed + predicted)

    cx = int(round(float(midpoint[0])))
    cy = int(round(float(midpoint[1])))

    h, w = raw_image.shape[:2]

    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(w, cx + radius + 1)
    y1 = min(h, cy + radius + 1)

    crop = raw_image[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return

    obs_local = observed - np.asarray([x0, y0], dtype=np.float64)
    pred_local = predicted - np.asarray([x0, y0], dtype=np.float64)

    cv2.line(
        crop,
        tuple(np.round(obs_local).astype(int)),
        tuple(np.round(pred_local).astype(int)),
        CYAN,
        1,
        cv2.LINE_AA,
    )

    draw_circle(
        crop,
        obs_local,
        YELLOW,
        radius=5,
    )
    draw_cross(
        crop,
        pred_local,
        RED,
        size=8,
        thickness=2,
    )

    enlarged = cv2.resize(
        crop,
        (
            crop.shape[1] * scale,
            crop.shape[0] * scale,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    header = 38
    canvas = np.full(
        (
            enlarged.shape[0] + header,
            enlarged.shape[1],
            3,
        ),
        20,
        dtype=np.uint8,
    )
    canvas[header:] = enlarged

    cv2.putText(
        canvas,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    h = max(
        left.shape[0],
        right.shape[0],
    )
    w = left.shape[1] + right.shape[1]

    canvas = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )
    canvas[
        : left.shape[0],
        : left.shape[1],
    ] = left
    canvas[
        : right.shape[0],
        left.shape[1] : left.shape[1] + right.shape[1],
    ] = right

    return canvas


def main() -> None:
    args = parse_args()

    calibration = load_veo_calibration(
        args.calibration
    )
    images = load_lenses(args)

    overlays = {
        "left": images["left"].copy(),
        "right": images["right"].copy(),
    }

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    crops_dir = args.out_dir / "crops"
    crops_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations = (
        calibration.selected_corner_observations()
    )

    print("=" * 78)
    print(args.calibration)
    print("=" * 78)
    print(
        f"field: {calibration.field_length:.6f} x "
        f"{calibration.field_width:.6f} m"
    )
    print()
    print(
        "YELLOW = Veo selected corner | "
        "RED = canonical projected corner"
    )
    print()

    report_rows: list[dict[str, Any]] = []

    for obs in observations:
        corner = str(obs["corner"])
        lens = str(obs["lens"])

        observed = np.asarray(
            obs["raw_pixel"],
            dtype=np.float64,
        ).reshape(2)

        expected_field = np.asarray(
            obs["expected_field"],
            dtype=np.float64,
        ).reshape(2)

        predicted = calibration.field_to_pixel(
            lens,
            expected_field,
        )

        residual = predicted - observed
        error_px = float(
            np.linalg.norm(residual)
        )

        overlay = overlays[lens]

        cv2.line(
            overlay,
            tuple(np.round(observed).astype(int)),
            tuple(np.round(predicted).astype(int)),
            CYAN,
            1,
            cv2.LINE_AA,
        )

        draw_circle(
            overlay,
            observed,
            YELLOW,
            radius=6,
        )
        draw_cross(
            overlay,
            predicted,
            RED,
            size=10,
            thickness=2,
        )

        draw_label(
            overlay,
            observed,
            f"{corner.upper()} veo",
            YELLOW,
            dy=-10,
        )
        draw_label(
            overlay,
            predicted,
            f"{corner.upper()} pred",
            RED,
            dy=18,
        )

        print(
            f"{corner.upper():>2s} {lens.upper():>5s} | "
            f"veo=({observed[0]:8.3f}, {observed[1]:8.3f}) "
            f"pred=({predicted[0]:8.3f}, {predicted[1]:8.3f}) "
            f"du={residual[0]:+8.3f}px "
            f"dv={residual[1]:+8.3f}px "
            f"err={error_px:8.3f}px"
        )

        save_crop(
            images[lens],
            observed,
            predicted,
            crops_dir / f"{lens}_{corner}.jpg",
            radius=args.crop_radius,
            scale=args.crop_scale,
            title=(
                f"{corner.upper()} {lens.upper()} "
                f"| yellow=Veo red=prediction "
                f"| {error_px:.2f}px"
            ),
        )

        report_rows.append(
            {
                "corner": corner,
                "lens": lens,
                "veo_selected_raw": observed.tolist(),
                "expected_field": expected_field.tolist(),
                "canonical_predicted_raw": predicted.tolist(),
                "residual_raw": residual.tolist(),
                "error_px": error_px,
            }
        )

    left_path = (
        args.out_dir / "left_overlay.jpg"
    )
    right_path = (
        args.out_dir / "right_overlay.jpg"
    )
    contact_path = (
        args.out_dir / "contact_sheet.jpg"
    )
    report_path = (
        args.out_dir / "inspect_selected_corners.json"
    )

    cv2.imwrite(
        str(left_path),
        overlays["left"],
    )
    cv2.imwrite(
        str(right_path),
        overlays["right"],
    )
    cv2.imwrite(
        str(contact_path),
        side_by_side(
            overlays["left"],
            overlays["right"],
        ),
    )

    payload = {
        "calibration": str(args.calibration),
        "field_length": calibration.field_length,
        "field_width": calibration.field_width,
        "note": (
            "Veo selected RAW coordinates are obtained through "
            "VeoCalibration.selected_corner_observations(); no manual "
            "stacked-coordinate conversion is duplicated here."
        ),
        "observations": report_rows,
    }

    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Written:")
    print(f"  {left_path}")
    print(f"  {right_path}")
    print(f"  {contact_path}")
    print(f"  {report_path}")
    print(f"  {crops_dir}/")


if __name__ == "__main__":
    main()
