#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

FRAME_PATH = ROOT / "samples" / "stacked_frame.jpg"
CALIBRATION_PATH = (
    ROOT
    / "data"
    / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
)

STACKED_OUTPUT = ROOT / "samples" / "stacked_with_corners.jpg"
LEFT_OUTPUT = ROOT / "samples" / "left_frame.jpg"
RIGHT_OUTPUT = ROOT / "samples" / "right_frame.jpg"
LEFT_CORNERS_OUTPUT = ROOT / "samples" / "left_with_corners.jpg"
RIGHT_CORNERS_OUTPUT = ROOT / "samples" / "right_with_corners.jpg"


def draw_point(img, x, y, label):
    p = (int(round(x)), int(round(y)))

    cv2.circle(img, p, 14, (0, 0, 255), 4)

    cv2.putText(
        img,
        label,
        (p[0] + 18, p[1] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect selected Veo calibration corners")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--stacked", type=Path, default=FRAME_PATH)
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix output filenames, for example match2",
    )
    args = parser.parse_args()

    frame_path = args.stacked
    calibration_path = args.calibration
    output_prefix = f"{args.prefix}_" if args.prefix else ""
    output_dir = frame_path.parent
    stacked_output = output_dir / f"{output_prefix}stacked_with_corners.jpg"
    left_output = output_dir / f"{output_prefix}left_frame.jpg"
    right_output = output_dir / f"{output_prefix}right_frame.jpg"
    left_corners_output = output_dir / f"{output_prefix}left_with_corners.jpg"
    right_corners_output = output_dir / f"{output_prefix}right_with_corners.jpg"

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise RuntimeError(f"Could not read {frame_path}")

    h, w = frame.shape[:2]

    print(f"Stacked frame: {w}x{h}")

    if h % 2 != 0:
        raise RuntimeError("Stacked frame height is not divisible by two")

    split_y = h // 2

    left = frame[:split_y].copy()
    right = frame[split_y:].copy()

    with calibration_path.open("r", encoding="utf-8") as f:
        calibration = json.load(f)

    corners = calibration["selected_corners"]["corners_stacked"]

    stacked_debug = frame.copy()
    left_debug = left.copy()
    right_debug = right.copy()

    for name in ("cl", "cr", "fl", "fr"):
        corner = corners[name]

        for lens_name in ("left", "right"):
            data = corner[lens_name]

            if not data["valid"]:
                continue

            nx, ny = data["stacked"]

            # Coordinates in complete 2048x2048 stacked image.
            x_stack = nx * w
            y_stack = ny * h

            print(
                f"{name.upper()} ({lens_name}): "
                f"normalized=({nx:.6f}, {ny:.6f}) "
                f"stacked_px=({x_stack:.1f}, {y_stack:.1f})"
            )

            draw_point(
                stacked_debug,
                x_stack,
                y_stack,
                f"{name.upper()} {lens_name}",
            )

            if lens_name == "left":
                x_local = x_stack
                y_local = y_stack

                draw_point(
                    left_debug,
                    x_local,
                    y_local,
                    name.upper(),
                )

            else:
                x_local = x_stack
                y_local = y_stack - split_y

                draw_point(
                    right_debug,
                    x_local,
                    y_local,
                    name.upper(),
                )

    cv2.imwrite(str(stacked_output), stacked_debug)
    cv2.imwrite(str(left_output), left)
    cv2.imwrite(str(right_output), right)
    cv2.imwrite(str(left_corners_output), left_debug)
    cv2.imwrite(str(right_corners_output), right_debug)

    print()
    print(f"Saved: {stacked_output}")
    print(f"Saved: {left_output}")
    print(f"Saved: {right_output}")
    print(f"Saved: {left_corners_output}")
    print(f"Saved: {right_corners_output}")


if __name__ == "__main__":
    main()
