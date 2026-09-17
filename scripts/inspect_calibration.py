#!/usr/bin/env python3

import json
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
    frame = cv2.imread(str(FRAME_PATH))

    if frame is None:
        raise RuntimeError(f"Could not read {FRAME_PATH}")

    h, w = frame.shape[:2]

    print(f"Stacked frame: {w}x{h}")

    if h % 2 != 0:
        raise RuntimeError("Stacked frame height is not divisible by two")

    split_y = h // 2

    left = frame[:split_y].copy()
    right = frame[split_y:].copy()

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
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

    cv2.imwrite(str(STACKED_OUTPUT), stacked_debug)
    cv2.imwrite(str(LEFT_OUTPUT), left)
    cv2.imwrite(str(RIGHT_OUTPUT), right)
    cv2.imwrite(str(LEFT_CORNERS_OUTPUT), left_debug)
    cv2.imwrite(str(RIGHT_CORNERS_OUTPUT), right_debug)

    print()
    print(f"Saved: {STACKED_OUTPUT}")
    print(f"Saved: {LEFT_OUTPUT}")
    print(f"Saved: {RIGHT_OUTPUT}")
    print(f"Saved: {LEFT_CORNERS_OUTPUT}")
    print(f"Saved: {RIGHT_CORNERS_OUTPUT}")


if __name__ == "__main__":
    main()
