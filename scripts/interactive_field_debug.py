#!/usr/bin/env python3

"""Small interactive pixel-to-field inspection tool for the Veo lab."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from test_pixel_to_field import parse_matrix, parse_vector, pixel_to_field


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATHS = {
    "left": ROOT / "samples" / "left_frame.jpg",
    "right": ROOT / "samples" / "right_frame.jpg",
}
IMAGE_WINDOW = "Veo lens image"
FIELD_WINDOW = "Projected football field"
FIELD_VIEW_SIZE = (960, 620)  # width, height
FIELD_MARGIN = 70


def find_calibration_path() -> Path:
    calibration_paths = sorted((ROOT / "data").glob("*.veo"))
    if len(calibration_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one .veo calibration file in {ROOT / 'data'}, "
            f"found {len(calibration_paths)}"
        )
    return calibration_paths[0]


def load_configuration(lens_name: str):
    image = cv2.imread(str(IMAGE_PATHS[lens_name]))
    if image is None:
        raise RuntimeError(f"Could not load {IMAGE_PATHS[lens_name]}")

    with find_calibration_path().open("r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)

    alignment = calibration["alignment"]
    camera2world = parse_matrix(alignment["camera2world"], 4, 4)
    lens = {
        "K": parse_matrix(alignment[f"intrinsic_{lens_name}"], 3, 3),
        "distortion": parse_vector(alignment[f"distortion_{lens_name}"]),
        "lens2camera": parse_matrix(
            alignment[f"lens_{lens_name}2camera"], 4, 4
        ),
    }

    field_length = float(alignment["field_length"])
    field_width = float(alignment["field_width"])
    return image, lens, camera2world, field_length, field_width


def field_to_canvas(x: float, z: float, field_length: float, field_width: float):
    canvas_width, canvas_height = FIELD_VIEW_SIZE
    usable_width = canvas_width - 2 * FIELD_MARGIN
    usable_height = canvas_height - 2 * FIELD_MARGIN
    scale = min(usable_width / field_length, usable_height / field_width)
    origin_x = canvas_width / 2.0
    origin_z = canvas_height / 2.0
    return (
        int(round(origin_x + x * scale)),
        int(round(origin_z - z * scale)),
    )


def draw_field(
    field_length: float,
    field_width: float,
    projected_points: list[tuple[float, float, bool]],
) -> np.ndarray:
    canvas_width, canvas_height = FIELD_VIEW_SIZE
    field = np.full((canvas_height, canvas_width, 3), 35, dtype=np.uint8)
    half_length = field_length / 2.0
    half_width = field_width / 2.0

    corners = [
        field_to_canvas(-half_length, -half_width, field_length, field_width),
        field_to_canvas(-half_length, +half_width, field_length, field_width),
        field_to_canvas(+half_length, +half_width, field_length, field_width),
        field_to_canvas(+half_length, -half_width, field_length, field_width),
    ]
    polygon = np.array(corners, dtype=np.int32)
    cv2.fillConvexPoly(field, polygon, (45, 125, 45))
    cv2.polylines(field, [polygon], True, (235, 235, 235), 3, cv2.LINE_AA)

    halfway_top = field_to_canvas(0.0, +half_width, field_length, field_width)
    halfway_bottom = field_to_canvas(0.0, -half_width, field_length, field_width)
    cv2.line(field, halfway_top, halfway_bottom, (235, 235, 235), 2, cv2.LINE_AA)

    cv2.putText(
        field,
        f"Field view: X length {field_length:.2f} m, Z width {field_width:.2f} m",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(field, "X", (canvas_width - 35, canvas_height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
    cv2.putText(field, "Z", (canvas_width // 2 + 10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)

    for index, (x, z, inside) in enumerate(projected_points, start=1):
        point = field_to_canvas(x, z, field_length, field_width)
        color = (0, 220, 255) if inside else (0, 70, 255)
        cv2.drawMarker(field, point, color, cv2.MARKER_CROSS, 20, 3)
        cv2.putText(field, str(index), (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    return field


def print_projection(lens_name: str, pixel: np.ndarray, world: np.ndarray,
                     field_length: float, field_width: float) -> bool:
    x, z = float(world[0]), float(world[2])
    inside = (
        -field_length / 2.0 <= x <= field_length / 2.0
        and -field_width / 2.0 <= z <= field_width / 2.0
    )
    print(f"LENS: {lens_name}")
    print(f"PIXEL (u,v): ({pixel[0]:.2f}, {pixel[1]:.2f})")
    print(f"VEO WORLD: X={world[0]:.3f}, Y={world[1]:.3f}, Z={world[2]:.3f}")
    print(f"FIELD: X={x:.3f} m, Z={z:.3f} m")
    print(f"INSIDE FIELD: {'YES' if inside else 'NO'}")
    print()
    return inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", choices=("left", "right"), required=True)
    args = parser.parse_args()

    image, lens, camera2world, field_length, field_width = load_configuration(args.lens)
    image_view = image.copy()
    projected_points: list[tuple[float, float, bool]] = []

    def refresh_field() -> None:
        cv2.imshow(
            FIELD_WINDOW,
            draw_field(field_length, field_width, projected_points),
        )

    def on_click(event, u, v, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pixel = np.array([float(u), float(v)], dtype=np.float64)
        try:
            world = pixel_to_field(
                pixel,
                lens["K"],
                lens["distortion"],
                lens["lens2camera"],
                camera2world,
            )
        except RuntimeError as error:
            print(f"LENS: {args.lens}")
            print(f"PIXEL (u,v): ({u:.2f}, {v:.2f})")
            print(f"PROJECTION ERROR: {error}\n")
            return

        inside = print_projection(args.lens, pixel, world, field_length, field_width)
        projected_points.append((float(world[0]), float(world[2]), inside))
        index = len(projected_points)
        cv2.drawMarker(image_view, (u, v), (0, 220, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.putText(image_view, str(index), (u + 12, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        refresh_field()

    # Keep the image window at native size so callback coordinates are raw
    # 2048x1024 transcode pixels, matching pixel_to_field's convention.
    cv2.namedWindow(IMAGE_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(FIELD_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(IMAGE_WINDOW, on_click)
    cv2.imshow(IMAGE_WINDOW, image_view)
    refresh_field()
    print("Click points in the image. Press 'c' to clear or 'q'/Escape to quit.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            projected_points.clear()
            image_view[:] = image
            cv2.imshow(IMAGE_WINDOW, image_view)
            refresh_field()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
