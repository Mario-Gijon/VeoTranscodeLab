#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_PATH = (
    ROOT
    / "data"
    / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
)

TRANSCODE_WIDTH = 2048
LENS_HEIGHT = 1024

CALIBRATION_WIDTH = 3840
CALIBRATION_HEIGHT = 1536

SCALE_X = TRANSCODE_WIDTH / CALIBRATION_WIDTH
SCALE_Y = LENS_HEIGHT / CALIBRATION_HEIGHT


def parse_matrix(value, rows, cols):
    return np.array(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    ).reshape(rows, cols)


def parse_vector(value):
    return np.array(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    )


def pixel_to_field(
    pixel,
    K,
    distortion,
    lens2camera,
    camera2world,
):
    """
    Raw 2048x1024 transcode pixel -> intersection with
    Veo ground plane Y=0.

    Returns world [X, Y, Z].
    """

    u, v = pixel

    # Transcode coordinates -> Veo calibration coordinates.
    u_cal = u / SCALE_X
    v_cal = v / SCALE_Y

    distorted = np.array(
        [[[u_cal, v_cal]]],
        dtype=np.float64,
    )

    # Remove optical distortion and obtain normalized lens ray.
    normalized = cv2.undistortPoints(
        distorted,
        K,
        distortion,
    ).reshape(2)

    ray_lens = np.array(
        [
            normalized[0],
            normalized[1],
            1.0,
        ],
        dtype=np.float64,
    )

    # Lens coordinate system -> Veo world.
    lens2world = camera2world @ lens2camera

    ray_origin = lens2world[:3, 3]

    ray_direction = (
        lens2world[:3, :3] @ ray_lens
    )

    # Intersect with field plane Y = 0.
    if abs(ray_direction[1]) < 1e-9:
        raise RuntimeError("Ray is parallel to ground plane")

    t = -ray_origin[1] / ray_direction[1]

    if t <= 0:
        raise RuntimeError(
            "Ground intersection is behind the camera"
        )

    world_point = (
        ray_origin + t * ray_direction
    )

    return world_point


def selected_corner_local(data, lens_name):
    nx, ny = data["stacked"]

    x = nx * 2048
    y = ny * 2048

    if lens_name == "right":
        y -= 1024

    return np.array([x, y], dtype=np.float64)


def main():
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    alignment = calibration["alignment"]
    selected = calibration["selected_corners"]["corners_stacked"]

    field_length = float(alignment["field_length"])
    field_width = float(alignment["field_width"])

    half_length = field_length / 2
    half_width = field_width / 2

    camera2world = parse_matrix(
        alignment["camera2world"],
        4,
        4,
    )

    lenses = {
        "left": {
            "K": parse_matrix(
                alignment["intrinsic_left"],
                3,
                3,
            ),
            "distortion": parse_vector(
                alignment["distortion_left"],
            ),
            "lens2camera": parse_matrix(
                alignment["lens_left2camera"],
                4,
                4,
            ),
        },
        "right": {
            "K": parse_matrix(
                alignment["intrinsic_right"],
                3,
                3,
            ),
            "distortion": parse_vector(
                alignment["distortion_right"],
            ),
            "lens2camera": parse_matrix(
                alignment["lens_right2camera"],
                4,
                4,
            ),
        },
    }

    expected = {
        "cl": np.array(
            [-half_length, 0.0, -half_width]
        ),
        "fl": np.array(
            [-half_length, 0.0, +half_width]
        ),
        "fr": np.array(
            [+half_length, 0.0, +half_width]
        ),
        "cr": np.array(
            [+half_length, 0.0, -half_width]
        ),
    }

    errors = []

    print(
        f"Field: {field_length:.3f} x "
        f"{field_width:.3f} m"
    )
    print()

    for name in ("cl", "fl", "fr", "cr"):
        if selected[name]["left"]["valid"]:
            lens_name = "left"
        elif selected[name]["right"]["valid"]:
            lens_name = "right"
        else:
            continue

        pixel = selected_corner_local(
            selected[name][lens_name],
            lens_name,
        )

        lens = lenses[lens_name]

        recovered = pixel_to_field(
            pixel,
            lens["K"],
            lens["distortion"],
            lens["lens2camera"],
            camera2world,
        )

        target = expected[name]

        error = np.linalg.norm(
            recovered[[0, 2]]
            - target[[0, 2]]
        )

        errors.append(error)

        print(
            f"{name.upper()} [{lens_name}]"
        )
        print(
            f"  pixel:      "
            f"({pixel[0]:.2f}, {pixel[1]:.2f})"
        )
        print(
            f"  recovered:  "
            f"X={recovered[0]:.3f} "
            f"Z={recovered[2]:.3f}"
        )
        print(
            f"  expected:   "
            f"X={target[0]:.3f} "
            f"Z={target[2]:.3f}"
        )
        print(
            f"  error:      {error:.3f} m"
        )
        print()

    print(
        f"Mean field error: "
        f"{np.mean(errors):.3f} m"
    )
    print(
        f"Max field error:  "
        f"{np.max(errors):.3f} m"
    )


if __name__ == "__main__":
    main()
