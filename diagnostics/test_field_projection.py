#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_PATH = (
    ROOT
    / "data"
    / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
)

LEFT_PATH = ROOT / "samples" / "left_frame.jpg"
RIGHT_PATH = ROOT / "samples" / "right_frame.jpg"

OUTPUT_DIR = ROOT / "output" / "field_projection"

TRANSCODE_WIDTH = 2048
TRANSCODE_LENS_HEIGHT = 1024

# Effective coordinate space used by Veo calibration.
CALIBRATION_WIDTH = 3840
CALIBRATION_HEIGHT = 1536

SCALE_X = TRANSCODE_WIDTH / CALIBRATION_WIDTH
SCALE_Y = TRANSCODE_LENS_HEIGHT / CALIBRATION_HEIGHT


def parse_matrix(value: str, rows: int, cols: int) -> np.ndarray:
    values = np.array(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    )
    return values.reshape(rows, cols)


def parse_vector(value: str) -> np.ndarray:
    return np.array(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    )


def project_world_point(
    world_point: np.ndarray,
    lens2camera: np.ndarray,
    camera2world: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """
    Project a 3D point from Veo world coordinates into one raw lens image.
    """

    world_h = np.array(
        [
            world_point[0],
            world_point[1],
            world_point[2],
            1.0,
        ],
        dtype=np.float64,
    )

    world2camera = np.linalg.inv(camera2world)
    camera2lens = np.linalg.inv(lens2camera)

    lens_h = camera2lens @ world2camera @ world_h
    lens_xyz = lens_h[:3]

    if lens_xyz[2] <= 0:
        raise RuntimeError(
            f"Point is behind lens: {world_point} -> {lens_xyz}"
        )

    object_point = lens_xyz.reshape(1, 1, 3)

    image_point, _ = cv2.projectPoints(
        object_point,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        distortion,
    )

    u_cal, v_cal = image_point.reshape(2)

    # Convert from Veo calibration coordinates to downloaded
    # 2048x1024 transcode lens coordinates.
    return np.array(
        [
            u_cal * SCALE_X,
            v_cal * SCALE_Y,
        ],
        dtype=np.float64,
    )


def selected_corner_local(
    data: dict,
    lens_name: str,
) -> np.ndarray:
    nx, ny = data["stacked"]

    x = nx * 2048
    y_stacked = ny * 2048

    if lens_name == "left":
        y = y_stacked
    else:
        y = y_stacked - 1024

    return np.array([x, y], dtype=np.float64)


def draw_cross(
    image: np.ndarray,
    point: np.ndarray,
    color,
    label: str,
):
    x = int(round(point[0]))
    y = int(round(point[1]))

    cv2.drawMarker(
        image,
        (x, y),
        color,
        cv2.MARKER_CROSS,
        28,
        4,
    )

    cv2.putText(
        image,
        label,
        (x + 14, y - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate Veo field projection")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--left", type=Path, default=LEFT_PATH)
    parser.add_argument("--right", type=Path, default=RIGHT_PATH)
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix output filenames, for example match2",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with args.calibration.open("r", encoding="utf-8") as f:
        calibration = json.load(f)

    alignment = calibration["alignment"]
    selected = calibration["selected_corners"]["corners_stacked"]

    field_length = float(alignment["field_length"])
    field_width = float(alignment["field_width"])

    half_length = field_length / 2.0
    half_width = field_width / 2.0

    # Veo world convention inferred from the calibration:
    #
    # X = field length
    # Y = vertical axis, ground plane Y=0
    # Z = field width
    #
    # C = close side, F = far side
    # L = left goal line, R = right goal line
    field_corners = {
        "cl": np.array([-half_length, 0.0, -half_width]),
        "fl": np.array([-half_length, 0.0, +half_width]),
        "fr": np.array([+half_length, 0.0, +half_width]),
        "cr": np.array([+half_length, 0.0, -half_width]),
    }

    camera2world = parse_matrix(
        alignment["camera2world"],
        4,
        4,
    )

    lens_data = {
        "left": {
            "image": cv2.imread(str(args.left)),
            "lens2camera": parse_matrix(
                alignment["lens_left2camera"],
                4,
                4,
            ),
            "K": parse_matrix(
                alignment["intrinsic_left"],
                3,
                3,
            ),
            "distortion": parse_vector(
                alignment["distortion_left"],
            ),
        },
        "right": {
            "image": cv2.imread(str(args.right)),
            "lens2camera": parse_matrix(
                alignment["lens_right2camera"],
                4,
                4,
            ),
            "K": parse_matrix(
                alignment["intrinsic_right"],
                3,
                3,
            ),
            "distortion": parse_vector(
                alignment["distortion_right"],
            ),
        },
    }

    for lens_name, lens in lens_data.items():
        if lens["image"] is None:
            raise RuntimeError(f"Could not load {lens_name} image")

        lens["debug"] = lens["image"].copy()

    errors = []

    for corner_name, world_point in field_corners.items():
        # Determine which lens contains Veo's selected corner.
        if selected[corner_name]["left"]["valid"]:
            lens_name = "left"
        elif selected[corner_name]["right"]["valid"]:
            lens_name = "right"
        else:
            continue

        lens = lens_data[lens_name]

        predicted = project_world_point(
            world_point,
            lens["lens2camera"],
            camera2world,
            lens["K"],
            lens["distortion"],
        )

        observed = selected_corner_local(
            selected[corner_name][lens_name],
            lens_name,
        )

        error = float(np.linalg.norm(predicted - observed))
        errors.append(error)

        print(
            f"{corner_name.upper()} [{lens_name}] "
            f"predicted=({predicted[0]:.2f}, {predicted[1]:.2f}) "
            f"veo=({observed[0]:.2f}, {observed[1]:.2f}) "
            f"error={error:.2f}px"
        )

        # RED = value stored by Veo.
        draw_cross(
            lens["debug"],
            observed,
            (0, 0, 255),
            f"{corner_name.upper()} Veo",
        )

        # GREEN = value reconstructed entirely from the matrices.
        draw_cross(
            lens["debug"],
            predicted,
            (0, 255, 0),
            f"{corner_name.upper()} projected",
        )

        cv2.line(
            lens["debug"],
            tuple(np.round(observed).astype(int)),
            tuple(np.round(predicted).astype(int)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    print()
    print(f"Mean error: {np.mean(errors):.2f}px")
    print(f"Max error:  {np.max(errors):.2f}px")

    for lens_name, lens in lens_data.items():
        prefix = f"{args.prefix}_" if args.prefix else ""
        output = OUTPUT_DIR / f"{prefix}{lens_name}_field_projection.jpg"

        cv2.imwrite(
            str(output),
            lens["debug"],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        print(f"Saved {output}")


if __name__ == "__main__":
    main()
