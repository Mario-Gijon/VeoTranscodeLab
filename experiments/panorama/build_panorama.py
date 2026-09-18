#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import load_veo_metadata

CALIBRATION_PATH = (
    ROOT
    / "data"
    / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
)

LEFT_PATH = ROOT / "samples" / "left_frame.jpg"
RIGHT_PATH = ROOT / "samples" / "right_frame.jpg"

OUTPUT_DIR = ROOT / "output" / "panorama"

# Very wide on purpose.
OUTPUT_WIDTH = 3072
OUTPUT_HEIGHT = 810

# Effective coordinate space inferred from the Veo calibration.
CALIBRATION_WIDTH = 3840.0
CALIBRATION_HEIGHT = 1536.0

SOURCE_WIDTH = 2048.0
SOURCE_HEIGHT = 1024.0

SCALE_X = SOURCE_WIDTH / CALIBRATION_WIDTH
SCALE_Y = SOURCE_HEIGHT / CALIBRATION_HEIGHT


def build_lens_map(
    lens2camera,
    camera2presentation,
    K,
    distortion,
    va,
    width,
    height,
):
    """
    Build an inverse panorama -> raw lens map.

    The panorama is represented as yaw/pitch angles in Veo's
    presentation coordinate system.

    For every panorama pixel:
        presentation ray
        -> camera ray
        -> lens ray
        -> distorted raw image pixel
    """

    map_x = np.full(
        (height, width),
        -1,
        dtype=np.float32,
    )

    map_y = np.full(
        (height, width),
        -1,
        dtype=np.float32,
    )

    quality = np.full(
        (height, width),
        -1,
        dtype=np.float32,
    )

    yaw_values = np.linspace(
        va["left"],
        va["right"],
        width,
        dtype=np.float64,
    )

    presentation2camera = np.linalg.inv(
        camera2presentation
    )

    camera2lens = np.linalg.inv(
        lens2camera
    )

    R_presentation2camera = presentation2camera[:3, :3]
    R_camera2lens = camera2lens[:3, :3]

    zero_rotation = np.zeros(
        3,
        dtype=np.float64,
    )

    zero_translation = np.zeros(
        3,
        dtype=np.float64,
    )

    for y in range(height):
        t = y / max(height - 1, 1)

        pitch = (
            va["top"]
            + t * (va["bottom"] - va["top"])
        )

        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)

        sin_yaw = np.sin(yaw_values)
        cos_yaw = np.cos(yaw_values)

        # Presentation frame.
        #
        # X = right
        # Y = down
        # Z = forward
        #
        # Positive pitch therefore means looking upward,
        # hence the negative Y component.
        rays_presentation = np.stack(
            [
                sin_yaw * cos_pitch,
                np.full_like(
                    yaw_values,
                    -sin_pitch,
                ),
                cos_yaw * cos_pitch,
            ],
            axis=1,
        )

        # Row-vector form of R @ column_vector.
        rays_camera = (
            rays_presentation
            @ R_presentation2camera.T
        )

        rays_lens = (
            rays_camera
            @ R_camera2lens.T
        )

        z = rays_lens[:, 2]

        in_front = z > 1e-6

        projected, _ = cv2.projectPoints(
            rays_lens.reshape(-1, 1, 3),
            zero_rotation,
            zero_translation,
            K,
            distortion,
        )

        projected = projected.reshape(-1, 2)

        u = projected[:, 0] * SCALE_X
        v = projected[:, 1] * SCALE_Y

        valid = (
            in_front
            & (u >= 0)
            & (u < SOURCE_WIDTH - 1)
            & (v >= 0)
            & (v < SOURCE_HEIGHT - 1)
        )

        map_x[y, valid] = u[valid].astype(
            np.float32
        )

        map_y[y, valid] = v[valid].astype(
            np.float32
        )

        # Cosine-like quality measure:
        # larger means the ray is closer to the optical axis
        # of this lens.
        norms = np.linalg.norm(
            rays_lens,
            axis=1,
        )

        q = z / np.maximum(norms, 1e-12)

        quality[y, valid] = q[valid].astype(
            np.float32
        )

    return map_x, map_y, quality


def warp_image(image, map_x, map_y):
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_mask(path, mask):
    cv2.imwrite(
        str(path),
        (mask.astype(np.uint8) * 255),
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    left = cv2.imread(str(LEFT_PATH))
    right = cv2.imread(str(RIGHT_PATH))

    if left is None:
        raise RuntimeError(
            f"Could not load {LEFT_PATH}"
        )

    if right is None:
        raise RuntimeError(
            f"Could not load {RIGHT_PATH}"
        )

    calibration = load_veo_metadata(CALIBRATION_PATH)
    va = calibration.va_presentation

    print("Presentation angular range:")
    print(
        f"  horizontal: "
        f"{np.degrees(va['left']):.2f}° "
        f"to {np.degrees(va['right']):.2f}°"
    )
    print(
        f"  vertical:   "
        f"{np.degrees(va['bottom']):.2f}° "
        f"to {np.degrees(va['top']):.2f}°"
    )
    print()

    print("Building LEFT panorama map...")

    left_map_x, left_map_y, left_quality = (
        build_lens_map(
            calibration.left.lens2camera,
            calibration.camera2presentation,
            calibration.left.intrinsic,
            calibration.left.distortion,
            va,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
        )
    )

    print("Building RIGHT panorama map...")

    right_map_x, right_map_y, right_quality = (
        build_lens_map(
            calibration.right.lens2camera,
            calibration.camera2presentation,
            calibration.right.intrinsic,
            calibration.right.distortion,
            va,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
        )
    )

    print("Warping images...")

    left_warp = warp_image(
        left,
        left_map_x,
        left_map_y,
    )

    right_warp = warp_image(
        right,
        right_map_x,
        right_map_y,
    )

    left_valid = left_map_x >= 0
    right_valid = right_map_x >= 0

    overlap = left_valid & right_valid

    #
    # Version 1: hard selection.
    #
    # Use whichever lens observes a direction closer
    # to its optical axis.
    #
    use_left = (
        left_valid
        & (
            ~right_valid
            | (left_quality >= right_quality)
        )
    )

    use_right = (
        right_valid
        & ~use_left
    )

    hard = np.zeros_like(left_warp)

    hard[use_left] = left_warp[use_left]
    hard[use_right] = right_warp[use_right]

    #
    # Version 2: simple weighted blend in overlap.
    #
    blend = np.zeros_like(
        left_warp,
        dtype=np.float32,
    )

    wl = np.where(
        left_valid,
        np.maximum(left_quality, 0.0) ** 4,
        0.0,
    )

    wr = np.where(
        right_valid,
        np.maximum(right_quality, 0.0) ** 4,
        0.0,
    )

    weight_sum = wl + wr

    valid_any = weight_sum > 1e-8

    blend[valid_any] = (
        left_warp[valid_any].astype(np.float32)
        * wl[valid_any, None]
        + right_warp[valid_any].astype(np.float32)
        * wr[valid_any, None]
    ) / weight_sum[valid_any, None]

    blend = np.clip(
        blend,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(OUTPUT_DIR / "left_warp.jpg"),
        left_warp,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "right_warp.jpg"),
        right_warp,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "panorama_hard.jpg"),
        hard,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "panorama_blend.jpg"),
        blend,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    save_mask(
        OUTPUT_DIR / "left_valid.png",
        left_valid,
    )

    save_mask(
        OUTPUT_DIR / "right_valid.png",
        right_valid,
    )

    save_mask(
        OUTPUT_DIR / "overlap.png",
        overlap,
    )

    total_pixels = OUTPUT_WIDTH * OUTPUT_HEIGHT

    print()
    print(
        "Left coverage:  "
        f"{100 * left_valid.sum() / total_pixels:.1f}%"
    )

    print(
        "Right coverage: "
        f"{100 * right_valid.sum() / total_pixels:.1f}%"
    )

    print(
        "Overlap:        "
        f"{100 * overlap.sum() / total_pixels:.1f}%"
    )

    print()
    print(
        f"Saved panorama outputs to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
