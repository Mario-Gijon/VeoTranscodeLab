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

LEFT_PATH = ROOT / "samples" / "left_frame.jpg"
RIGHT_PATH = ROOT / "samples" / "right_frame.jpg"

OUTPUT_DIR = ROOT / "output" / "undistortion"


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


def transform_intrinsics(
    K: np.ndarray,
    sx: float,
    sy: float,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """
    Transform an intrinsic matrix after resizing/cropping the image.

    u' = sx * u + tx
    v' = sy * v + ty
    """
    result = K.copy()

    result[0, 0] *= sx
    result[1, 1] *= sy

    result[0, 2] = result[0, 2] * sx + tx
    result[1, 2] = result[1, 2] * sy + ty

    return result


def build_candidates(K: np.ndarray):
    # Calibration coordinate space inferred from Veo geometry.
    # The selected field corners indicate that the intrinsic matrix
    # behaves as if it were defined on a 3840x1536 image.
    veo_calibrated = transform_intrinsics(
        K,
        sx=2048 / 3840,
        sy=1024 / 1536,
    )
  
    # Hypothesis A:
    # Original calibration is UHD 3840x2160 and the image was
    # independently resized to 2048x1024.
    stretch = transform_intrinsics(
        K,
        sx=2048 / 3840,
        sy=1024 / 2160,
    )

    # Hypothesis B:
    # Original calibration is UHD 3840x2160.
    # Resize preserving aspect ratio:
    #
    # 3840x2160 -> 2048x1152
    #
    # and then center-crop vertically:
    #
    # 2048x1152 -> 2048x1024
    #
    scale = 2048 / 3840
    scaled_height = 2160 * scale
    crop_y = (scaled_height - 1024) / 2

    center_crop = transform_intrinsics(
        K,
        sx=scale,
        sy=scale,
        ty=-crop_y,
    )

    # Hypothesis C:
    # Calibration coordinates correspond to a 4096x2048 image
    # and the transcode is exactly half resolution.
    half_4096 = transform_intrinsics(
        K,
        sx=0.5,
        sy=0.5,
    )

    return {
        "veo_3840x1536": veo_calibrated,
        "uhd_stretch": stretch,
        "uhd_center_crop": center_crop,
        "4096_half": half_4096,
    }


def undistort_image(
    image: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    h, w = image.shape[:2]

    map_x, map_y = cv2.initUndistortRectifyMap(
        K,
        distortion,
        np.eye(3, dtype=np.float64),
        K,
        (w, h),
        cv2.CV_32FC1,
    )

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    left = cv2.imread(str(LEFT_PATH))
    right = cv2.imread(str(RIGHT_PATH))

    if left is None:
        raise RuntimeError(f"Could not load {LEFT_PATH}")

    if right is None:
        raise RuntimeError(f"Could not load {RIGHT_PATH}")

    print(f"Left image:  {left.shape[1]}x{left.shape[0]}")
    print(f"Right image: {right.shape[1]}x{right.shape[0]}")

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    alignment = calibration["alignment"]

    K_left = parse_matrix(
        alignment["intrinsic_left"],
        3,
        3,
    )

    K_right = parse_matrix(
        alignment["intrinsic_right"],
        3,
        3,
    )

    distortion_left = parse_vector(
        alignment["distortion_left"]
    )

    distortion_right = parse_vector(
        alignment["distortion_right"]
    )

    print()
    print("Raw left K:")
    print(K_left)

    print()
    print("Raw right K:")
    print(K_right)

    print()
    print(
        f"Distortion coefficients: "
        f"{len(distortion_left)}"
    )

    candidates_left = build_candidates(K_left)
    candidates_right = build_candidates(K_right)

    for name in candidates_left:
        print()
        print("=" * 60)
        print(name)

        Kl = candidates_left[name]
        Kr = candidates_right[name]

        print()
        print("Left K:")
        print(Kl)

        print()
        print("Right K:")
        print(Kr)

        left_undistorted = undistort_image(
            left,
            Kl,
            distortion_left,
        )

        right_undistorted = undistort_image(
            right,
            Kr,
            distortion_right,
        )

        left_output = OUTPUT_DIR / f"{name}_left.jpg"
        right_output = OUTPUT_DIR / f"{name}_right.jpg"

        cv2.imwrite(
            str(left_output),
            left_undistorted,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        cv2.imwrite(
            str(right_output),
            right_undistorted,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        print(f"Saved {left_output}")
        print(f"Saved {right_output}")


if __name__ == "__main__":
    main()
