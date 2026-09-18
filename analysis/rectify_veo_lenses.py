#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_WIDTH = 3840.0
CALIBRATION_HEIGHT = 1920.0


def parse_numeric_array(value):
    if isinstance(value, str):
        return np.fromstring(value, sep=",", dtype=np.float64)

    return np.asarray(value, dtype=np.float64).reshape(-1)


def load_optics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    alignment = document["alignment"]

    result = {}

    for lens in ("left", "right"):
        k = parse_numeric_array(
            alignment[f"intrinsic_{lens}"]
        ).reshape(3, 3)

        d = parse_numeric_array(
            alignment[f"distortion_{lens}"]
        )

        result[lens] = {
            "K": k,
            "D": d,
        }

    return result


def scale_intrinsics(
    k: np.ndarray,
    raw_width: int,
    raw_height: int,
):
    sx = raw_width / CALIBRATION_WIDTH
    sy = raw_height / CALIBRATION_HEIGHT

    k_raw = k.copy()

    k_raw[0, 0] *= sx
    k_raw[1, 1] *= sy

    k_raw[0, 2] *= sx
    k_raw[1, 2] *= sy

    return k_raw


def rectify_lens(
    image: np.ndarray,
    k_raw: np.ndarray,
    distortion: np.ndarray,
    alpha: float,
):
    h, w = image.shape[:2]

    new_k, roi = cv2.getOptimalNewCameraMatrix(
        k_raw,
        distortion,
        (w, h),
        alpha,
        (w, h),
        centerPrincipalPoint=False,
    )

    map_x, map_y = cv2.initUndistortRectifyMap(
        k_raw,
        distortion,
        None,
        new_k,
        (w, h),
        cv2.CV_32FC1,
    )

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return rectified, new_k, roi


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stacked",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/rectified_veo"),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=(
            "0 crops aggressively, "
            "1 preserves maximum field of view."
        ),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stacked = cv2.imread(
        str(args.stacked)
    )

    if stacked is None:
        raise RuntimeError(
            f"Could not load {args.stacked}"
        )

    h, w = stacked.shape[:2]

    if h % 2 != 0:
        raise ValueError(
            "Stacked image height must be even."
        )

    lens_h = h // 2

    images = {
        "left": stacked[:lens_h].copy(),
        "right": stacked[lens_h:].copy(),
    }

    optics = load_optics(
        args.calibration
    )

    rectified = {}

    print(
        f"RAW lens size: {w}x{lens_h}"
    )

    print(
        f"Assumed Veo calibration space: "
        f"{int(CALIBRATION_WIDTH)}x"
        f"{int(CALIBRATION_HEIGHT)}"
    )

    print()

    for lens in ("left", "right"):
        k_raw = scale_intrinsics(
            optics[lens]["K"],
            w,
            lens_h,
        )

        corrected, new_k, roi = rectify_lens(
            images[lens],
            k_raw,
            optics[lens]["D"],
            args.alpha,
        )

        rectified[lens] = corrected

        cv2.imwrite(
            str(
                args.out
                / f"{lens}_raw.jpg"
            ),
            images[lens],
        )

        cv2.imwrite(
            str(
                args.out
                / f"{lens}_rectified.jpg"
            ),
            corrected,
        )

        print(
            f"{lens.upper()}"
        )

        print(
            "K RAW:"
        )

        print(k_raw)

        print(
            "K RECTIFIED:"
        )

        print(new_k)

        print(
            f"ROI: {roi}"
        )

        print()

    combined = np.hstack(
        [
            rectified["left"],
            rectified["right"],
        ]
    )

    cv2.imwrite(
        str(
            args.out
            / "rectified_side_by_side.jpg"
        ),
        combined,
    )

    print(
        f"Saved results to {args.out}"
    )


if __name__ == "__main__":
    main()
