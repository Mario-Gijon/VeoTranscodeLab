#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from analysis.veo_metadata import load_veo_metadata
except ModuleNotFoundError:
    from veo_metadata import load_veo_metadata


DEFAULT_CALIBRATION_WIDTH = 3840.0
DEFAULT_CALIBRATION_HEIGHT = 1920.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a canonical rectilinear view from one Veo RAW lens "
            "using the optical calibration stored in the .veo file."
        )
    )

    parser.add_argument("--stacked", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)

    parser.add_argument(
        "--lens",
        required=True,
        choices=("left", "right"),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/canonical_rectification"),
    )

    parser.add_argument(
        "--calibration-width",
        type=float,
        default=DEFAULT_CALIBRATION_WIDTH,
    )

    parser.add_argument(
        "--calibration-height",
        type=float,
        default=DEFAULT_CALIBRATION_HEIGHT,
    )

    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help=(
            "Virtual rectified-camera focal multiplier. "
            "Lower values preserve more FOV; higher values zoom/crop."
        ),
    )

    parser.add_argument(
        "--center-x",
        type=float,
        default=0.5,
        help="Rectified principal point as fraction of output width.",
    )

    parser.add_argument(
        "--center-y",
        type=float,
        default=0.5,
        help="Rectified principal point as fraction of output height.",
    )

    return parser.parse_args()


def split_lenses(stacked: np.ndarray) -> dict[str, np.ndarray]:
    height, width = stacked.shape[:2]

    if height % 2:
        raise ValueError(
            f"Stacked frame height must be even, got {height}"
        )

    lens_height = height // 2

    return {
        "left": stacked[:lens_height].copy(),
        "right": stacked[lens_height:].copy(),
    }


def build_raw_intrinsics(
    veo_k: np.ndarray,
    raw_width: int,
    raw_height: int,
    calibration_width: float,
    calibration_height: float,
) -> tuple[np.ndarray, float, float]:

    scale_x = raw_width / calibration_width
    scale_y = raw_height / calibration_height

    k_raw = veo_k.astype(np.float64).copy()

    k_raw[0, 0] *= scale_x
    k_raw[0, 2] *= scale_x

    k_raw[1, 1] *= scale_y
    k_raw[1, 2] *= scale_y

    return k_raw, scale_x, scale_y


def build_rectified_intrinsics(
    k_raw: np.ndarray,
    output_width: int,
    output_height: int,
    focal_scale: float,
    center_x: float,
    center_y: float,
) -> np.ndarray:

    # After the 3840x1920 -> 2048x1024 hypothesis, fx and fy
    # should already be very similar. Use their geometric mean
    # as the focal length of our virtual pinhole camera.
    base_focal = float(
        np.sqrt(k_raw[0, 0] * k_raw[1, 1])
    )

    focal = base_focal * focal_scale

    k_rect = np.array(
        [
            [
                focal,
                0.0,
                center_x * output_width,
            ],
            [
                0.0,
                focal,
                center_y * output_height,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    return k_rect


def rectify_image(
    raw_image: np.ndarray,
    k_raw: np.ndarray,
    distortion: np.ndarray,
    k_rect: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:

    map_x, map_y = cv2.initUndistortRectifyMap(
        k_raw,
        distortion,
        np.eye(3, dtype=np.float64),
        k_rect,
        output_size,
        cv2.CV_32FC1,
    )

    rectified = cv2.remap(
        raw_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    source_mask = np.full(
        raw_image.shape[:2],
        255,
        dtype=np.uint8,
    )

    valid_mask = cv2.remap(
        source_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return rectified, valid_mask


def rectify_points(
    points: np.ndarray,
    k_raw: np.ndarray,
    distortion: np.ndarray,
    k_rect: np.ndarray,
) -> np.ndarray:

    points = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    rectified = cv2.undistortPoints(
        points,
        k_raw,
        distortion,
        R=np.eye(3, dtype=np.float64),
        P=k_rect,
    )

    return rectified.reshape(-1, 2)


def fit_line_metrics(points: np.ndarray) -> dict | None:
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if len(points) < 3:
        return None

    center = points.mean(axis=0)
    centered = points - center

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    direction = vh[0]

    normal = np.array(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    residuals = centered @ normal
    along = centered @ direction

    span = float(np.ptp(along))

    if span <= 1e-9:
        return None

    rms = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    return {
        "rms_px": rms,
        "max_px": float(
            np.max(
                np.abs(residuals)
            )
        ),
        "span_px": span,
        "normalized_rms": rms / span,
    }


def draw_line_fit(
    image: np.ndarray,
    points: np.ndarray,
    label: str,
) -> None:

    if len(points) == 0:
        return

    h, w = image.shape[:2]

    for x, y in points:
        ix = int(round(x))
        iy = int(round(y))

        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(
                image,
                (ix, iy),
                4,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )

    if len(points) >= 2:
        fitted = cv2.fitLine(
            points.astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).reshape(-1)

        vx, vy, x0, y0 = fitted

        distance = 10000

        p1 = (
            int(round(x0 - vx * distance)),
            int(round(y0 - vy * distance)),
        )

        p2 = (
            int(round(x0 + vx * distance)),
            int(round(y0 + vy * distance)),
        )

        ok, clipped_p1, clipped_p2 = cv2.clipLine(
            (0, 0, w, h),
            p1,
            p2,
        )

        if ok:
            cv2.line(
                image,
                clipped_p1,
                clipped_p2,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    x0, y0 = points[0]

    cv2.putText(
        image,
        label,
        (
            int(np.clip(round(x0) + 6, 0, w - 1)),
            int(np.clip(round(y0) - 6, 0, h - 1)),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def load_annotations(
    path: Path | None,
    lens: str,
) -> tuple[dict[str, list], dict[str, list]]:

    if path is None:
        return {}, {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        document = json.load(source)

    annotations = document.get(
        "annotations",
        {},
    )

    lines = (
        annotations
        .get("lines", {})
        .get(lens, {})
    )

    landmarks = (
        annotations
        .get("landmarks", {})
        .get(lens, {})
    )

    return lines, landmarks


def draw_landmarks(
    image: np.ndarray,
    landmarks: dict[str, list],
    transform=None,
) -> None:

    h, w = image.shape[:2]

    for label, point in landmarks.items():
        pts = np.asarray(
            [point],
            dtype=np.float64,
        )

        if transform is not None:
            pts = transform(pts)

        x, y = pts[0]

        ix = int(round(x))
        iy = int(round(y))

        if not (
            0 <= ix < w
            and 0 <= iy < h
        ):
            continue

        cv2.drawMarker(
            image,
            (ix, iy),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            14,
            2,
        )

        cv2.putText(
            image,
            label,
            (ix + 5, iy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

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

    lenses = split_lenses(
        stacked
    )

    raw_image = lenses[
        args.lens
    ]

    raw_height, raw_width = raw_image.shape[:2]

    metadata = load_veo_metadata(
        args.calibration
    )

    lens_metadata = metadata.lens(
        args.lens
    )

    veo_k = lens_metadata.intrinsic
    distortion = lens_metadata.distortion

    k_raw, scale_x, scale_y = build_raw_intrinsics(
        veo_k,
        raw_width,
        raw_height,
        args.calibration_width,
        args.calibration_height,
    )

    output_width = (
        args.output_width
        if args.output_width is not None
        else raw_width
    )

    output_height = (
        args.output_height
        if args.output_height is not None
        else raw_height
    )

    k_rect = build_rectified_intrinsics(
        k_raw,
        output_width,
        output_height,
        args.focal_scale,
        args.center_x,
        args.center_y,
    )

    rectified, valid_mask = rectify_image(
        raw_image,
        k_raw,
        distortion,
        k_rect,
        (
            output_width,
            output_height,
        ),
    )

    lines, landmarks = load_annotations(
        args.annotations,
        args.lens,
    )

    raw_overlay = raw_image.copy()
    rect_overlay = rectified.copy()

    metrics = {}

    for label, raw_points_list in lines.items():
        raw_points = np.asarray(
            raw_points_list,
            dtype=np.float64,
        )

        if len(raw_points) == 0:
            continue

        rect_points = rectify_points(
            raw_points,
            k_raw,
            distortion,
            k_rect,
        )

        draw_line_fit(
            raw_overlay,
            raw_points,
            label,
        )

        draw_line_fit(
            rect_overlay,
            rect_points,
            label,
        )

        raw_metrics = fit_line_metrics(
            raw_points
        )

        rect_metrics = fit_line_metrics(
            rect_points
        )

        if (
            raw_metrics is not None
            and rect_metrics is not None
        ):
            ratio = (
                rect_metrics[
                    "normalized_rms"
                ]
                / raw_metrics[
                    "normalized_rms"
                ]
            )

            metrics[label] = {
                "raw": raw_metrics,
                "rectified": rect_metrics,
                "ratio": float(ratio),
            }

    draw_landmarks(
        raw_overlay,
        landmarks,
    )

    draw_landmarks(
        rect_overlay,
        landmarks,
        transform=lambda pts: rectify_points(
            pts,
            k_raw,
            distortion,
            k_rect,
        ),
    )

    prefix = args.lens

    cv2.imwrite(
        str(
            args.out
            / f"{prefix}_raw.jpg"
        ),
        raw_image,
    )

    cv2.imwrite(
        str(
            args.out
            / f"{prefix}_raw_overlay.jpg"
        ),
        raw_overlay,
    )

    cv2.imwrite(
        str(
            args.out
            / f"{prefix}_rectified.jpg"
        ),
        rectified,
    )

    cv2.imwrite(
        str(
            args.out
            / f"{prefix}_rectified_overlay.jpg"
        ),
        rect_overlay,
    )

    cv2.imwrite(
        str(
            args.out
            / f"{prefix}_valid_mask.png"
        ),
        valid_mask,
    )

    valid_fraction = float(
        np.count_nonzero(valid_mask)
        / valid_mask.size
    )

    ratios = [
        value["ratio"]
        for value in metrics.values()
    ]

    summary = {
        "lens": args.lens,
        "source": {
            "stacked": str(args.stacked),
            "calibration": str(args.calibration),
            "annotations": (
                str(args.annotations)
                if args.annotations
                else None
            ),
        },
        "raw_size": [
            raw_width,
            raw_height,
        ],
        "calibration_space": [
            args.calibration_width,
            args.calibration_height,
        ],
        "calibration_to_raw": {
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
        "veo_intrinsic": veo_k.tolist(),
        "effective_raw_intrinsic": k_raw.tolist(),
        "distortion": distortion.tolist(),
        "rectified_intrinsic": k_rect.tolist(),
        "rectified_output_size": [
            output_width,
            output_height,
        ],
        "focal_scale": args.focal_scale,
        "center_x": args.center_x,
        "center_y": args.center_y,
        "valid_fraction": valid_fraction,
        "line_metrics": metrics,
        "aggregate": {
            "line_count": len(ratios),
            "mean_ratio": (
                float(np.mean(ratios))
                if ratios
                else None
            ),
            "median_ratio": (
                float(np.median(ratios))
                if ratios
                else None
            ),
        },
    }

    summary_path = (
        args.out
        / f"{prefix}_profile.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        json.dump(
            summary,
            target,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"LENS: {args.lens.upper()}"
    )

    print(
        f"RAW:  {raw_width}x{raw_height}"
    )

    print(
        "Calibration-space assumption: "
        f"{args.calibration_width:g}x"
        f"{args.calibration_height:g}"
    )

    print(
        f"scale_x = {scale_x:.9f}"
    )

    print(
        f"scale_y = {scale_y:.9f}"
    )

    print()
    print("Effective RAW K:")
    print(k_raw)

    print()
    print("Rectified virtual-camera K:")
    print(k_rect)

    print()

    for label, result in metrics.items():
        verdict = (
            "BETTER"
            if result["ratio"] < 1.0
            else "WORSE"
        )

        print(
            f"{label:32s} "
            f"ratio={result['ratio']:.3f} "
            f"{verdict}"
        )

    if ratios:
        print()
        print(
            "Median straightness ratio: "
            f"{np.median(ratios):.3f}"
        )

    print()
    print(
        "Valid rectified image fraction: "
        f"{valid_fraction * 100:.1f}%"
    )

    print()
    print(
        f"Saved to: {args.out}"
    )


if __name__ == "__main__":
    main()
