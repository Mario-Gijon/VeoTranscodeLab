#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


HYPOTHESES = {
    # Our previous hypothesis.
    "3840x1536": (3840.0, 1536.0),

    # Uniform scaling candidate.
    "3840x1920": (3840.0, 1920.0),

    # Another previously interesting effective space.
    "3072x1536": (3072.0, 1536.0),

    # Control: use Veo K directly without scaling.
    "original_k": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether Veo OpenCV intrinsics/distortion straighten "
            "manually annotated physical pitch lines."
        )
    )
    parser.add_argument("--stacked", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/verify_veo_distortion"),
    )
    return parser.parse_args()


def parse_numeric_array(value):
    if isinstance(value, str):
        return np.fromstring(value, sep=",", dtype=np.float64)

    return np.asarray(value, dtype=np.float64).reshape(-1)


def load_veo_optics(path):
    with path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    alignment = document["alignment"]

    k_left = parse_numeric_array(
        alignment["intrinsic_left"]
    ).reshape(3, 3)

    k_right = parse_numeric_array(
        alignment["intrinsic_right"]
    ).reshape(3, 3)

    d_left = parse_numeric_array(
        alignment["distortion_left"]
    )

    d_right = parse_numeric_array(
        alignment["distortion_right"]
    )

    if d_left.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"Unsupported LEFT distortion length: {d_left.size}"
        )

    if d_right.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"Unsupported RIGHT distortion length: {d_right.size}"
        )

    return {
        "left": (k_left, d_left),
        "right": (k_right, d_right),
    }


def scale_intrinsics(k, raw_width, raw_height, calibration_size):
    if calibration_size is None:
        return k.copy()

    calibration_width, calibration_height = calibration_size

    sx = raw_width / calibration_width
    sy = raw_height / calibration_height

    scaled = k.copy()

    scaled[0, 0] *= sx  # fx
    scaled[0, 2] *= sx  # cx

    scaled[1, 1] *= sy  # fy
    scaled[1, 2] *= sy  # cy

    return scaled


def undistort_points(points, k, d):
    points = np.asarray(points, dtype=np.float64)

    result = cv2.undistortPoints(
        points.reshape(-1, 1, 2),
        k,
        d,
        P=k,
    )

    return result.reshape(-1, 2)


def line_metrics(points):
    """
    Orthogonal best-fit line using PCA/SVD.

    Returns:
      rms_px
      max_px
      span_px
      normalized_rms = rms / span
    """

    points = np.asarray(points, dtype=np.float64)

    if len(points) < 3:
        return None

    center = points.mean(axis=0)
    centered = points - center

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    direction = vh[0]
    normal = np.array(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    residuals = centered @ normal
    longitudinal = centered @ direction

    rms = float(np.sqrt(np.mean(residuals ** 2)))
    maximum = float(np.max(np.abs(residuals)))
    span = float(np.ptp(longitudinal))

    normalized = rms / span if span > 1e-9 else float("inf")

    return {
        "rms_px": rms,
        "max_px": maximum,
        "span_px": span,
        "normalized_rms": normalized,
    }


def fit_line_for_drawing(points):
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    scale = 10000

    p1 = (
        int(round(x0 - vx * scale)),
        int(round(y0 - vy * scale)),
    )

    p2 = (
        int(round(x0 + vx * scale)),
        int(round(y0 + vy * scale)),
    )

    return p1, p2


def draw_annotations(image, grouped_points):
    result = image.copy()
    h, w = result.shape[:2]

    for label, points in grouped_points.items():
        if len(points) < 1:
            continue

        points_np = np.asarray(points, dtype=np.float64)

        for u, v in points_np:
            x = int(round(u))
            y = int(round(v))

            if 0 <= x < w and 0 <= y < h:
                cv2.circle(
                    result,
                    (x, y),
                    4,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

        if len(points_np) >= 2:
            fitted = fit_line_for_drawing(points_np)

            if fitted is not None:
                p1, p2 = fitted

                ok, cp1, cp2 = cv2.clipLine(
                    (0, 0, w, h),
                    p1,
                    p2,
                )

                if ok:
                    cv2.line(
                        result,
                        cp1,
                        cp2,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        first = points_np[0]

        x = int(np.clip(round(first[0]), 0, w - 1))
        y = int(np.clip(round(first[1]), 0, h - 1))

        cv2.putText(
            result,
            label,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


def undistort_image(image, k, d):
    h, w = image.shape[:2]

    map_x, map_y = cv2.initUndistortRectifyMap(
        k,
        d,
        None,
        k,
        (w, h),
        cv2.CV_32FC1,
    )

    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.stacked))

    if image is None:
        raise RuntimeError(f"Could not read image: {args.stacked}")

    stacked_height, stacked_width = image.shape[:2]

    if stacked_height % 2 != 0:
        raise ValueError(
            f"Expected vertically stacked equal-height lenses; "
            f"image height is {stacked_height}"
        )

    lens_height = stacked_height // 2
    lens_width = stacked_width

    raw_images = {
        "left": image[:lens_height].copy(),
        "right": image[lens_height:].copy(),
    }

    with args.annotations.open("r", encoding="utf-8") as f:
        annotations_document = json.load(f)

    line_annotations = annotations_document[
        "annotations"
    ]["lines"]

    optics = load_veo_optics(args.calibration)

    all_rows = []
    summary = {}

    print()
    print("RAW lens size:")
    print(f"  {lens_width} x {lens_height}")
    print()

    for hypothesis_name, calibration_size in HYPOTHESES.items():
        print("=" * 72)
        print(f"HYPOTHESIS: {hypothesis_name}")

        summary[hypothesis_name] = {}

        normalized_ratios = []

        hypothesis_dir = args.out / hypothesis_name
        hypothesis_dir.mkdir(parents=True, exist_ok=True)

        for lens in ("left", "right"):
            raw_image = raw_images[lens]

            k_original, d = optics[lens]

            k = scale_intrinsics(
                k_original,
                lens_width,
                lens_height,
                calibration_size,
            )

            print()
            print(f"{lens.upper()} K:")
            print(k)

            rectified_image = undistort_image(
                raw_image,
                k,
                d,
            )

            transformed_groups = {}

            lens_summary = {}

            for label, points in line_annotations.get(lens, {}).items():
                if len(points) < 3:
                    print(
                        f"  SKIP {label}: only {len(points)} point(s)"
                    )
                    continue

                raw_points = np.asarray(
                    points,
                    dtype=np.float64,
                )

                rectified_points = undistort_points(
                    raw_points,
                    k,
                    d,
                )

                transformed_groups[label] = (
                    rectified_points.tolist()
                )

                raw_metrics = line_metrics(raw_points)
                rect_metrics = line_metrics(rectified_points)

                ratio = (
                    rect_metrics["normalized_rms"]
                    / raw_metrics["normalized_rms"]
                )

                normalized_ratios.append(ratio)

                lens_summary[label] = {
                    "raw": raw_metrics,
                    "undistorted": rect_metrics,
                    "normalized_error_ratio": ratio,
                }

                all_rows.append({
                    "hypothesis": hypothesis_name,
                    "lens": lens,
                    "line": label,
                    "points": len(points),
                    "raw_rms_px": raw_metrics["rms_px"],
                    "undistorted_rms_px": rect_metrics["rms_px"],
                    "raw_normalized_rms": raw_metrics[
                        "normalized_rms"
                    ],
                    "undistorted_normalized_rms": rect_metrics[
                        "normalized_rms"
                    ],
                    "normalized_error_ratio": ratio,
                })

                verdict = "BETTER" if ratio < 1.0 else "WORSE"

                print(
                    f"  {label:32s} "
                    f"raw={raw_metrics['normalized_rms']:.6f} "
                    f"und={rect_metrics['normalized_rms']:.6f} "
                    f"ratio={ratio:.3f} "
                    f"{verdict}"
                )

            summary[hypothesis_name][lens] = lens_summary

            raw_overlay = draw_annotations(
                raw_image,
                line_annotations.get(lens, {}),
            )

            rect_overlay = draw_annotations(
                rectified_image,
                transformed_groups,
            )

            cv2.imwrite(
                str(hypothesis_dir / f"{lens}_raw_lines.jpg"),
                raw_overlay,
            )

            cv2.imwrite(
                str(
                    hypothesis_dir
                    / f"{lens}_undistorted_lines.jpg"
                ),
                rect_overlay,
            )

            cv2.imwrite(
                str(
                    hypothesis_dir
                    / f"{lens}_undistorted.jpg"
                ),
                rectified_image,
            )

        if normalized_ratios:
            ratios = np.asarray(
                normalized_ratios,
                dtype=np.float64,
            )

            aggregate = {
                "mean_ratio": float(np.mean(ratios)),
                "median_ratio": float(np.median(ratios)),
                "best_ratio": float(np.min(ratios)),
                "worst_ratio": float(np.max(ratios)),
                "lines_evaluated": int(len(ratios)),
            }

            summary[hypothesis_name]["aggregate"] = aggregate

            print()
            print(
                "AGGREGATE: "
                f"median ratio={aggregate['median_ratio']:.3f}, "
                f"mean ratio={aggregate['mean_ratio']:.3f}"
            )

    json_path = args.out / "metrics.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = args.out / "metrics.csv"

    if all_rows:
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(all_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(all_rows)

    print()
    print("=" * 72)
    print("DONE")
    print(f"Results: {args.out}")
    print(f"JSON:    {json_path}")
    print(f"CSV:     {csv_path}")

    ranked = []

    for name, data in summary.items():
        aggregate = data.get("aggregate")

        if aggregate:
            ranked.append(
                (
                    aggregate["median_ratio"],
                    name,
                )
            )

    ranked.sort()

    if ranked:
        print()
        print("Hypotheses ordered by median normalized line error:")

        for ratio, name in ranked:
            print(
                f"  {name:16s} ratio={ratio:.3f}"
            )

        print()
        print(
            "Interpretation: ratio < 1 means the Veo correction "
            "made the annotated physical lines straighter."
        )


if __name__ == "__main__":
    main()
