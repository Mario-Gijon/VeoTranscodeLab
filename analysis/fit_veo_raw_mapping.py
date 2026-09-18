#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError as exc:
    raise SystemExit(
        "This script requires scipy. Install it in the project environment "
        "with: pip install scipy"
    ) from exc


INITIAL_SCALE = 2048.0 / 3840.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit the mapping from Veo calibration-space intrinsics to the "
            "downloaded RAW lens images while keeping Veo distortion fixed."
        )
    )

    parser.add_argument("--stacked", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/fit_veo_raw_mapping"),
    )

    parser.add_argument(
        "--model",
        choices=("uniform", "anisotropic", "both"),
        default="both",
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=1500,
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


def load_lines(path):
    with path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    return document["annotations"]["lines"]


def split_stacked_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    h, w = image.shape[:2]

    if h % 2 != 0:
        raise ValueError(
            f"Expected equal vertically stacked lenses, got image height {h}"
        )

    lens_h = h // 2

    return {
        "left": image[:lens_h].copy(),
        "right": image[lens_h:].copy(),
    }


def build_raw_k(k_veo, params, model):
    if model == "uniform":
        s, ox, oy = params

        sx = s
        sy = s

    elif model == "anisotropic":
        sx, sy, ox, oy = params

    else:
        raise ValueError(model)

    k = k_veo.copy()

    k[0, 0] = sx * k_veo[0, 0]
    k[1, 1] = sy * k_veo[1, 1]

    k[0, 2] = sx * k_veo[0, 2] + ox
    k[1, 2] = sy * k_veo[1, 2] + oy

    return k


def undistort_points(points, k, d):
    points = np.asarray(points, dtype=np.float64)

    corrected = cv2.undistortPoints(
        points.reshape(-1, 1, 2),
        k,
        d,
        P=k,
    )

    return corrected.reshape(-1, 2)


def fit_line(points):
    points = np.asarray(points, dtype=np.float64)

    center = np.mean(points, axis=0)

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

    longitudinal = centered @ direction
    orthogonal = centered @ normal

    span = float(np.ptp(longitudinal))

    return {
        "center": center,
        "direction": direction,
        "normal": normal,
        "orthogonal": orthogonal,
        "span": span,
    }


def normalized_line_residuals(points):
    line = fit_line(points)

    span = max(line["span"], 1.0)

    return line["orthogonal"] / span


def line_metrics(points):
    points = np.asarray(points, dtype=np.float64)

    if len(points) < 3:
        return None

    line = fit_line(points)

    residuals = line["orthogonal"]
    span = max(line["span"], 1.0)

    rms_px = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    normalized_rms = rms_px / span

    return {
        "rms_px": rms_px,
        "normalized_rms": normalized_rms,
        "span_px": float(line["span"]),
        "max_abs_px": float(
            np.max(
                np.abs(residuals)
            )
        ),
    }


def collect_valid_lines(line_annotations):
    valid = []

    for lens in ("left", "right"):
        for label, points in line_annotations.get(lens, {}).items():
            if len(points) < 3:
                continue

            valid.append(
                {
                    "lens": lens,
                    "label": label,
                    "points": np.asarray(
                        points,
                        dtype=np.float64,
                    ),
                }
            )

    if len(valid) < 2:
        raise ValueError(
            "At least two annotated lines with >= 3 points are required."
        )

    return valid


def make_objective(lines, optics, model):
    def objective(params):
        residual_blocks = []

        for item in lines:
            lens = item["lens"]
            points = item["points"]

            k = build_raw_k(
                optics[lens]["K"],
                params,
                model,
            )

            d = optics[lens]["D"]

            corrected = undistort_points(
                points,
                k,
                d,
            )

            residuals = normalized_line_residuals(
                corrected
            )

            # Equalize influence between long and short point groups.
            residuals = (
                residuals
                / np.sqrt(len(residuals))
            )

            residual_blocks.append(
                residuals
            )

        return np.concatenate(
            residual_blocks
        )

    return objective


def optimize_model(lines, optics, model, max_nfev):
    if model == "uniform":
        x0 = np.array(
            [
                INITIAL_SCALE,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        lower = np.array(
            [
                0.45,
                -200.0,
                -200.0,
            ]
        )

        upper = np.array(
            [
                0.65,
                200.0,
                200.0,
            ]
        )

    elif model == "anisotropic":
        x0 = np.array(
            [
                INITIAL_SCALE,
                INITIAL_SCALE,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        lower = np.array(
            [
                0.45,
                0.45,
                -200.0,
                -200.0,
            ]
        )

        upper = np.array(
            [
                0.65,
                0.65,
                200.0,
                200.0,
            ]
        )

    else:
        raise ValueError(model)

    objective = make_objective(
        lines,
        optics,
        model,
    )

    result = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.002,
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        verbose=0,
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
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def draw_line_overlay(image, grouped_points):
    result = image.copy()

    h, w = result.shape[:2]

    for label, points in grouped_points.items():
        points = np.asarray(
            points,
            dtype=np.float64,
        )

        if len(points) == 0:
            continue

        for u, v in points:
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

        if len(points) >= 2:
            line = fit_line(points)

            center = line["center"]
            direction = line["direction"]

            p1 = center - direction * 5000
            p2 = center + direction * 5000

            p1 = tuple(
                np.round(p1).astype(int)
            )

            p2 = tuple(
                np.round(p2).astype(int)
            )

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

        first = points[0]

        cv2.putText(
            result,
            label,
            (
                int(
                    np.clip(
                        round(first[0]) + 5,
                        0,
                        w - 1,
                    )
                ),
                int(
                    np.clip(
                        round(first[1]) - 5,
                        0,
                        h - 1,
                    )
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


def evaluate_solution(
    model,
    params,
    lines,
    optics,
):
    rows = []

    ratios = []

    for item in lines:
        lens = item["lens"]
        label = item["label"]
        raw_points = item["points"]

        k = build_raw_k(
            optics[lens]["K"],
            params,
            model,
        )

        corrected = undistort_points(
            raw_points,
            k,
            optics[lens]["D"],
        )

        raw_metrics = line_metrics(
            raw_points
        )

        corrected_metrics = line_metrics(
            corrected
        )

        ratio = (
            corrected_metrics["normalized_rms"]
            / raw_metrics["normalized_rms"]
        )

        ratios.append(ratio)

        rows.append(
            {
                "lens": lens,
                "line": label,
                "points": len(raw_points),
                "raw_rms_px": raw_metrics["rms_px"],
                "corrected_rms_px": corrected_metrics["rms_px"],
                "raw_normalized_rms": raw_metrics[
                    "normalized_rms"
                ],
                "corrected_normalized_rms": corrected_metrics[
                    "normalized_rms"
                ],
                "ratio": ratio,
            }
        )

    ratios = np.asarray(
        ratios,
        dtype=np.float64,
    )

    aggregate = {
        "mean_ratio": float(
            np.mean(ratios)
        ),
        "median_ratio": float(
            np.median(ratios)
        ),
        "worst_ratio": float(
            np.max(ratios)
        ),
        "best_ratio": float(
            np.min(ratios)
        ),
        "lines": int(
            len(ratios)
        ),
    }

    return rows, aggregate


def save_visuals(
    model_dir,
    model,
    params,
    images,
    lines,
    optics,
):
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped = {
        "left": {},
        "right": {},
    }

    for item in lines:
        lens = item["lens"]
        label = item["label"]

        k = build_raw_k(
            optics[lens]["K"],
            params,
            model,
        )

        corrected = undistort_points(
            item["points"],
            k,
            optics[lens]["D"],
        )

        grouped[lens][label] = corrected.tolist()

    for lens in ("left", "right"):
        k = build_raw_k(
            optics[lens]["K"],
            params,
            model,
        )

        d = optics[lens]["D"]

        rectified = undistort_image(
            images[lens],
            k,
            d,
        )

        overlay = draw_line_overlay(
            rectified,
            grouped[lens],
        )

        cv2.imwrite(
            str(
                model_dir
                / f"{lens}_rectified.jpg"
            ),
            rectified,
        )

        cv2.imwrite(
            str(
                model_dir
                / f"{lens}_rectified_lines.jpg"
            ),
            overlay,
        )


def params_to_dict(model, params):
    if model == "uniform":
        s, ox, oy = params

        return {
            "scale_x": float(s),
            "scale_y": float(s),
            "offset_x": float(ox),
            "offset_y": float(oy),
        }

    sx, sy, ox, oy = params

    return {
        "scale_x": float(sx),
        "scale_y": float(sy),
        "offset_x": float(ox),
        "offset_y": float(oy),
    }


def main():
    args = parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    optics = load_veo_optics(
        args.calibration
    )

    annotations = load_lines(
        args.annotations
    )

    lines = collect_valid_lines(
        annotations
    )

    images = split_stacked_image(
        args.stacked
    )

    if args.model == "both":
        models = [
            "uniform",
            "anisotropic",
        ]
    else:
        models = [
            args.model,
        ]

    output = {
        "initial_scale": INITIAL_SCALE,
        "models": {},
    }

    all_csv_rows = []

    print()
    print(
        f"Valid annotated lines: {len(lines)}"
    )

    for item in lines:
        print(
            f"  {item['lens']:5s} "
            f"{item['label']:32s} "
            f"{len(item['points'])} points"
        )

    for model in models:
        print()
        print("=" * 72)
        print(
            f"MODEL: {model}"
        )

        result = optimize_model(
            lines,
            optics,
            model,
            args.max_nfev,
        )

        params = result.x

        params_dict = params_to_dict(
            model,
            params,
        )

        rows, aggregate = evaluate_solution(
            model,
            params,
            lines,
            optics,
        )

        print()
        print("Fitted mapping:")

        for key, value in params_dict.items():
            print(
                f"  {key:10s} = {value:.9f}"
            )

        print()
        print(
            f"Optimizer success: {result.success}"
        )

        print(
            f"Cost:              {result.cost:.12f}"
        )

        print(
            f"Evaluations:       {result.nfev}"
        )

        print()
        print("Per-line results:")

        for row in rows:
            verdict = (
                "BETTER"
                if row["ratio"] < 1.0
                else "WORSE"
            )

            print(
                f"  {row['lens']:5s} "
                f"{row['line']:32s} "
                f"raw={row['raw_normalized_rms']:.6f} "
                f"corr={row['corrected_normalized_rms']:.6f} "
                f"ratio={row['ratio']:.3f} "
                f"{verdict}"
            )

            all_csv_rows.append(
                {
                    "model": model,
                    **row,
                }
            )

        print()
        print(
            "Aggregate:"
        )

        print(
            f"  median ratio = "
            f"{aggregate['median_ratio']:.3f}"
        )

        print(
            f"  mean ratio   = "
            f"{aggregate['mean_ratio']:.3f}"
        )

        print(
            f"  worst ratio  = "
            f"{aggregate['worst_ratio']:.3f}"
        )

        output["models"][model] = {
            "parameters": params_dict,
            "optimizer": {
                "success": bool(
                    result.success
                ),
                "message": str(
                    result.message
                ),
                "cost": float(
                    result.cost
                ),
                "nfev": int(
                    result.nfev
                ),
            },
            "aggregate": aggregate,
            "lines": rows,
        }

        save_visuals(
            args.out / model,
            model,
            params,
            images,
            lines,
            optics,
        )

    results_path = (
        args.out
        / "results.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if all_csv_rows:
        csv_path = (
            args.out
            / "comparison.csv"
        )

        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    all_csv_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                all_csv_rows
            )

    print()
    print("=" * 72)
    print("DONE")
    print(
        f"Results: {results_path}"
    )

    if "uniform" in output["models"]:
        p = output["models"]["uniform"]["parameters"]

        print()
        print(
            "Uniform model interpretation:"
        )

        print(
            f"  Veo calibration -> RAW scale: "
            f"{p['scale_x']:.9f}"
        )

        print(
            f"  X offset: {p['offset_x']:.3f} px"
        )

        print(
            f"  Y offset: {p['offset_y']:.3f} px"
        )

        print()
        print(
            "Reference scale for 3840x1920 -> "
            f"2048x1024 is {INITIAL_SCALE:.9f}"
        )


if __name__ == "__main__":
    main()
