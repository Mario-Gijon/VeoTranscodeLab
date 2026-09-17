#!/usr/bin/env python3
"""
Analyze the final calibration-pixel -> RAW-transcode pixel mapping used by Veo.

This is a diagnostic experiment for VeoTranscodeLab. It does not modify any
calibration or canonical geometry.

Why this exists
---------------
The current canonical model assumes that OpenCV projection happens in an
effective Veo calibration image of:

    3840 x 1536

and that the downloaded RAW lens frame is:

    2048 x 1024

with a pure anisotropic resize and no crop/offset:

    raw_u = (2048 / 3840) * calibration_u
    raw_v = (1024 / 1536) * calibration_v

This script tests that assumption empirically using the selected corners stored
inside multiple `.veo` files.

For every valid selected corner it obtains:
- expected field coordinate from field_length / field_width;
- projected RAW pixel from the canonical geometry;
- the corresponding pre-resize calibration pixel;
- observed RAW corner stored by Veo.

Then it fits simple mapping models:

    raw = scale * calibration
    raw = scale * calibration + offset

globally and per lens.

If fitted scale/offset stay extremely close to the current constants and the
same individual corners remain outliers, the RAW/calibration resize mapping is
unlikely to be the missing source of geometric error.

Typical usage
-------------
    python analysis/analyze_veo_pixel_mapping.py \
        data/camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo \
        data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
        --json-out output/veo_pixel_mapping.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.veo_geometry import (  # noqa: E402
    CALIBRATION_HEIGHT,
    CALIBRATION_WIDTH,
    RAW_HEIGHT,
    RAW_WIDTH,
    SCALE_X,
    SCALE_Y,
    VeoCalibration,
    load_veo_calibration,
)


@dataclass(frozen=True)
class MappingObservation:
    source: str
    corner: str
    lens: str
    calibration_pixel: list[float]
    observed_raw: list[float]
    canonical_predicted_raw: list[float]
    canonical_residual_raw: list[float]
    canonical_error_px: float


@dataclass(frozen=True)
class AxisFit:
    model: str
    count: int
    scale: float
    offset: float
    rmse_px: float
    median_abs_px: float
    max_abs_px: float


@dataclass(frozen=True)
class MappingFit:
    name: str
    u: AxisFit
    v: AxisFit
    vector_rmse_px: float
    vector_median_px: float
    vector_max_px: float


def gather_observations(
    paths: Iterable[Path],
) -> list[MappingObservation]:
    rows: list[MappingObservation] = []

    for path in paths:
        calibration = load_veo_calibration(path)

        for obs in calibration.selected_corner_observations():
            lens = str(obs["lens"])
            field = np.asarray(
                obs["expected_field"],
                dtype=np.float64,
            ).reshape(2)
            observed_raw = np.asarray(
                obs["raw_pixel"],
                dtype=np.float64,
            ).reshape(2)

            predicted_raw = calibration.field_to_pixel(
                lens,
                field,
            )

            # Undo only the currently assumed final resize. Everything before
            # this point is the canonical camera/lens/distortion projection.
            calibration_pixel = np.asarray(
                [
                    predicted_raw[0] / SCALE_X,
                    predicted_raw[1] / SCALE_Y,
                ],
                dtype=np.float64,
            )

            residual = predicted_raw - observed_raw

            rows.append(
                MappingObservation(
                    source=str(path),
                    corner=str(obs["corner"]),
                    lens=lens,
                    calibration_pixel=calibration_pixel.tolist(),
                    observed_raw=observed_raw.tolist(),
                    canonical_predicted_raw=predicted_raw.tolist(),
                    canonical_residual_raw=residual.tolist(),
                    canonical_error_px=float(
                        np.linalg.norm(residual)
                    ),
                )
            )

    if not rows:
        raise RuntimeError(
            "No valid selected-corner observations were found."
        )

    return rows


def fit_axis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    with_offset: bool,
) -> AxisFit:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError("x/y size mismatch")

    if with_offset:
        design = np.column_stack(
            [
                x,
                np.ones_like(x),
            ]
        )
        parameters, *_ = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )
        scale = float(parameters[0])
        offset = float(parameters[1])
        predicted = scale * x + offset
        model = "scale+offset"
    else:
        denominator = float(np.dot(x, x))
        if denominator <= 1e-18:
            raise RuntimeError(
                "Cannot fit scale-only model: degenerate input"
            )
        scale = float(
            np.dot(x, y) / denominator
        )
        offset = 0.0
        predicted = scale * x
        model = "scale-only"

    residual = predicted - y
    abs_residual = np.abs(residual)

    return AxisFit(
        model=model,
        count=int(x.size),
        scale=scale,
        offset=offset,
        rmse_px=float(
            np.sqrt(np.mean(residual * residual))
        ),
        median_abs_px=float(
            np.median(abs_residual)
        ),
        max_abs_px=float(
            np.max(abs_residual)
        ),
    )


def evaluate_mapping(
    rows: list[MappingObservation],
    name: str,
    *,
    with_offset: bool,
) -> MappingFit:
    calibration_pixels = np.asarray(
        [row.calibration_pixel for row in rows],
        dtype=np.float64,
    )
    observed_raw = np.asarray(
        [row.observed_raw for row in rows],
        dtype=np.float64,
    )

    fit_u = fit_axis(
        calibration_pixels[:, 0],
        observed_raw[:, 0],
        with_offset=with_offset,
    )
    fit_v = fit_axis(
        calibration_pixels[:, 1],
        observed_raw[:, 1],
        with_offset=with_offset,
    )

    predicted = np.column_stack(
        [
            (
                fit_u.scale * calibration_pixels[:, 0]
                + fit_u.offset
            ),
            (
                fit_v.scale * calibration_pixels[:, 1]
                + fit_v.offset
            ),
        ]
    )

    errors = np.linalg.norm(
        predicted - observed_raw,
        axis=1,
    )

    return MappingFit(
        name=name,
        u=fit_u,
        v=fit_v,
        vector_rmse_px=float(
            np.sqrt(np.mean(errors * errors))
        ),
        vector_median_px=float(
            np.median(errors)
        ),
        vector_max_px=float(
            np.max(errors)
        ),
    )


def canonical_fit(
    rows: list[MappingObservation],
) -> MappingFit:
    calibration_pixels = np.asarray(
        [row.calibration_pixel for row in rows],
        dtype=np.float64,
    )
    observed_raw = np.asarray(
        [row.observed_raw for row in rows],
        dtype=np.float64,
    )

    predicted = np.column_stack(
        [
            SCALE_X * calibration_pixels[:, 0],
            SCALE_Y * calibration_pixels[:, 1],
        ]
    )
    residual = predicted - observed_raw
    errors = np.linalg.norm(
        residual,
        axis=1,
    )

    def axis_info(
        axis: int,
        scale: float,
    ) -> AxisFit:
        r = residual[:, axis]
        a = np.abs(r)
        return AxisFit(
            model="canonical",
            count=len(rows),
            scale=float(scale),
            offset=0.0,
            rmse_px=float(
                np.sqrt(np.mean(r * r))
            ),
            median_abs_px=float(
                np.median(a)
            ),
            max_abs_px=float(
                np.max(a)
            ),
        )

    return MappingFit(
        name="canonical",
        u=axis_info(0, SCALE_X),
        v=axis_info(1, SCALE_Y),
        vector_rmse_px=float(
            np.sqrt(np.mean(errors * errors))
        ),
        vector_median_px=float(
            np.median(errors)
        ),
        vector_max_px=float(
            np.max(errors)
        ),
    )


def print_fit(fit: MappingFit) -> None:
    print(fit.name)
    print("-" * len(fit.name))
    print(
        "U: "
        f"scale={fit.u.scale:.12f} "
        f"offset={fit.u.offset:+.6f}px | "
        f"RMSE={fit.u.rmse_px:.6f}px "
        f"median={fit.u.median_abs_px:.6f}px "
        f"max={fit.u.max_abs_px:.6f}px"
    )
    print(
        "V: "
        f"scale={fit.v.scale:.12f} "
        f"offset={fit.v.offset:+.6f}px | "
        f"RMSE={fit.v.rmse_px:.6f}px "
        f"median={fit.v.median_abs_px:.6f}px "
        f"max={fit.v.max_abs_px:.6f}px"
    )
    print(
        "2D: "
        f"RMSE={fit.vector_rmse_px:.6f}px "
        f"median={fit.vector_median_px:.6f}px "
        f"max={fit.vector_max_px:.6f}px"
    )

    if fit.u.scale > 0 and fit.v.scale > 0:
        implied_width = RAW_WIDTH / fit.u.scale
        implied_height = RAW_HEIGHT / fit.v.scale
        print(
            "Implied zero-offset calibration size from scales: "
            f"{implied_width:.3f} x {implied_height:.3f}"
        )

    print()


def print_observations(
    rows: list[MappingObservation],
) -> None:
    print("Canonical per-corner residuals")
    print("------------------------------")
    for row in rows:
        r = row.canonical_residual_raw
        print(
            f"{Path(row.source).name:64s} "
            f"{row.corner.upper():>2s} {row.lens.upper():>5s} | "
            f"du={r[0]:+8.3f}px "
            f"dv={r[1]:+8.3f}px "
            f"| error={row.canonical_error_px:8.3f}px"
        )
    print()


def subset(
    rows: list[MappingObservation],
    *,
    lens: str | None = None,
) -> list[MappingObservation]:
    return [
        row
        for row in rows
        if lens is None or row.lens == lens
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the assumed Veo calibration-pixel -> RAW-transcode "
            "resize/crop mapping using selected corners across .veo files."
        )
    )
    parser.add_argument(
        "calibrations",
        nargs="+",
        type=Path,
        help="Two or more .veo files are recommended.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional machine-readable JSON report.",
    )
    args = parser.parse_args()

    missing = [
        str(path)
        for path in args.calibrations
        if not path.is_file()
    ]
    if missing:
        parser.error(
            "Missing calibration file(s): "
            + ", ".join(missing)
        )

    return args


def main() -> None:
    args = parse_args()

    rows = gather_observations(
        args.calibrations
    )

    print("Veo calibration-pixel -> RAW mapping audit")
    print("==========================================")
    print(
        f"Canonical mapping: "
        f"sx={SCALE_X:.12f} "
        f"sy={SCALE_Y:.12f} "
        f"offset=(0,0)"
    )
    print(
        f"Assumed calibration size: "
        f"{CALIBRATION_WIDTH} x {CALIBRATION_HEIGHT}"
    )
    print(
        f"RAW lens size: "
        f"{RAW_WIDTH} x {RAW_HEIGHT}"
    )
    print(
        f"Observations: {len(rows)} "
        f"from {len(args.calibrations)} file(s)"
    )
    print()

    print_observations(rows)

    fits: list[MappingFit] = []

    all_canonical = canonical_fit(rows)
    fits.append(all_canonical)
    print_fit(all_canonical)

    for with_offset in (False, True):
        fit = evaluate_mapping(
            rows,
            (
                "global fitted scale-only"
                if not with_offset
                else "global fitted scale+offset"
            ),
            with_offset=with_offset,
        )
        fits.append(fit)
        print_fit(fit)

    for lens in ("left", "right"):
        lens_rows = subset(
            rows,
            lens=lens,
        )
        if len(lens_rows) < 2:
            continue

        for with_offset in (False, True):
            fit = evaluate_mapping(
                lens_rows,
                (
                    f"{lens} fitted scale-only"
                    if not with_offset
                    else f"{lens} fitted scale+offset"
                ),
                with_offset=with_offset,
            )
            fits.append(fit)
            print_fit(fit)

    print("Interpretation guide")
    print("--------------------")
    print(
        "- If fitted global scale/offset remains close to the canonical "
        "mapping and residuals remain concentrated in individual corners, "
        "the 3840x1536 -> 2048x1024 mapping is probably not the main error."
    )
    print(
        "- A consistent non-zero offset or substantially different scale "
        "across both lenses/matches would suggest a crop/resize transform "
        "that should be modeled explicitly."
    )
    print(
        "- Per-lens fits are diagnostic only; with few selected corners they "
        "can overfit and should not be promoted into canonical geometry "
        "without additional recordings."
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        payload = {
            "canonical": {
                "calibration_size": [
                    CALIBRATION_WIDTH,
                    CALIBRATION_HEIGHT,
                ],
                "raw_size": [
                    RAW_WIDTH,
                    RAW_HEIGHT,
                ],
                "scale_x": SCALE_X,
                "scale_y": SCALE_Y,
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
            "observations": [
                asdict(row)
                for row in rows
            ],
            "fits": [
                asdict(fit)
                for fit in fits
            ],
        }
        args.json_out.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print()
        print(
            f"JSON report written to: "
            f"{args.json_out}"
        )


if __name__ == "__main__":
    main()
