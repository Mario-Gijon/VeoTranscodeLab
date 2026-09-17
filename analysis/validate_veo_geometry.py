#!/usr/bin/env python3
"""
Canonical validation for analysis/veo_geometry.py.

This script verifies that the centralized Veo geometry reproduces the
calibration behaviour already observed in the original diagnostics.

For every valid Veo selected corner it reports:

1) Inverse validation
   selected RAW pixel -> pixel_to_field() -> expected field corner
   Error unit: meters.

2) Forward validation
   expected field corner -> field_to_pixel() -> selected RAW pixel
   Error unit: RAW pixels.

3) Model round-trip validation
   expected field corner -> RAW pixel -> field
   Error unit: meters.

The round-trip metric checks internal mathematical consistency.
The selected-corner metrics check agreement with Veo's stored observations.

This script does not fit, optimize, correct or modify any calibration.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Make repository root importable when executed as:
#   python analysis/validate_veo_geometry.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.veo_geometry import VeoCalibration  # noqa: E402


@dataclass
class CornerResult:
    corner: str
    lens: str

    observed_raw: np.ndarray
    expected_field: np.ndarray

    recovered_field: np.ndarray
    predicted_raw: np.ndarray
    roundtrip_field: np.ndarray

    inverse_error_m: float
    forward_error_px: float
    roundtrip_error_m: float


def norm2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def validate_corner(
    calibration: VeoCalibration,
    observation: dict,
) -> CornerResult:
    corner = str(observation["corner"])
    lens = str(observation["lens"])

    observed_raw = np.asarray(
        observation["raw_pixel"],
        dtype=np.float64,
    ).reshape(2)

    expected_field = np.asarray(
        observation["expected_field"],
        dtype=np.float64,
    ).reshape(2)

    recovered_field = calibration.pixel_to_field(
        lens,
        observed_raw,
    )

    predicted_raw = calibration.field_to_pixel(
        lens,
        expected_field,
    )

    roundtrip_field = calibration.pixel_to_field(
        lens,
        predicted_raw,
    )

    return CornerResult(
        corner=corner,
        lens=lens,
        observed_raw=observed_raw,
        expected_field=expected_field,
        recovered_field=recovered_field,
        predicted_raw=predicted_raw,
        roundtrip_field=roundtrip_field,
        inverse_error_m=norm2(
            recovered_field,
            expected_field,
        ),
        forward_error_px=norm2(
            predicted_raw,
            observed_raw,
        ),
        roundtrip_error_m=norm2(
            roundtrip_field,
            expected_field,
        ),
    )


def fmt_pair(v: np.ndarray, decimals: int = 3) -> str:
    return (
        f"({v[0]:.{decimals}f}, "
        f"{v[1]:.{decimals}f})"
    )


def stats(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.max(arr)),
    )


def print_metric_summary(
    label: str,
    values: list[float],
    unit: str,
) -> None:
    mean, median, maximum = stats(values)
    print(
        f"{label}: "
        f"mean={mean:.6f} {unit}  "
        f"median={median:.6f} {unit}  "
        f"max={maximum:.6f} {unit}"
    )


def validate_file(path: Path) -> list[CornerResult]:
    calibration = VeoCalibration.from_file(path)
    observations = calibration.selected_corner_observations()

    if not observations:
        raise RuntimeError(
            f"No valid selected corners found in {path}"
        )

    results = [
        validate_corner(calibration, obs)
        for obs in observations
    ]

    print()
    print("=" * 78)
    print(path)
    print("=" * 78)
    print(
        f"field: {calibration.field_length:.6f} x "
        f"{calibration.field_width:.6f} m"
    )
    print(
        f"calibration_version={calibration.calibration_version}  "
        f"worldmap_version={calibration.worldmap_version}"
    )
    print()

    for result in results:
        print(
            f"{result.corner.upper():>2s} "
            f"{result.lens.upper():>5s}"
        )
        print(
            f"  Veo RAW:          "
            f"{fmt_pair(result.observed_raw)}"
        )
        print(
            f"  expected field:   "
            f"{fmt_pair(result.expected_field)} m"
        )
        print(
            f"  recovered field:  "
            f"{fmt_pair(result.recovered_field)} m"
        )
        print(
            f"  inverse error:    "
            f"{result.inverse_error_m:.6f} m"
        )
        print(
            f"  projected RAW:    "
            f"{fmt_pair(result.predicted_raw)}"
        )
        print(
            f"  forward error:    "
            f"{result.forward_error_px:.6f} px"
        )
        print(
            f"  roundtrip field:  "
            f"{fmt_pair(result.roundtrip_field)} m"
        )
        print(
            f"  roundtrip error:  "
            f"{result.roundtrip_error_m:.9f} m"
        )
        print()

    print("Summary")
    print("-------")

    print_metric_summary(
        "inverse selected-corner error",
        [r.inverse_error_m for r in results],
        "m",
    )

    print_metric_summary(
        "forward selected-corner error",
        [r.forward_error_px for r in results],
        "px",
    )

    print_metric_summary(
        "field -> pixel -> field roundtrip",
        [r.roundtrip_error_m for r in results],
        "m",
    )

    return results


def print_global_summary(
    all_results: list[CornerResult],
) -> None:
    if not all_results:
        return

    print()
    print("=" * 78)
    print("GLOBAL SUMMARY")
    print("=" * 78)

    print_metric_summary(
        "inverse selected-corner error",
        [r.inverse_error_m for r in all_results],
        "m",
    )

    print_metric_summary(
        "forward selected-corner error",
        [r.forward_error_px for r in all_results],
        "px",
    )

    print_metric_summary(
        "field -> pixel -> field roundtrip",
        [r.roundtrip_error_m for r in all_results],
        "m",
    )

    print()
    print(
        "Interpretation:"
    )
    print(
        "- Roundtrip error measures internal consistency of our mathematical model."
    )
    print(
        "- Forward/inverse selected-corner errors measure agreement with Veo's "
        "stored corner observations."
    )
    print(
        "- A low roundtrip error together with a larger selected-corner error "
        "means the implementation is self-consistent, but Veo's stored "
        "observation and the modeled projection do not coincide perfectly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate analysis/veo_geometry.py against selected corners "
            "stored in one or more Veo calibration files."
        )
    )
    parser.add_argument(
        "calibrations",
        nargs="+",
        type=Path,
        help="One or more .veo files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing = [
        str(path)
        for path in args.calibrations
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "Missing calibration file(s): "
            + ", ".join(missing)
        )

    all_results: list[CornerResult] = []

    for path in args.calibrations:
        all_results.extend(
            validate_file(path)
        )

    if len(args.calibrations) > 1:
        print_global_summary(all_results)


if __name__ == "__main__":
    main()
