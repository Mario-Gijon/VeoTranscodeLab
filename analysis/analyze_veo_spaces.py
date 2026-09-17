#!/usr/bin/env python3
"""
Analyze the coordinate spaces stored in Veo `.veo` calibration files.

This script is diagnostic only. It does not modify calibration data.

It studies the relationship between:

    camera2world
    camera2logic
    camera2presentation

and checks two concrete hypotheses suggested by the currently observed Veo
files:

1) `camera2logic` is a leveled camera frame derived from `camera2world`
   orientation:
   - it preserves the world-up direction as seen from the camera;
   - it removes the field/world heading (yaw);
   - it has no meaningful translation.

2) `camera2presentation` is derived from `camera2logic` by an additional
   rotation almost entirely around the logic X axis, i.e. a display/view tilt.

If those relationships hold with tiny numerical residuals, then LOGIC and
PRESENTATION are very likely derived coordinate spaces rather than independent
sources of pitch geometry.

Typical use:

    python analysis/analyze_veo_spaces.py \
        data/camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo \
        data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo

Optional JSON output:

    python analysis/analyze_veo_spaces.py data/*.veo \
        --json-out output/veo_spaces_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class RotationCheck:
    angle_deg: float
    max_abs_matrix_delta: float


@dataclass(frozen=True)
class SpaceAnalysis:
    source: str

    camera2world_translation: list[float]
    camera2logic_translation: list[float]
    camera2presentation_translation: list[float]

    camera2world_det: float
    camera2logic_det: float
    camera2presentation_det: float

    camera2world_orthogonality_error: float
    camera2logic_orthogonality_error: float
    camera2presentation_orthogonality_error: float

    logic_reconstruction_error_deg: float
    logic_reconstruction_max_abs: float

    world_from_logic_yaw_deg: float
    world_from_logic_yaw_residual_deg: float
    world_from_logic_yaw_max_abs: float

    presentation_from_logic_x_tilt_deg: float
    presentation_from_logic_x_residual_deg: float
    presentation_from_logic_x_max_abs: float

    field_length: float
    field_width: float

    va_camera: dict[str, float] | None
    va_logic: dict[str, float] | None
    va_presentation: dict[str, float] | None


def parse_matrix(value: str, rows: int = 4, cols: int = 4) -> np.ndarray:
    values = np.asarray(
        [float(x.strip()) for x in value.split(",")],
        dtype=np.float64,
    )
    expected = rows * cols
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} numeric values for {rows}x{cols}, "
            f"got {values.size}"
        )
    return values.reshape(rows, cols)


def normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / n


def rotation_angle_deg(rotation: np.ndarray) -> float:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def compare_rotations(a: np.ndarray, b: np.ndarray) -> RotationCheck:
    ra = np.asarray(a, dtype=np.float64).reshape(3, 3)
    rb = np.asarray(b, dtype=np.float64).reshape(3, 3)

    relative = ra @ rb.T
    return RotationCheck(
        angle_deg=rotation_angle_deg(relative),
        max_abs_matrix_delta=float(np.max(np.abs(ra - rb))),
    )


def orthogonality_error(rotation: np.ndarray) -> float:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return float(np.max(np.abs(r @ r.T - np.eye(3))))


def rotation_y_degrees(angle_deg: float) -> np.ndarray:
    """
    Y-axis rotation matching the convention observed in:
        R_world ~= R_y(yaw) @ R_logic

    Matrix:
        [ cos  0  -sin ]
        [  0   1    0  ]
        [ sin  0   cos ]
    """
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)

    return np.asarray(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_x_degrees(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)

    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def reconstruct_logic_from_world(camera2world_rotation: np.ndarray) -> np.ndarray:
    """
    Reconstruct the candidate LOGIC rotation directly from CAMERA->WORLD.

    For a CAMERA->WORLD rotation, the second row is the WORLD +Y/up axis
    expressed in camera coordinates.

    The observed Veo construction is:

        logic_y = world_up_in_camera
        logic_x = normalize(cross(logic_y, camera_forward))
        logic_z = cross(logic_x, logic_y)

    with camera_forward = +Z in camera coordinates.

    This removes world heading/yaw while preserving camera pitch/roll relative
    to the ground.
    """
    r_world = np.asarray(
        camera2world_rotation,
        dtype=np.float64,
    ).reshape(3, 3)

    logic_y = normalize(r_world[1, :])

    camera_forward = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    logic_x = normalize(
        np.cross(logic_y, camera_forward)
    )
    logic_z = normalize(
        np.cross(logic_x, logic_y)
    )

    return np.vstack(
        [
            logic_x,
            logic_y,
            logic_z,
        ]
    )


def extract_world_from_logic_yaw(
    world_from_logic: np.ndarray,
) -> float:
    """
    For:
        R_world_from_logic ~= R_y(yaw)

    recover yaw using:
        atan2(R[2,0], R[0,0])
    """
    r = np.asarray(
        world_from_logic,
        dtype=np.float64,
    ).reshape(3, 3)

    return math.degrees(
        math.atan2(
            float(r[2, 0]),
            float(r[0, 0]),
        )
    )


def extract_presentation_x_tilt(
    presentation_from_logic: np.ndarray,
) -> float:
    """
    For:
        R_presentation_from_logic ~= R_x(tilt)

    recover tilt using:
        atan2(R[2,1], R[1,1])
    """
    r = np.asarray(
        presentation_from_logic,
        dtype=np.float64,
    ).reshape(3, 3)

    return math.degrees(
        math.atan2(
            float(r[2, 1]),
            float(r[1, 1]),
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_va(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None

    keys = ("top", "right", "bottom", "left")
    if not all(key in value for key in keys):
        return None

    return {
        key: float(value[key])
        for key in keys
    }


def analyze_file(path: Path) -> SpaceAnalysis:
    data = load_json(path)
    alignment = data["alignment"]

    camera2world = parse_matrix(
        alignment["camera2world"]
    )
    camera2logic = parse_matrix(
        alignment["camera2logic"]
    )
    camera2presentation = parse_matrix(
        alignment["camera2presentation"]
    )

    r_world = camera2world[:3, :3]
    r_logic = camera2logic[:3, :3]
    r_presentation = camera2presentation[:3, :3]

    expected_logic = reconstruct_logic_from_world(
        r_world
    )
    logic_check = compare_rotations(
        r_logic,
        expected_logic,
    )

    # Since both matrices map CAMERA -> target frame:
    #
    #   p_world = R_world p_camera
    #   p_logic = R_logic p_camera
    #
    # therefore:
    #
    #   p_world = R_world R_logic^T p_logic
    #
    # so this is LOGIC -> WORLD.
    world_from_logic = (
        r_world @ r_logic.T
    )

    yaw_deg = extract_world_from_logic_yaw(
        world_from_logic
    )
    expected_world_from_logic = rotation_y_degrees(
        yaw_deg
    )
    yaw_check = compare_rotations(
        world_from_logic,
        expected_world_from_logic,
    )

    # LOGIC -> PRESENTATION:
    #
    #   p_presentation =
    #       R_presentation R_logic^T p_logic
    presentation_from_logic = (
        r_presentation @ r_logic.T
    )

    presentation_tilt_deg = extract_presentation_x_tilt(
        presentation_from_logic
    )
    expected_presentation_from_logic = rotation_x_degrees(
        presentation_tilt_deg
    )
    presentation_check = compare_rotations(
        presentation_from_logic,
        expected_presentation_from_logic,
    )

    return SpaceAnalysis(
        source=str(path),

        camera2world_translation=(
            camera2world[:3, 3].astype(float).tolist()
        ),
        camera2logic_translation=(
            camera2logic[:3, 3].astype(float).tolist()
        ),
        camera2presentation_translation=(
            camera2presentation[:3, 3].astype(float).tolist()
        ),

        camera2world_det=float(
            np.linalg.det(r_world)
        ),
        camera2logic_det=float(
            np.linalg.det(r_logic)
        ),
        camera2presentation_det=float(
            np.linalg.det(r_presentation)
        ),

        camera2world_orthogonality_error=orthogonality_error(
            r_world
        ),
        camera2logic_orthogonality_error=orthogonality_error(
            r_logic
        ),
        camera2presentation_orthogonality_error=orthogonality_error(
            r_presentation
        ),

        logic_reconstruction_error_deg=logic_check.angle_deg,
        logic_reconstruction_max_abs=logic_check.max_abs_matrix_delta,

        world_from_logic_yaw_deg=yaw_deg,
        world_from_logic_yaw_residual_deg=yaw_check.angle_deg,
        world_from_logic_yaw_max_abs=yaw_check.max_abs_matrix_delta,

        presentation_from_logic_x_tilt_deg=presentation_tilt_deg,
        presentation_from_logic_x_residual_deg=presentation_check.angle_deg,
        presentation_from_logic_x_max_abs=presentation_check.max_abs_matrix_delta,

        field_length=float(
            alignment["field_length"]
        ),
        field_width=float(
            alignment["field_width"]
        ),

        va_camera=normalize_va(
            alignment.get("va_camera")
        ),
        va_logic=normalize_va(
            alignment.get("va_logic")
        ),
        va_presentation=normalize_va(
            alignment.get("va_presentation")
        ),
    )


def vector_norm(values: list[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(values, dtype=np.float64)
        )
    )


def print_analysis(result: SpaceAnalysis) -> None:
    print()
    print("=" * 78)
    print(result.source)
    print("=" * 78)

    print(
        f"field: {result.field_length:.6f} x "
        f"{result.field_width:.6f} m"
    )
    print()

    print("Translations")
    print("------------")
    print(
        "camera2world:        "
        f"{result.camera2world_translation} "
        f"| norm={vector_norm(result.camera2world_translation):.9f}"
    )
    print(
        "camera2logic:        "
        f"{result.camera2logic_translation} "
        f"| norm={vector_norm(result.camera2logic_translation):.9f}"
    )
    print(
        "camera2presentation: "
        f"{result.camera2presentation_translation} "
        f"| norm={vector_norm(result.camera2presentation_translation):.9f}"
    )
    print()

    print("Rotation-matrix sanity")
    print("----------------------")
    print(
        "camera2world:        "
        f"det={result.camera2world_det:.10f} "
        f"orth_err={result.camera2world_orthogonality_error:.3e}"
    )
    print(
        "camera2logic:        "
        f"det={result.camera2logic_det:.10f} "
        f"orth_err={result.camera2logic_orthogonality_error:.3e}"
    )
    print(
        "camera2presentation: "
        f"det={result.camera2presentation_det:.10f} "
        f"orth_err={result.camera2presentation_orthogonality_error:.3e}"
    )
    print()

    print("Hypothesis 1: CAMERA -> LOGIC")
    print("----------------------------")
    print(
        "Rebuild LOGIC from WORLD-up + camera-forward:"
    )
    print(
        f"  angular residual: "
        f"{result.logic_reconstruction_error_deg:.9f} deg"
    )
    print(
        f"  max matrix delta: "
        f"{result.logic_reconstruction_max_abs:.3e}"
    )
    print()
    print(
        "LOGIC -> WORLD as pure Y-axis yaw:"
    )
    print(
        f"  yaw:              "
        f"{result.world_from_logic_yaw_deg:.9f} deg"
    )
    print(
        f"  angular residual: "
        f"{result.world_from_logic_yaw_residual_deg:.9f} deg"
    )
    print(
        f"  max matrix delta: "
        f"{result.world_from_logic_yaw_max_abs:.3e}"
    )
    print()

    print("Hypothesis 2: LOGIC -> PRESENTATION")
    print("-----------------------------------")
    print(
        "Approximate pure X-axis display tilt:"
    )
    print(
        f"  tilt:             "
        f"{result.presentation_from_logic_x_tilt_deg:.9f} deg"
    )
    print(
        f"  angular residual: "
        f"{result.presentation_from_logic_x_residual_deg:.9f} deg"
    )
    print(
        f"  max matrix delta: "
        f"{result.presentation_from_logic_x_max_abs:.3e}"
    )
    print()

    print("Angular-view bounds")
    print("-------------------")
    for name, va in (
        ("camera", result.va_camera),
        ("logic", result.va_logic),
        ("presentation", result.va_presentation),
    ):
        if va is None:
            print(f"{name:12s}: unavailable")
            continue

        horizontal_span = va["right"] - va["left"]
        vertical_span = va["top"] - va["bottom"]

        print(
            f"{name:12s}: "
            f"top={va['top']:+.9f} "
            f"right={va['right']:+.9f} "
            f"bottom={va['bottom']:+.9f} "
            f"left={va['left']:+.9f} "
            f"| spans H={horizontal_span:.9f} "
            f"V={vertical_span:.9f}"
        )


def print_cross_file_summary(results: list[SpaceAnalysis]) -> None:
    if len(results) < 2:
        return

    print()
    print("=" * 78)
    print("CROSS-FILE SUMMARY")
    print("=" * 78)

    logic_errors = np.asarray(
        [r.logic_reconstruction_error_deg for r in results],
        dtype=np.float64,
    )
    yaw_residuals = np.asarray(
        [r.world_from_logic_yaw_residual_deg for r in results],
        dtype=np.float64,
    )
    presentation_residuals = np.asarray(
        [r.presentation_from_logic_x_residual_deg for r in results],
        dtype=np.float64,
    )

    print(
        "max LOGIC reconstruction angular residual: "
        f"{float(np.max(logic_errors)):.9f} deg"
    )
    print(
        "max LOGIC->WORLD pure-yaw residual:         "
        f"{float(np.max(yaw_residuals)):.9f} deg"
    )
    print(
        "max LOGIC->PRESENTATION X-tilt residual:    "
        f"{float(np.max(presentation_residuals)):.9f} deg"
    )

    print()
    print("Per-recording derived angles:")
    for r in results:
        print(
            f"  {Path(r.source).name}: "
            f"world heading={r.world_from_logic_yaw_deg:+.6f} deg, "
            f"presentation tilt={r.presentation_from_logic_x_tilt_deg:+.6f} deg"
        )

    print()
    print("Diagnostic interpretation:")
    print(
        "- If LOGIC reconstruction and pure-yaw residuals are tiny, "
        "`camera2logic` is algebraically derivable from `camera2world` "
        "orientation and does not appear to add an independent field pose."
    )
    print(
        "- If PRESENTATION residual is tiny, `camera2presentation` appears "
        "to be a derived visualization/view rotation relative to LOGIC."
    )
    print(
        "- These are algebraic observations from the supplied files, not a "
        "claim about undocumented Veo implementation semantics."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze relationships between camera2world, camera2logic and "
            "camera2presentation in Veo `.veo` files."
        )
    )
    parser.add_argument(
        "calibrations",
        nargs="+",
        type=Path,
        help="One or more `.veo` files.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
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

    results = [
        analyze_file(path)
        for path in args.calibrations
    ]

    for result in results:
        print_analysis(result)

    print_cross_file_summary(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.json_out.write_text(
            json.dumps(
                [asdict(r) for r in results],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print()
        print(f"JSON report written to: {args.json_out}")


if __name__ == "__main__":
    main()
