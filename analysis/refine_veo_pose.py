#!/usr/bin/env python3
"""
Refine Veo camera2world pose from the selected field corners stored in `.veo`.

This is an analysis experiment. It does not modify canonical geometry and does
not overwrite input `.veo` files.

What is optimized
-----------------
Only the 6-DoF rigid pose `camera2world`:

    rotation:    Rodrigues vector (3 parameters)
    translation: XYZ translation (3 parameters)

What stays fixed
----------------
- LEFT/RIGHT intrinsics
- LEFT/RIGHT distortion
- LEFT/RIGHT lens2camera
- field_length
- field_width
- RAW/calibration resolution mapping

Observations
------------
The script uses the valid `selected_corners` already stored by Veo. The
expected field coordinates come from the calibrated field length/width.

Optimization
------------
A small Levenberg-Marquardt solver is implemented here with NumPy/OpenCV only,
so SciPy is not required. The default objective uses a point-wise Huber robust
loss in RAW pixel space; this prevents one noisy selected corner from
dominating the full rig pose.

Examples
--------
Analyze both known calibrations:

    python analysis/refine_veo_pose.py \
        data/camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo \
        data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo

Also write refined `.veo` copies for overlay experiments:

    python analysis/refine_veo_pose.py data/file.veo --write-refined

Then compare visually with:

    python analysis/project_field_overlay.py \
        --calibration output/refine_veo_pose/file_refined.veo \
        --stacked samples/frame.jpg \
        --out-dir output/refined_overlay
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.veo_geometry import VeoCalibration, load_veo_calibration  # noqa: E402


LossName = Literal["linear", "huber"]


@dataclass(frozen=True)
class Observation:
    corner: str
    lens: str
    raw_pixel: np.ndarray
    expected_field: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    parameters: np.ndarray
    camera2world: np.ndarray
    iterations: int
    accepted_steps: int
    initial_cost: float
    final_cost: float
    converged: bool
    reason: str


@dataclass(frozen=True)
class CornerEvaluation:
    corner: str
    lens: str
    observed_raw: np.ndarray
    expected_field: np.ndarray
    projected_raw: np.ndarray
    reprojection_error_px: float
    recovered_field: np.ndarray
    field_error_m: float


def pose_to_parameters(camera2world: np.ndarray) -> np.ndarray:
    """Convert a rigid 4x4 camera2world matrix to [rvec(3), translation(3)]."""
    pose = np.asarray(camera2world, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    return np.concatenate([rvec.reshape(3), pose[:3, 3]])


def parameters_to_pose(parameters: np.ndarray) -> np.ndarray:
    """Convert [rvec(3), translation(3)] to a rigid 4x4 camera2world matrix."""
    p = np.asarray(parameters, dtype=np.float64).reshape(6)
    rotation, _ = cv2.Rodrigues(p[:3].reshape(3, 1))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = p[3:6]
    return pose


def extract_observations(calibration: VeoCalibration) -> list[Observation]:
    observations: list[Observation] = []

    for item in calibration.selected_corner_observations():
        observations.append(
            Observation(
                corner=str(item["corner"]),
                lens=str(item["lens"]),
                raw_pixel=np.asarray(item["raw_pixel"], dtype=np.float64).reshape(2),
                expected_field=np.asarray(
                    item["expected_field"],
                    dtype=np.float64,
                ).reshape(2),
            )
        )

    if len(observations) < 3:
        raise RuntimeError(
            "At least three valid selected-corner observations are required "
            f"to refine pose; found {len(observations)}."
        )

    return observations


def calibration_with_pose(
    calibration: VeoCalibration,
    pose: np.ndarray,
) -> VeoCalibration:
    """
    Return a lightweight calibration copy sharing the immutable lens data but
    using another camera2world pose.

    This deliberately reuses the canonical `field_to_pixel()` and
    `pixel_to_field()` implementations instead of duplicating camera geometry.
    """
    return replace(
        calibration,
        camera2world=np.asarray(pose, dtype=np.float64).reshape(4, 4),
    )


def raw_residual_vector(
    calibration: VeoCalibration,
    observations: list[Observation],
    parameters: np.ndarray,
) -> np.ndarray:
    """
    Return unweighted RAW reprojection residuals:

        [du0, dv0, du1, dv1, ...]

    Each residual is predicted - observed.
    """
    candidate = calibration_with_pose(
        calibration,
        parameters_to_pose(parameters),
    )

    residuals: list[float] = []

    for obs in observations:
        try:
            predicted = candidate.field_to_pixel(
                obs.lens,
                obs.expected_field,
            )
            residual = predicted - obs.raw_pixel
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # Keep the optimizer away from invalid poses.
            residual = np.asarray([1e6, 1e6], dtype=np.float64)

        if not np.all(np.isfinite(residual)):
            residual = np.asarray([1e6, 1e6], dtype=np.float64)

        residuals.extend(
            [
                float(residual[0]),
                float(residual[1]),
            ]
        )

    return np.asarray(residuals, dtype=np.float64)


def point_weights(
    raw_residuals: np.ndarray,
    loss: LossName,
    huber_delta_px: float,
) -> np.ndarray:
    """
    One scalar IRLS weight per observed 2D corner.

    For Huber:
        w = 1                         if ||r_i|| <= delta
        w = delta / ||r_i||           otherwise

    The returned weights are expanded to [u,v] components later.
    """
    residuals = np.asarray(raw_residuals, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(residuals, axis=1)

    if loss == "linear":
        return np.ones_like(errors)

    if loss != "huber":
        raise ValueError(f"Unsupported loss: {loss}")

    weights = np.ones_like(errors)
    mask = errors > huber_delta_px
    weights[mask] = huber_delta_px / np.maximum(errors[mask], 1e-12)
    return weights


def weighted_residual_vector(
    raw_residuals: np.ndarray,
    point_weight_values: np.ndarray,
) -> np.ndarray:
    expanded = np.repeat(
        np.sqrt(np.asarray(point_weight_values, dtype=np.float64)),
        2,
    )
    return raw_residuals * expanded


def robust_cost(
    raw_residuals: np.ndarray,
    loss: LossName,
    huber_delta_px: float,
) -> float:
    residuals = np.asarray(raw_residuals, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(residuals, axis=1)

    if loss == "linear":
        return 0.5 * float(np.sum(errors * errors))

    delta = float(huber_delta_px)
    quadratic = errors <= delta

    cost = 0.5 * np.sum(errors[quadratic] ** 2)

    linear_errors = errors[~quadratic]
    if linear_errors.size:
        cost += np.sum(
            delta * (linear_errors - 0.5 * delta)
        )

    return float(cost)


def numeric_jacobian(
    calibration: VeoCalibration,
    observations: list[Observation],
    parameters: np.ndarray,
    fixed_point_weights: np.ndarray,
    rotation_step: float,
    translation_step: float,
) -> np.ndarray:
    """
    Central-difference Jacobian of the IRLS-weighted RAW residual vector.

    Huber weights are frozen at the current iterate, which is the standard
    IRLS approximation for one LM step.
    """
    p = np.asarray(parameters, dtype=np.float64).reshape(6)

    base_raw = raw_residual_vector(
        calibration,
        observations,
        p,
    )
    residual_count = base_raw.size

    jacobian = np.empty(
        (residual_count, 6),
        dtype=np.float64,
    )

    for axis in range(6):
        step = rotation_step if axis < 3 else translation_step

        plus = p.copy()
        minus = p.copy()
        plus[axis] += step
        minus[axis] -= step

        r_plus = weighted_residual_vector(
            raw_residual_vector(
                calibration,
                observations,
                plus,
            ),
            fixed_point_weights,
        )
        r_minus = weighted_residual_vector(
            raw_residual_vector(
                calibration,
                observations,
                minus,
            ),
            fixed_point_weights,
        )

        jacobian[:, axis] = (
            r_plus - r_minus
        ) / (2.0 * step)

    return jacobian


def refine_pose(
    calibration: VeoCalibration,
    observations: list[Observation],
    *,
    loss: LossName,
    huber_delta_px: float,
    max_iterations: int,
    initial_damping: float,
    rotation_step: float,
    translation_step: float,
    parameter_tolerance: float,
    cost_tolerance: float,
) -> OptimizationResult:
    """
    Refine camera2world with a small Levenberg-Marquardt / IRLS solver.

    No external optimization dependency is required.
    """
    parameters = pose_to_parameters(
        calibration.camera2world
    )

    initial_raw = raw_residual_vector(
        calibration,
        observations,
        parameters,
    )
    initial_cost = robust_cost(
        initial_raw,
        loss,
        huber_delta_px,
    )

    current_cost = initial_cost
    damping = float(initial_damping)

    accepted_steps = 0
    converged = False
    reason = "maximum iterations reached"

    for iteration in range(1, max_iterations + 1):
        current_raw = raw_residual_vector(
            calibration,
            observations,
            parameters,
        )

        weights = point_weights(
            current_raw,
            loss,
            huber_delta_px,
        )

        weighted_current = weighted_residual_vector(
            current_raw,
            weights,
        )

        jacobian = numeric_jacobian(
            calibration,
            observations,
            parameters,
            weights,
            rotation_step,
            translation_step,
        )

        jtj = jacobian.T @ jacobian
        gradient = jacobian.T @ weighted_current

        # Scale damping with local curvature while keeping every parameter
        # numerically regularized.
        diagonal = np.diag(jtj).copy()
        diagonal = np.maximum(diagonal, 1e-12)

        system = jtj + damping * np.diag(diagonal)

        try:
            delta = np.linalg.solve(
                system,
                -gradient,
            )
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        if not np.all(np.isfinite(delta)):
            damping *= 10.0
            continue

        if float(np.linalg.norm(delta)) <= parameter_tolerance:
            converged = True
            reason = "parameter step below tolerance"
            break

        trial_parameters = parameters + delta

        trial_raw = raw_residual_vector(
            calibration,
            observations,
            trial_parameters,
        )
        trial_cost = robust_cost(
            trial_raw,
            loss,
            huber_delta_px,
        )

        if trial_cost < current_cost:
            previous_cost = current_cost

            parameters = trial_parameters
            current_cost = trial_cost
            accepted_steps += 1

            damping = max(damping / 3.0, 1e-12)

            improvement = previous_cost - current_cost
            if improvement <= cost_tolerance:
                converged = True
                reason = "cost improvement below tolerance"
                break
        else:
            damping = min(damping * 10.0, 1e12)

    else:
        iteration = max_iterations

    return OptimizationResult(
        parameters=parameters,
        camera2world=parameters_to_pose(parameters),
        iterations=iteration,
        accepted_steps=accepted_steps,
        initial_cost=initial_cost,
        final_cost=current_cost,
        converged=converged,
        reason=reason,
    )


def evaluate_pose(
    calibration: VeoCalibration,
    observations: list[Observation],
    pose: np.ndarray,
) -> list[CornerEvaluation]:
    candidate = calibration_with_pose(
        calibration,
        pose,
    )

    results: list[CornerEvaluation] = []

    for obs in observations:
        projected = candidate.field_to_pixel(
            obs.lens,
            obs.expected_field,
        )
        reprojection_error = float(
            np.linalg.norm(
                projected - obs.raw_pixel
            )
        )

        recovered_field = candidate.pixel_to_field(
            obs.lens,
            obs.raw_pixel,
        )
        field_error = float(
            np.linalg.norm(
                recovered_field - obs.expected_field
            )
        )

        results.append(
            CornerEvaluation(
                corner=obs.corner,
                lens=obs.lens,
                observed_raw=obs.raw_pixel.copy(),
                expected_field=obs.expected_field.copy(),
                projected_raw=np.asarray(
                    projected,
                    dtype=np.float64,
                ).reshape(2),
                reprojection_error_px=reprojection_error,
                recovered_field=np.asarray(
                    recovered_field,
                    dtype=np.float64,
                ).reshape(2),
                field_error_m=field_error,
            )
        )

    return results


def summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
    }


def rotation_delta_deg(
    original_pose: np.ndarray,
    refined_pose: np.ndarray,
) -> float:
    r0 = np.asarray(original_pose, dtype=np.float64)[:3, :3]
    r1 = np.asarray(refined_pose, dtype=np.float64)[:3, :3]

    delta = r0.T @ r1
    cosine = float(
        np.clip(
            (np.trace(delta) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def translation_delta_m(
    original_pose: np.ndarray,
    refined_pose: np.ndarray,
) -> float:
    t0 = np.asarray(original_pose, dtype=np.float64)[:3, 3]
    t1 = np.asarray(refined_pose, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(t1 - t0))


def fmt_pair(value: np.ndarray, decimals: int = 3) -> str:
    v = np.asarray(value).reshape(2)
    return f"({v[0]:.{decimals}f}, {v[1]:.{decimals}f})"


def fmt_matrix(matrix: np.ndarray) -> str:
    return np.array2string(
        np.asarray(matrix, dtype=np.float64),
        precision=8,
        suppress_small=False,
    )


def print_evaluation_table(
    original: list[CornerEvaluation],
    refined: list[CornerEvaluation],
) -> None:
    print()
    print("Per-corner comparison")
    print("---------------------")

    for before, after in zip(original, refined):
        print(
            f"{before.corner.upper():>2s} {before.lens.upper():>5s} | "
            f"RAW observed {fmt_pair(before.observed_raw)}"
        )
        print(
            f"  reprojection px: "
            f"{before.reprojection_error_px:.6f} -> "
            f"{after.reprojection_error_px:.6f}"
        )
        print(
            f"  field error m:   "
            f"{before.field_error_m:.6f} -> "
            f"{after.field_error_m:.6f}"
        )
        print(
            f"  recovered field: "
            f"{fmt_pair(before.recovered_field)} -> "
            f"{fmt_pair(after.recovered_field)} | "
            f"expected {fmt_pair(before.expected_field)}"
        )
        print()


def print_summary(
    original: list[CornerEvaluation],
    refined: list[CornerEvaluation],
) -> None:
    original_px = summary(
        [x.reprojection_error_px for x in original]
    )
    refined_px = summary(
        [x.reprojection_error_px for x in refined]
    )
    original_m = summary(
        [x.field_error_m for x in original]
    )
    refined_m = summary(
        [x.field_error_m for x in refined]
    )

    print("Summary")
    print("-------")
    print(
        "RAW reprojection error:"
    )
    print(
        f"  original mean={original_px['mean']:.6f} px  "
        f"median={original_px['median']:.6f} px  "
        f"max={original_px['max']:.6f} px  "
        f"rms={original_px['rms']:.6f} px"
    )
    print(
        f"  refined  mean={refined_px['mean']:.6f} px  "
        f"median={refined_px['median']:.6f} px  "
        f"max={refined_px['max']:.6f} px  "
        f"rms={refined_px['rms']:.6f} px"
    )
    print()

    print(
        "Selected-corner field error:"
    )
    print(
        f"  original mean={original_m['mean']:.6f} m  "
        f"median={original_m['median']:.6f} m  "
        f"max={original_m['max']:.6f} m  "
        f"rms={original_m['rms']:.6f} m"
    )
    print(
        f"  refined  mean={refined_m['mean']:.6f} m  "
        f"median={refined_m['median']:.6f} m  "
        f"max={refined_m['max']:.6f} m  "
        f"rms={refined_m['rms']:.6f} m"
    )


def evaluation_to_json(
    result: CornerEvaluation,
) -> dict[str, Any]:
    return {
        "corner": result.corner,
        "lens": result.lens,
        "observed_raw": result.observed_raw.tolist(),
        "expected_field": result.expected_field.tolist(),
        "projected_raw": result.projected_raw.tolist(),
        "reprojection_error_px": result.reprojection_error_px,
        "recovered_field": result.recovered_field.tolist(),
        "field_error_m": result.field_error_m,
    }


def write_refined_veo(
    input_calibration: VeoCalibration,
    refined_pose: np.ndarray,
    output_path: Path,
) -> None:
    """
    Write a copy of the original `.veo` with only alignment.camera2world
    replaced by the refined rigid pose.
    """
    payload = copy.deepcopy(
        input_calibration.raw_json
    )

    flattened = ",".join(
        f"{float(value):.8f}"
        for value in np.asarray(
            refined_pose,
            dtype=np.float64,
        ).reshape(-1)
    )

    payload["alignment"]["camera2world"] = flattened

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def analyze_file(
    path: Path,
    *,
    output_root: Path,
    loss: LossName,
    huber_delta_px: float,
    max_iterations: int,
    write_refined: bool,
) -> None:
    calibration = load_veo_calibration(path)
    observations = extract_observations(
        calibration
    )

    original_pose = np.asarray(
        calibration.camera2world,
        dtype=np.float64,
    ).copy()

    result = refine_pose(
        calibration,
        observations,
        loss=loss,
        huber_delta_px=huber_delta_px,
        max_iterations=max_iterations,
        initial_damping=1e-3,
        rotation_step=1e-6,
        translation_step=1e-5,
        parameter_tolerance=1e-9,
        cost_tolerance=1e-12,
    )

    original_eval = evaluate_pose(
        calibration,
        observations,
        original_pose,
    )
    refined_eval = evaluate_pose(
        calibration,
        observations,
        result.camera2world,
    )

    print()
    print("=" * 78)
    print(path)
    print("=" * 78)
    print(
        f"observations={len(observations)}  "
        f"loss={loss}"
        + (
            f" delta={huber_delta_px:.3f}px"
            if loss == "huber"
            else ""
        )
    )
    print(
        f"optimizer: iterations={result.iterations}  "
        f"accepted={result.accepted_steps}  "
        f"converged={'YES' if result.converged else 'NO'}"
    )
    print(f"reason: {result.reason}")
    print(
        f"robust cost: "
        f"{result.initial_cost:.9f} -> {result.final_cost:.9f}"
    )

    rot_delta = rotation_delta_deg(
        original_pose,
        result.camera2world,
    )
    trans_delta = translation_delta_m(
        original_pose,
        result.camera2world,
    )

    print(
        f"pose delta: rotation={rot_delta:.9f} deg  "
        f"translation={trans_delta:.9f} m"
    )

    print_evaluation_table(
        original_eval,
        refined_eval,
    )
    print_summary(
        original_eval,
        refined_eval,
    )

    print()
    print("Original camera2world")
    print("---------------------")
    print(fmt_matrix(original_pose))
    print()

    print("Refined camera2world")
    print("--------------------")
    print(fmt_matrix(result.camera2world))
    print()

    stem = path.stem
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_root
        / f"{stem}_report.json"
    )

    original_px_summary = summary(
        [x.reprojection_error_px for x in original_eval]
    )
    refined_px_summary = summary(
        [x.reprojection_error_px for x in refined_eval]
    )
    original_m_summary = summary(
        [x.field_error_m for x in original_eval]
    )
    refined_m_summary = summary(
        [x.field_error_m for x in refined_eval]
    )

    report = {
        "source": str(path),
        "loss": {
            "name": loss,
            "huber_delta_px": (
                huber_delta_px
                if loss == "huber"
                else None
            ),
        },
        "optimizer": {
            "iterations": result.iterations,
            "accepted_steps": result.accepted_steps,
            "converged": result.converged,
            "reason": result.reason,
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
        },
        "pose_delta": {
            "rotation_deg": rot_delta,
            "translation_m": trans_delta,
        },
        "camera2world": {
            "original": original_pose.tolist(),
            "refined": result.camera2world.tolist(),
        },
        "summary": {
            "reprojection_px": {
                "original": original_px_summary,
                "refined": refined_px_summary,
            },
            "field_error_m": {
                "original": original_m_summary,
                "refined": refined_m_summary,
            },
        },
        "observations": {
            "original": [
                evaluation_to_json(x)
                for x in original_eval
            ],
            "refined": [
                evaluation_to_json(x)
                for x in refined_eval
            ],
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"JSON report: {report_path}")

    if write_refined:
        refined_path = (
            output_root
            / f"{stem}_refined.veo"
        )
        write_refined_veo(
            calibration,
            result.camera2world,
            refined_path,
        )
        print(f"Refined .veo: {refined_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimentally refine only Veo camera2world pose using "
            "selected corners already stored in the .veo."
        )
    )

    parser.add_argument(
        "calibrations",
        nargs="+",
        type=Path,
        help="One or more .veo files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/refine_veo_pose"),
        help=(
            "Report/output directory "
            "(default: output/refine_veo_pose)."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("linear", "huber"),
        default="huber",
        help=(
            "RAW reprojection loss. Huber is recommended because stored "
            "selected corners may contain outliers (default: huber)."
        ),
    )
    parser.add_argument(
        "--huber-delta-px",
        type=float,
        default=3.0,
        help=(
            "Point-error transition for Huber loss in RAW pixels "
            "(default: 3.0)."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum LM iterations (default: 100).",
    )
    parser.add_argument(
        "--write-refined",
        action="store_true",
        help=(
            "Also write a copy of the .veo with only camera2world replaced. "
            "The input file is never modified."
        ),
    )

    args = parser.parse_args()

    if args.huber_delta_px <= 0:
        parser.error("--huber-delta-px must be > 0")

    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")

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

    for path in args.calibrations:
        analyze_file(
            path,
            output_root=args.output_dir,
            loss=args.loss,
            huber_delta_px=args.huber_delta_px,
            max_iterations=args.max_iterations,
            write_refined=args.write_refined,
        )


if __name__ == "__main__":
    main()
