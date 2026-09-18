#!/usr/bin/env python3
"""
trace_pitch_boundaries.py

Experimental white pitch-boundary tracer for Veo dual-lens RAW frames.

Goal
----
Use Veo's selected field corners ONLY as geometric anchors, then follow the
ACTUAL painted white outer boundary lines in the original RAW lens images.

This is deliberately different from projecting the field through camera2world:
the red boundary is inferred from image evidence (white paint on/next to grass),
starting from the .veo corner locations.

What it traces
--------------
LEFT lens:
    CL -> FL              left goal line
    CL -> overlap/right   close touchline
    FL -> overlap/right   far touchline

RIGHT lens:
    CR -> FR              right goal line
    CR -> overlap/left    close touchline
    FR -> overlap/left    far touchline

Occlusions
----------
The tracker predicts a smooth continuation when the painted line is temporarily
hidden (player, goal frame, shadow, etc.) and attempts to reacquire it later.

After tracing, a robust smooth polynomial is fitted through the accepted image
observations while forcing the .veo corner anchor. Therefore, if visible
sections A and C agree on a smooth boundary, the fitted curve fills the hidden
section B between them.

This is an EXPERIMENT / diagnostic. It does not modify calibration.

Outputs
-------
- left_boundary_overlay.jpg
- right_boundary_overlay.jpg
- stacked_boundary_overlay.jpg
- left_white_candidates.png
- right_white_candidates.png
- boundary_trace.json

Overlay:
    red      = final inferred painted boundary
    green    = image observations accepted by tracker
    yellow   = .veo corner anchor
    magenta  = tracker prediction through a temporary gap

Example
-------
python analysis/trace_pitch_boundaries.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --stacked samples/match2_calibration_frame.jpg \
  --out-dir output/pitch_boundary_match2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.veo_geometry import (  # noqa: E402
    RAW_HEIGHT,
    RAW_WIDTH,
    VeoCalibration,
    load_veo_calibration,
)


LensName = Literal["left", "right"]

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


@dataclass
class TraceResult:
    name: str
    lens: LensName
    seed_corner: str
    seed: np.ndarray
    initial_direction: np.ndarray
    observed_points: np.ndarray
    predicted_gap_points: np.ndarray
    final_curve: np.ndarray
    polynomial: dict[str, Any]
    stopped_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the actual painted white outer pitch boundary starting "
            "from .veo selected corners."
        )
    )

    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Path to the .veo calibration file.",
    )
    parser.add_argument(
        "--stacked",
        required=True,
        type=Path,
        help="2048x2048 stacked frame: LEFT top, RIGHT bottom.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--step-px",
        type=float,
        default=6.0,
        help="Forward tracking step in RAW pixels. Default: 6.",
    )
    parser.add_argument(
        "--search-radius-px",
        type=int,
        default=15,
        help="Radius used to reacquire white paint around prediction. Default: 15.",
    )
    parser.add_argument(
        "--max-gap-px",
        type=float,
        default=90.0,
        help=(
            "Maximum invisible distance that can be bridged before stopping. "
            "Default: 90 px."
        ),
    )
    parser.add_argument(
        "--initial-lookahead-px",
        type=float,
        default=105.0,
        help="Distance used to estimate the initial sideline direction.",
    )
    parser.add_argument(
        "--goal-corridor-px",
        type=float,
        default=55.0,
        help=(
            "Half-width of candidate corridor around the corner-to-corner "
            "goal-line chord. Default: 55."
        ),
    )

    parser.add_argument(
        "--white-value-min",
        type=int,
        default=112,
        help="Minimum HSV value for a white candidate. Default: 112.",
    )
    parser.add_argument(
        "--white-saturation-max",
        type=int,
        default=105,
        help="Maximum HSV saturation for a white candidate. Default: 105.",
    )
    parser.add_argument(
        "--white-chroma-max",
        type=int,
        default=78,
        help="Maximum BGR max-min chroma for a white candidate. Default: 78.",
    )
    parser.add_argument(
        "--grass-support-min",
        type=float,
        default=0.08,
        help=(
            "Minimum blurred neighbourhood fraction classified as grass. "
            "Default: 0.08."
        ),
    )

    parser.add_argument(
        "--poly-degree",
        type=int,
        default=3,
        choices=(1, 2, 3, 4),
        help="Maximum robust polynomial degree. Default: 3.",
    )
    parser.add_argument(
        "--fit-outlier-px",
        type=float,
        default=6.0,
        help="Polynomial iterative outlier threshold. Default: 6 px.",
    )

    return parser.parse_args()


def read_stacked(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if stacked is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = stacked.shape[:2]

    if (w, h) != (RAW_WIDTH, RAW_HEIGHT * 2):
        raise ValueError(
            f"Expected stacked {RAW_WIDTH}x{RAW_HEIGHT * 2}, got {w}x{h}"
        )

    left = stacked[:RAW_HEIGHT].copy()
    right = stacked[RAW_HEIGHT:].copy()

    return stacked, left, right


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError("Cannot normalize zero-length 2D vector")
    return vector / norm


def inside_image(point: np.ndarray, margin: float = 0.0) -> bool:
    x, y = np.asarray(point, dtype=np.float64).reshape(2)
    return (
        margin <= x < RAW_WIDTH - margin
        and margin <= y < RAW_HEIGHT - margin
    )


def build_white_line_evidence(
    image: np.ndarray,
    *,
    value_min: int,
    saturation_max: int,
    chroma_max: int,
    grass_support_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a permissive white-paint likelihood.

    Important choices:
    - yellow/orange auxiliary lines have high saturation -> suppressed;
    - white painted football lines have low chroma -> retained;
    - candidate must have some grass-like neighbourhood support, which removes
      much of the sky, walls and advertising;
    - we keep the mask permissive because the tracker supplies continuity and
      direction priors.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    b = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)

    max_channel = np.maximum(
        np.maximum(b, g),
        r,
    )
    min_channel = np.minimum(
        np.minimum(b, g),
        r,
    )
    chroma = max_channel - min_channel

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    white = (
        (value >= value_min)
        & (saturation <= saturation_max)
        & (chroma <= chroma_max)
    )

    # Broad grass detector. It intentionally accepts lit and shaded artificial
    # turf rather than requiring one narrow HSV hue range.
    green_dominance = (
        (g >= r + 3)
        & (g >= b + 3)
        & (g >= 35)
    )

    grass_u8 = (
        green_dominance.astype(np.uint8) * 255
    )

    grass_support = cv2.GaussianBlur(
        grass_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=8.0,
        sigmaY=8.0,
    )

    candidate = (
        white
        & (grass_support >= grass_support_min)
    ).astype(np.uint8) * 255

    # Join small holes along painted lines without aggressively joining
    # independent markings.
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    # Likelihood, not only binary mask. Bright low-saturation pixels with grass
    # support receive larger values.
    value_score = np.clip(
        (value.astype(np.float32) - value_min)
        / max(1.0, 255.0 - value_min),
        0.0,
        1.0,
    )

    saturation_score = np.clip(
        1.0
        - saturation.astype(np.float32)
        / max(1.0, float(saturation_max)),
        0.0,
        1.0,
    )

    chroma_score = np.clip(
        1.0
        - chroma.astype(np.float32)
        / max(1.0, float(chroma_max)),
        0.0,
        1.0,
    )

    likelihood = (
        0.40 * value_score
        + 0.30 * saturation_score
        + 0.20 * chroma_score
        + 0.10 * np.clip(grass_support, 0.0, 1.0)
    )

    likelihood *= (
        candidate.astype(np.float32) / 255.0
    )

    return candidate, likelihood, grass_support


def local_line_score(
    likelihood: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    length_px: float,
    *,
    sample_step: float = 2.0,
    perpendicular_radius: int = 3,
) -> float:
    """
    Score how much white-line evidence exists along a short ray.

    At each longitudinal sample, take the best likelihood in a small
    perpendicular strip. This tolerates a few pixels of seed error/line width.
    """
    direction = unit(direction)
    perp = np.asarray(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    scores: list[float] = []

    distance = 4.0
    while distance <= length_px:
        center = start + direction * distance

        if not inside_image(center, margin=2):
            break

        local_best = 0.0

        for offset in range(
            -perpendicular_radius,
            perpendicular_radius + 1,
        ):
            point = center + perp * float(offset)
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))

            if (
                0 <= x < RAW_WIDTH
                and 0 <= y < RAW_HEIGHT
            ):
                local_best = max(
                    local_best,
                    float(likelihood[y, x]),
                )

        scores.append(local_best)
        distance += sample_step

    if not scores:
        return -1e9

    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    # Reward sustained evidence much more than one isolated bright object.
    positive_fraction = float(
        np.mean(values > 0.05)
    )

    return (
        float(np.mean(values))
        + 0.75 * positive_fraction
    )


def estimate_sideline_direction(
    likelihood: np.ndarray,
    seed: np.ndarray,
    goal_direction: np.ndarray,
    horizontal_sign: float,
    lookahead_px: float,
) -> np.ndarray:
    """
    At a corner two painted branches meet:
        1. goal line
        2. touchline

    We know the goal-line direction from the two .veo corners in this lens.
    Search directions that:
      - progress toward the overlap/centre of the dual-lens view;
      - are not approximately the goal-line branch;
      - contain sustained white evidence.
    """
    goal_unit = unit(goal_direction)

    best_score = -1e18
    best_direction: np.ndarray | None = None

    # The sideline must travel toward the opposite lens:
    # LEFT -> increasing x, RIGHT -> decreasing x.
    for angle_deg in np.linspace(
        -82.0,
        82.0,
        165,
    ):
        angle = math.radians(float(angle_deg))

        direction = np.asarray(
            [
                horizontal_sign * math.cos(angle),
                math.sin(angle),
            ],
            dtype=np.float64,
        )
        direction = unit(direction)

        horizontal_progress = (
            horizontal_sign * direction[0]
        )

        if horizontal_progress < 0.20:
            continue

        # Reject the branch that simply follows the known goal line.
        alignment_with_goal = abs(
            float(np.dot(direction, goal_unit))
        )
        if alignment_with_goal > 0.88:
            continue

        evidence = local_line_score(
            likelihood,
            seed,
            direction,
            lookahead_px,
        )

        # Mild preference for progressing toward overlap rather than vertically.
        score = (
            evidence
            + 0.12 * horizontal_progress
            - 0.15 * alignment_with_goal
        )

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        raise RuntimeError(
            "Could not estimate initial sideline direction"
        )

    return best_direction


def candidate_pixels_in_window(
    mask: np.ndarray,
    center: np.ndarray,
    radius: int,
) -> np.ndarray:
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))

    x0 = max(0, cx - radius)
    x1 = min(RAW_WIDTH, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(RAW_HEIGHT, cy + radius + 1)

    roi = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(roi)

    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    return np.column_stack(
        [
            xs.astype(np.float64) + x0,
            ys.astype(np.float64) + y0,
        ]
    )


def choose_tracking_candidate(
    candidates: np.ndarray,
    likelihood: np.ndarray,
    current: np.ndarray,
    predicted: np.ndarray,
    direction: np.ndarray,
    step_px: float,
    horizontal_sign: float,
) -> np.ndarray | None:
    if candidates.size == 0:
        return None

    direction = unit(direction)
    perp = np.asarray(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    relative_to_current = (
        candidates - current.reshape(1, 2)
    )
    relative_to_prediction = (
        candidates - predicted.reshape(1, 2)
    )

    forward = (
        relative_to_current @ direction
    )
    perpendicular = np.abs(
        relative_to_prediction @ perp
    )
    prediction_distance = np.linalg.norm(
        relative_to_prediction,
        axis=1,
    )

    # Do not jump backwards or to a crossing line that heads away from overlap.
    x_progress = (
        horizontal_sign
        * (
            candidates[:, 0]
            - current[0]
        )
    )

    valid = (
        (forward > 0.15 * step_px)
        & (x_progress > -1.5)
    )

    if not np.any(valid):
        return None

    candidates = candidates[valid]
    perpendicular = perpendicular[valid]
    prediction_distance = prediction_distance[valid]
    forward = forward[valid]

    xs = np.clip(
        np.round(candidates[:, 0]).astype(int),
        0,
        RAW_WIDTH - 1,
    )
    ys = np.clip(
        np.round(candidates[:, 1]).astype(int),
        0,
        RAW_HEIGHT - 1,
    )

    evidence = likelihood[ys, xs]

    score = (
        4.0 * evidence
        - 0.080 * perpendicular
        - 0.035 * prediction_distance
        - 0.018 * np.abs(forward - step_px)
    )

    best = int(np.argmax(score))

    if float(score[best]) < -0.25:
        return None

    selected = candidates[best]

    # Center the point on the local painted stripe using nearby candidate
    # evidence. This makes tracking less dependent on which edge pixel won.
    local_delta = candidates - selected.reshape(1, 2)
    close = (
        np.linalg.norm(
            local_delta,
            axis=1,
        )
        <= 3.5
    )

    local = candidates[close]
    local_evidence = evidence[close]

    if local.shape[0] >= 2:
        weights = (
            local_evidence.astype(np.float64)
            + 0.05
        )
        selected = np.average(
            local,
            axis=0,
            weights=weights,
        )

    return selected.astype(np.float64)


def direction_from_recent_points(
    points: list[np.ndarray],
    previous_direction: np.ndarray,
    horizontal_sign: float,
) -> np.ndarray:
    if len(points) < 4:
        return previous_direction

    recent = np.asarray(
        points[-8:],
        dtype=np.float64,
    )

    centered = (
        recent - np.mean(recent, axis=0)
    )

    covariance = (
        centered.T @ centered
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    direction = eigenvectors[
        :,
        int(np.argmax(eigenvalues)),
    ]

    if (
        horizontal_sign * direction[0]
        < 0.0
    ):
        direction = -direction

    direction = unit(direction)

    previous_direction = unit(
        previous_direction
    )

    # Smooth orientation update so one intersection/bright object cannot
    # abruptly rotate the tracker.
    mixed = (
        0.72 * previous_direction
        + 0.28 * direction
    )

    if np.linalg.norm(mixed) < 1e-8:
        return previous_direction

    return unit(mixed)


def track_sideline(
    *,
    name: str,
    lens: LensName,
    seed_corner: str,
    seed: np.ndarray,
    goal_direction: np.ndarray,
    mask: np.ndarray,
    likelihood: np.ndarray,
    horizontal_sign: float,
    step_px: float,
    search_radius_px: int,
    max_gap_px: float,
    initial_lookahead_px: float,
) -> TraceResult:
    direction = estimate_sideline_direction(
        likelihood,
        seed,
        goal_direction,
        horizontal_sign,
        initial_lookahead_px,
    )

    initial_direction = direction.copy()

    current = seed.astype(
        np.float64,
        copy=True,
    )

    observed: list[np.ndarray] = [
        current.copy()
    ]
    prediction_path: list[np.ndarray] = []

    invisible_distance = 0.0
    max_iterations = int(
        1.5 * RAW_WIDTH / max(step_px, 1.0)
    )

    stopped_reason = "iteration_limit"

    for _ in range(max_iterations):
        predicted = (
            current + direction * step_px
        )

        if not inside_image(
            predicted,
            margin=2.0,
        ):
            stopped_reason = "image_boundary"
            break

        candidates = candidate_pixels_in_window(
            mask,
            predicted,
            search_radius_px,
        )

        selected = choose_tracking_candidate(
            candidates,
            likelihood,
            current,
            predicted,
            direction,
            step_px,
            horizontal_sign,
        )

        if selected is None:
            prediction_path.append(
                predicted.copy()
            )
            current = predicted
            invisible_distance += step_px

            if invisible_distance > max_gap_px:
                stopped_reason = "max_gap_exceeded"
                break

            continue

        # Successful reacquisition. The final smooth fit will bridge every
        # predicted point between the previous and current observations.
        current = selected
        observed.append(
            selected.copy()
        )
        invisible_distance = 0.0

        direction = direction_from_recent_points(
            observed,
            direction,
            horizontal_sign,
        )

        # Once the tracker reaches the image edge toward the overlap, there is
        # nothing else useful to recover in this lens.
        if (
            horizontal_sign > 0
            and current[0] >= RAW_WIDTH - 3
        ):
            stopped_reason = "target_side_edge"
            break

        if (
            horizontal_sign < 0
            and current[0] <= 2
        ):
            stopped_reason = "target_side_edge"
            break

    observed_array = np.asarray(
        observed,
        dtype=np.float64,
    )

    gap_array = (
        np.asarray(
            prediction_path,
            dtype=np.float64,
        )
        if prediction_path
        else np.empty((0, 2), dtype=np.float64)
    )

    return TraceResult(
        name=name,
        lens=lens,
        seed_corner=seed_corner,
        seed=seed.copy(),
        initial_direction=initial_direction,
        observed_points=observed_array,
        predicted_gap_points=gap_array,
        final_curve=np.empty((0, 2), dtype=np.float64),
        polynomial={},
        stopped_reason=stopped_reason,
    )


def point_segment_distance(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point return:
      - perpendicular distance to finite segment a->b
      - segment parameter t
    """
    ab = b - a
    denominator = float(
        np.dot(ab, ab)
    )

    if denominator < 1e-12:
        raise RuntimeError(
            "Goal-line corner anchors coincide"
        )

    ap = points - a.reshape(1, 2)

    t = (
        ap @ ab
    ) / denominator

    projection = (
        a.reshape(1, 2)
        + t.reshape(-1, 1)
        * ab.reshape(1, 2)
    )

    distance = np.linalg.norm(
        points - projection,
        axis=1,
    )

    return distance, t


def collect_goal_line_candidates(
    mask: np.ndarray,
    likelihood: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
    corridor_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)

    points = np.column_stack(
        [
            xs.astype(np.float64),
            ys.astype(np.float64),
        ]
    )

    if points.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    distance, t = point_segment_distance(
        points,
        corner_a,
        corner_b,
    )

    valid = (
        (distance <= corridor_px)
        & (t >= -0.05)
        & (t <= 1.05)
    )

    selected = points[valid]

    if selected.shape[0] == 0:
        return (
            selected,
            np.empty((0,), dtype=np.float64),
        )

    x_int = np.clip(
        np.round(selected[:, 0]).astype(int),
        0,
        RAW_WIDTH - 1,
    )
    y_int = np.clip(
        np.round(selected[:, 1]).astype(int),
        0,
        RAW_HEIGHT - 1,
    )

    weights = likelihood[
        y_int,
        x_int,
    ].astype(np.float64)

    return selected, weights


def robust_polyfit(
    independent: np.ndarray,
    dependent: np.ndarray,
    *,
    degree: int,
    weights: np.ndarray | None,
    outlier_px: float,
    anchor_independent: np.ndarray | None = None,
    anchor_dependent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    independent = np.asarray(
        independent,
        dtype=np.float64,
    ).reshape(-1)
    dependent = np.asarray(
        dependent,
        dtype=np.float64,
    ).reshape(-1)

    if independent.size != dependent.size:
        raise ValueError(
            "Independent/dependent size mismatch"
        )

    if independent.size < 2:
        raise RuntimeError(
            "Too few points for polynomial fit"
        )

    if weights is None:
        weights = np.ones_like(
            independent,
            dtype=np.float64,
        )
    else:
        weights = np.asarray(
            weights,
            dtype=np.float64,
        ).reshape(-1)

    if (
        anchor_independent is not None
        and anchor_dependent is not None
    ):
        anchor_independent = np.asarray(
            anchor_independent,
            dtype=np.float64,
        ).reshape(-1)
        anchor_dependent = np.asarray(
            anchor_dependent,
            dtype=np.float64,
        ).reshape(-1)

        independent = np.concatenate(
            [
                independent,
                np.repeat(
                    anchor_independent,
                    12,
                ),
            ]
        )
        dependent = np.concatenate(
            [
                dependent,
                np.repeat(
                    anchor_dependent,
                    12,
                ),
            ]
        )
        weights = np.concatenate(
            [
                weights,
                np.full(
                    12 * anchor_independent.size,
                    8.0,
                    dtype=np.float64,
                ),
            ]
        )

    finite = (
        np.isfinite(independent)
        & np.isfinite(dependent)
        & np.isfinite(weights)
    )

    keep = finite.copy()

    for _ in range(5):
        count = int(
            np.count_nonzero(keep)
        )

        current_degree = min(
            degree,
            max(1, count - 1),
        )

        if count < current_degree + 1:
            break

        coeff = np.polyfit(
            independent[keep],
            dependent[keep],
            deg=current_degree,
            w=np.maximum(
                weights[keep],
                0.05,
            ),
        )

        predicted = np.polyval(
            coeff,
            independent,
        )
        residual = np.abs(
            dependent - predicted
        )

        # Robust threshold: use requested threshold, but allow noisier initial
        # candidate clouds to settle before the final iterations.
        median = float(
            np.median(
                residual[keep]
            )
        )
        mad = float(
            np.median(
                np.abs(
                    residual[keep] - median
                )
            )
        )

        adaptive = max(
            outlier_px,
            median + 2.8 * max(mad, 0.5),
        )

        new_keep = (
            finite
            & (residual <= adaptive)
        )

        if np.array_equal(
            new_keep,
            keep,
        ):
            keep = new_keep
            break

        keep = new_keep

    count = int(
        np.count_nonzero(keep)
    )
    final_degree = min(
        degree,
        max(1, count - 1),
    )

    if count < final_degree + 1:
        raise RuntimeError(
            "Too few inliers after robust polynomial fit"
        )

    coeff = np.polyfit(
        independent[keep],
        dependent[keep],
        deg=final_degree,
        w=np.maximum(
            weights[keep],
            0.05,
        ),
    )

    return coeff, keep


def fit_sideline_curve(
    result: TraceResult,
    *,
    degree: int,
    outlier_px: float,
) -> TraceResult:
    points = result.observed_points

    if points.shape[0] < 4:
        result.final_curve = points.copy()
        result.polynomial = {
            "status": "too_few_observations"
        }
        return result

    # Touchlines move monotonically across x in each lens, so y=f(x) is the
    # most stable representation.
    independent = points[:, 0]
    dependent = points[:, 1]

    # Mildly weight observations near the seed and recent ones equally; image
    # continuity is already doing the heavy lifting.
    weights = np.ones(
        independent.shape[0],
        dtype=np.float64,
    )

    coeff, keep = robust_polyfit(
        independent,
        dependent,
        degree=degree,
        weights=weights,
        outlier_px=outlier_px,
        anchor_independent=np.asarray(
            [result.seed[0]]
        ),
        anchor_dependent=np.asarray(
            [result.seed[1]]
        ),
    )

    inlier_points = points[keep[: points.shape[0]]]

    if inlier_points.shape[0] < 2:
        inlier_points = points

    x_start = float(result.seed[0])
    x_end = float(
        inlier_points[-1, 0]
    )

    # Keep generation in the direction the tracker actually travelled.
    sample_count = max(
        2,
        int(
            abs(x_end - x_start)
        )
        + 1,
    )

    xs = np.linspace(
        x_start,
        x_end,
        sample_count,
    )
    ys = np.polyval(
        coeff,
        xs,
    )

    valid = (
        (xs >= 0.0)
        & (xs < RAW_WIDTH)
        & (ys >= 0.0)
        & (ys < RAW_HEIGHT)
    )

    result.final_curve = np.column_stack(
        [
            xs[valid],
            ys[valid],
        ]
    )

    result.polynomial = {
        "representation": "y=f(x)",
        "degree": int(
            len(coeff) - 1
        ),
        "coefficients": coeff.tolist(),
        "observations": int(
            points.shape[0]
        ),
        "fit_inliers": int(
            np.count_nonzero(
                keep[: points.shape[0]]
            )
        ),
    }

    return result


def fit_goal_line(
    *,
    name: str,
    lens: LensName,
    seed_corner: str,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
    mask: np.ndarray,
    likelihood: np.ndarray,
    corridor_px: float,
    degree: int,
    outlier_px: float,
) -> TraceResult:
    candidates, weights = (
        collect_goal_line_candidates(
            mask,
            likelihood,
            corner_a,
            corner_b,
            corridor_px,
        )
    )

    if candidates.shape[0] < 8:
        # Fall back to the corner chord rather than hallucinating a curve.
        curve = np.linspace(
            corner_a,
            corner_b,
            300,
        )

        return TraceResult(
            name=name,
            lens=lens,
            seed_corner=seed_corner,
            seed=corner_a.copy(),
            initial_direction=unit(
                corner_b - corner_a
            ),
            observed_points=candidates,
            predicted_gap_points=np.empty(
                (0, 2),
                dtype=np.float64,
            ),
            final_curve=curve,
            polynomial={
                "status": "fallback_corner_chord",
                "candidate_count": int(
                    candidates.shape[0]
                ),
            },
            stopped_reason="goal_line_fit",
        )

    delta = np.abs(
        corner_b - corner_a
    )

    # Choose the axis with the larger anchor span to avoid fitting a vertical
    # line as y=f(x).
    if delta[1] >= delta[0]:
        independent = candidates[:, 1]
        dependent = candidates[:, 0]

        anchor_independent = np.asarray(
            [corner_a[1], corner_b[1]],
            dtype=np.float64,
        )
        anchor_dependent = np.asarray(
            [corner_a[0], corner_b[0]],
            dtype=np.float64,
        )

        representation = "x=f(y)"

    else:
        independent = candidates[:, 0]
        dependent = candidates[:, 1]

        anchor_independent = np.asarray(
            [corner_a[0], corner_b[0]],
            dtype=np.float64,
        )
        anchor_dependent = np.asarray(
            [corner_a[1], corner_b[1]],
            dtype=np.float64,
        )

        representation = "y=f(x)"

    coeff, keep = robust_polyfit(
        independent,
        dependent,
        degree=degree,
        weights=weights,
        outlier_px=outlier_px,
        anchor_independent=anchor_independent,
        anchor_dependent=anchor_dependent,
    )

    start = float(
        anchor_independent[0]
    )
    end = float(
        anchor_independent[1]
    )

    samples = max(
        2,
        int(abs(end - start)) + 1,
    )

    independent_samples = np.linspace(
        start,
        end,
        samples,
    )

    dependent_samples = np.polyval(
        coeff,
        independent_samples,
    )

    if representation == "x=f(y)":
        curve = np.column_stack(
            [
                dependent_samples,
                independent_samples,
            ]
        )
    else:
        curve = np.column_stack(
            [
                independent_samples,
                dependent_samples,
            ]
        )

    valid = (
        (curve[:, 0] >= 0.0)
        & (curve[:, 0] < RAW_WIDTH)
        & (curve[:, 1] >= 0.0)
        & (curve[:, 1] < RAW_HEIGHT)
    )

    curve = curve[valid]

    return TraceResult(
        name=name,
        lens=lens,
        seed_corner=seed_corner,
        seed=corner_a.copy(),
        initial_direction=unit(
            corner_b - corner_a
        ),
        observed_points=candidates[keep[: candidates.shape[0]]],
        predicted_gap_points=np.empty(
            (0, 2),
            dtype=np.float64,
        ),
        final_curve=curve,
        polynomial={
            "status": "ok",
            "representation": representation,
            "degree": int(
                len(coeff) - 1
            ),
            "coefficients": coeff.tolist(),
            "candidate_count": int(
                candidates.shape[0]
            ),
            "fit_inliers": int(
                np.count_nonzero(
                    keep[: candidates.shape[0]]
                )
            ),
        },
        stopped_reason="goal_line_fit",
    )


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if points.shape[0] < 2:
        return

    integer_points = np.round(
        points
    ).astype(np.int32)

    integer_points[:, 0] = np.clip(
        integer_points[:, 0],
        0,
        RAW_WIDTH - 1,
    )
    integer_points[:, 1] = np.clip(
        integer_points[:, 1],
        0,
        RAW_HEIGHT - 1,
    )

    cv2.polylines(
        image,
        [
            integer_points.reshape(
                -1,
                1,
                2,
            )
        ],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_trace_debug(
    image: np.ndarray,
    result: TraceResult,
) -> None:
    # Gap prediction is diagnostic only. The final red fit is what the script
    # considers the inferred pitch boundary.
    for point in result.predicted_gap_points[::2]:
        cv2.circle(
            image,
            tuple(
                np.round(point).astype(int)
            ),
            1,
            MAGENTA,
            -1,
            cv2.LINE_AA,
        )

    for point in result.observed_points[::3]:
        cv2.circle(
            image,
            tuple(
                np.round(point).astype(int)
            ),
            2,
            GREEN,
            -1,
            cv2.LINE_AA,
        )

    draw_polyline(
        image,
        result.final_curve,
        RED,
        3,
    )

    seed_xy = tuple(
        np.round(
            result.seed
        ).astype(int)
    )

    cv2.circle(
        image,
        seed_xy,
        7,
        YELLOW,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        result.name,
        (
            seed_xy[0] + 8,
            max(
                18,
                seed_xy[1] - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_corner(
    image: np.ndarray,
    name: str,
    point: np.ndarray,
) -> None:
    center = tuple(
        np.round(point).astype(int)
    )

    cv2.circle(
        image,
        center,
        5,
        YELLOW,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        name.upper(),
        (
            center[0] + 6,
            center[1] - 6,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        YELLOW,
        1,
        cv2.LINE_AA,
    )


def require_corner(
    calibration: VeoCalibration,
    corner: str,
    lens: LensName,
) -> np.ndarray:
    point = (
        calibration.selected_corner_raw_pixel(
            corner,
            lens,
        )
    )

    if point is None:
        raise RuntimeError(
            f".veo has no valid {corner.upper()} selected corner "
            f"for {lens.upper()} lens"
        )

    return np.asarray(
        point,
        dtype=np.float64,
    )


def trace_all_boundaries(
    calibration: VeoCalibration,
    left: np.ndarray,
    right: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    list[TraceResult],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    left_mask, left_likelihood, _ = (
        build_white_line_evidence(
            left,
            value_min=args.white_value_min,
            saturation_max=args.white_saturation_max,
            chroma_max=args.white_chroma_max,
            grass_support_min=args.grass_support_min,
        )
    )

    right_mask, right_likelihood, _ = (
        build_white_line_evidence(
            right,
            value_min=args.white_value_min,
            saturation_max=args.white_saturation_max,
            chroma_max=args.white_chroma_max,
            grass_support_min=args.grass_support_min,
        )
    )

    cl = require_corner(
        calibration,
        "cl",
        "left",
    )
    fl = require_corner(
        calibration,
        "fl",
        "left",
    )
    cr = require_corner(
        calibration,
        "cr",
        "right",
    )
    fr = require_corner(
        calibration,
        "fr",
        "right",
    )

    results: list[TraceResult] = []

    # Goal/end lines: both endpoints are directly anchored by .veo.
    results.append(
        fit_goal_line(
            name="LEFT_GOAL_LINE",
            lens="left",
            seed_corner="cl/fl",
            corner_a=cl,
            corner_b=fl,
            mask=left_mask,
            likelihood=left_likelihood,
            corridor_px=args.goal_corridor_px,
            degree=args.poly_degree,
            outlier_px=args.fit_outlier_px,
        )
    )

    results.append(
        fit_goal_line(
            name="RIGHT_GOAL_LINE",
            lens="right",
            seed_corner="cr/fr",
            corner_a=cr,
            corner_b=fr,
            mask=right_mask,
            likelihood=right_likelihood,
            corridor_px=args.goal_corridor_px,
            degree=args.poly_degree,
            outlier_px=args.fit_outlier_px,
        )
    )

    # Touchlines: start from the exact .veo corner and follow the actual white
    # painted branch toward the lens overlap.
    left_goal_vector = fl - cl
    right_goal_vector = fr - cr

    results.append(
        fit_sideline_curve(
            track_sideline(
                name="CLOSE_TOUCHLINE_LEFT",
                lens="left",
                seed_corner="cl",
                seed=cl,
                goal_direction=left_goal_vector,
                mask=left_mask,
                likelihood=left_likelihood,
                horizontal_sign=+1.0,
                step_px=args.step_px,
                search_radius_px=args.search_radius_px,
                max_gap_px=args.max_gap_px,
                initial_lookahead_px=args.initial_lookahead_px,
            ),
            degree=args.poly_degree,
            outlier_px=args.fit_outlier_px,
        )
    )

    results.append(
        fit_sideline_curve(
            track_sideline(
                name="FAR_TOUCHLINE_LEFT",
                lens="left",
                seed_corner="fl",
                seed=fl,
                goal_direction=-left_goal_vector,
                mask=left_mask,
                likelihood=left_likelihood,
                horizontal_sign=+1.0,
                step_px=args.step_px,
                search_radius_px=args.search_radius_px,
                max_gap_px=args.max_gap_px,
                initial_lookahead_px=args.initial_lookahead_px,
            ),
            degree=args.poly_degree,
            outlier_px=args.fit_outlier_px,
        )
    )

    results.append(
        fit_sideline_curve(
            track_sideline(
                name="CLOSE_TOUCHLINE_RIGHT",
                lens="right",
                seed_corner="cr",
                seed=cr,
                goal_direction=right_goal_vector,
                mask=right_mask,
                likelihood=right_likelihood,
                horizontal_sign=-1.0,
                step_px=args.step_px,
                search_radius_px=args.search_radius_px,
                max_gap_px=args.max_gap_px,
                initial_lookahead_px=args.initial_lookahead_px,
            ),
            degree=args.poly_degree,
            outlier_px=args.fit_outlier_px,
        )
    )

    results.append(
        fit_sideline_curve(
            track_sideline(
                name="FAR_TOUCHLINE_RIGHT",
                lens="right",
                seed_corner="fr",
                seed=fr,
                goal_direction=-right_goal_vector,
                mask=right_mask,
                likelihood=right_likelihood,
                horizontal_sign=-1.0,
                step_px=args.step_px,
                search_radius_px=args.search_radius_px,
                max_gap_px=args.max_gap_px,
                initial_lookahead_px=args.initial_lookahead_px,
            ),
            degree=args.poly_degree,
            outlier_px=args.fit_outlier_px,
        )
    )

    return (
        results,
        left_mask,
        right_mask,
        left_likelihood,
        right_likelihood,
    )


def serialize_trace(
    result: TraceResult,
) -> dict[str, Any]:
    return {
        "name": result.name,
        "lens": result.lens,
        "seed_corner": result.seed_corner,
        "seed_raw": result.seed.tolist(),
        "initial_direction": (
            result.initial_direction.tolist()
        ),
        "stopped_reason": result.stopped_reason,
        "observed_count": int(
            result.observed_points.shape[0]
        ),
        "predicted_gap_count": int(
            result.predicted_gap_points.shape[0]
        ),
        "final_curve_count": int(
            result.final_curve.shape[0]
        ),
        "polynomial": result.polynomial,
        "observed_points_raw": (
            result.observed_points.tolist()
        ),
        "predicted_gap_points_raw": (
            result.predicted_gap_points.tolist()
        ),
        "final_curve_raw": (
            result.final_curve.tolist()
        ),
    }


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = load_veo_calibration(
        args.calibration
    )

    _, left, right = read_stacked(
        args.stacked
    )

    (
        traces,
        left_mask,
        right_mask,
        _left_likelihood,
        _right_likelihood,
    ) = trace_all_boundaries(
        calibration,
        left,
        right,
        args,
    )

    left_overlay = left.copy()
    right_overlay = right.copy()

    # Mark canonical anchors independently of tracking.
    for corner in ("cl", "fl"):
        point = require_corner(
            calibration,
            corner,
            "left",
        )
        draw_corner(
            left_overlay,
            corner,
            point,
        )

    for corner in ("cr", "fr"):
        point = require_corner(
            calibration,
            corner,
            "right",
        )
        draw_corner(
            right_overlay,
            corner,
            point,
        )

    for trace in traces:
        target = (
            left_overlay
            if trace.lens == "left"
            else right_overlay
        )

        draw_trace_debug(
            target,
            trace,
        )

    stacked_overlay = np.vstack(
        [
            left_overlay,
            right_overlay,
        ]
    )

    left_overlay_path = (
        args.out_dir
        / "left_boundary_overlay.jpg"
    )
    right_overlay_path = (
        args.out_dir
        / "right_boundary_overlay.jpg"
    )
    stacked_overlay_path = (
        args.out_dir
        / "stacked_boundary_overlay.jpg"
    )
    left_mask_path = (
        args.out_dir
        / "left_white_candidates.png"
    )
    right_mask_path = (
        args.out_dir
        / "right_white_candidates.png"
    )
    report_path = (
        args.out_dir
        / "boundary_trace.json"
    )

    cv2.imwrite(
        str(left_overlay_path),
        left_overlay,
    )
    cv2.imwrite(
        str(right_overlay_path),
        right_overlay,
    )
    cv2.imwrite(
        str(stacked_overlay_path),
        stacked_overlay,
    )
    cv2.imwrite(
        str(left_mask_path),
        left_mask,
    )
    cv2.imwrite(
        str(right_mask_path),
        right_mask,
    )

    payload = {
        "calibration": str(
            args.calibration
        ),
        "stacked": str(
            args.stacked
        ),
        "purpose": (
            "Trace actual painted WHITE outer pitch boundaries from .veo "
            "selected corner anchors without using camera2world projection."
        ),
        "parameters": {
            "step_px": args.step_px,
            "search_radius_px": (
                args.search_radius_px
            ),
            "max_gap_px": args.max_gap_px,
            "initial_lookahead_px": (
                args.initial_lookahead_px
            ),
            "goal_corridor_px": (
                args.goal_corridor_px
            ),
            "white_value_min": (
                args.white_value_min
            ),
            "white_saturation_max": (
                args.white_saturation_max
            ),
            "white_chroma_max": (
                args.white_chroma_max
            ),
            "grass_support_min": (
                args.grass_support_min
            ),
            "poly_degree": args.poly_degree,
            "fit_outlier_px": (
                args.fit_outlier_px
            ),
        },
        "legend": {
            "red": (
                "final inferred painted boundary"
            ),
            "green": (
                "accepted image observations"
            ),
            "yellow": ".veo corner anchor",
            "magenta": (
                "tracker prediction while line was not observed"
            ),
        },
        "traces": [
            serialize_trace(trace)
            for trace in traces
        ],
    }

    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Veo white pitch-boundary tracing")
    print("=" * 88)

    for trace in traces:
        print(
            f"{trace.name:<24} "
            f"{trace.lens.upper():<5} "
            f"observed={trace.observed_points.shape[0]:>4d} "
            f"gap_predictions={trace.predicted_gap_points.shape[0]:>4d} "
            f"curve={trace.final_curve.shape[0]:>4d} "
            f"stop={trace.stopped_reason}"
        )

    print()
    print("Written:")
    print(f"  {left_overlay_path}")
    print(f"  {right_overlay_path}")
    print(f"  {stacked_overlay_path}")
    print(f"  {left_mask_path}")
    print(f"  {right_mask_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
