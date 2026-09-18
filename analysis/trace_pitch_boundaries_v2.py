#!/usr/bin/env python3
"""
trace_pitch_boundaries_v2.py

V2: image-driven tracing of the ACTUAL painted white football-pitch boundaries
in Veo dual-lens RAW frames.

Key change from V1
------------------
V1 depended too much on an absolute "white pixel" threshold and sequential
tracking. That fails when a white line enters shadow even though it is still
visually obvious.

V2 instead uses:
  - local brightness contrast against the surrounding turf;
  - colour neutrality / low saturation to reject yellow-orange markings;
  - edge strength;
  - broad grass neighbourhood support;
  - a GLOBAL smooth-path optimiser that can cross long invisible/occluded gaps.

The .veo selected corners are used as anchors. camera2world is NOT used for the
four OUTER boundary traces.

A diagnostic halfway-line detector is also included. For that one only, the
current .veo geometry is used as a BROAD search corridor; the final cyan line
is snapped to image evidence.

Overlay
-------
RED      final inferred OUTER pitch boundary
GREEN    boundary samples supported by image evidence
MAGENTA  boundary samples inferred through weak/hidden evidence
YELLOW   .veo selected corner anchors
CYAN     diagnostic halfway line fitted to image evidence

Example
-------
python analysis/trace_pitch_boundaries_v2.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --stacked samples/match2_calibration_frame.jpg \
  --out-dir output/pitch_boundary_match2_v2
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
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


@dataclass
class BoundaryTrace:
    name: str
    lens: LensName
    anchor_names: list[str]
    anchors: np.ndarray
    path: np.ndarray
    evidence_along_path: np.ndarray
    inferred_mask: np.ndarray
    initial_direction: np.ndarray | None
    mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Trace actual white pitch boundaries from .veo corner anchors "
            "using adaptive image evidence and global smooth paths."
        )
    )

    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--stacked", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)

    p.add_argument(
        "--x-step",
        type=int,
        default=2,
        help="Horizontal DP sampling step in RAW pixels. Default: 2.",
    )
    p.add_argument(
        "--touchline-corridor-px",
        type=float,
        default=190.0,
        help="Soft corridor around initial touchline prediction. Default: 190.",
    )
    p.add_argument(
        "--goal-corridor-px",
        type=float,
        default=85.0,
        help="Soft corridor around corner-to-corner goal line. Default: 85.",
    )
    p.add_argument(
        "--halfway-corridor-px",
        type=int,
        default=95,
        help="Broad search corridor for halfway-line diagnostic. Default: 95.",
    )
    p.add_argument(
        "--max-dy-per-step",
        type=int,
        default=8,
        help="Maximum vertical change between DP columns. Default: 8.",
    )
    p.add_argument(
        "--smoothness",
        type=float,
        default=0.055,
        help="DP slope smoothness penalty. Default: 0.055.",
    )
    p.add_argument(
        "--corridor-penalty",
        type=float,
        default=1.6,
        help="Penalty for leaving the soft expected corridor. Default: 1.6.",
    )
    p.add_argument(
        "--evidence-threshold",
        type=float,
        default=0.28,
        help=(
            "Samples below this image-evidence score are displayed magenta "
            "as inferred/occluded. Default: 0.28."
        ),
    )
    p.add_argument(
        "--initial-lookahead-px",
        type=float,
        default=130.0,
        help="Lookahead for choosing touchline branch at a corner. Default: 130.",
    )
    p.add_argument(
        "--halfway",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable diagnostic halfway-line fitting.",
    )

    return p.parse_args()


def read_stacked(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    h, w = image.shape[:2]
    if (w, h) != (RAW_WIDTH, RAW_HEIGHT * 2):
        raise ValueError(
            f"Expected {RAW_WIDTH}x{RAW_HEIGHT * 2}, got {w}x{h}"
        )

    return (
        image,
        image[:RAW_HEIGHT].copy(),
        image[RAW_HEIGHT:].copy(),
    )


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RuntimeError("Cannot normalize zero vector")
    return v / n


def require_corner(
    calibration: VeoCalibration,
    corner: str,
    lens: LensName,
) -> np.ndarray:
    point = calibration.selected_corner_raw_pixel(corner, lens)
    if point is None:
        raise RuntimeError(
            f"Missing valid .veo corner {corner.upper()} for {lens.upper()}"
        )
    return np.asarray(point, dtype=np.float64)


def adaptive_line_evidence(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Return a float32 [0,1] likelihood for white painted football lines.

    Crucially, the line does NOT need to be globally bright. A grey line in
    shadow can still score well if it is:
      - locally brighter than nearby turf;
      - colour-neutral / low-saturation;
      - edge-supported;
      - surrounded by grass-like pixels.
    """
    image_f = image.astype(np.float32)

    b = image_f[:, :, 0]
    g = image_f[:, :, 1]
    r = image_f[:, :, 2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    value = hsv[:, :, 2].astype(np.float32) / 255.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    # Large local background estimate. White paint in shadow can still be
    # brighter than the nearby turf even when its absolute value is modest.
    local_bg = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=11.0,
        sigmaY=11.0,
    )

    local_contrast = np.clip(
        (luminance - local_bg + 0.025) / 0.22,
        0.0,
        1.0,
    )

    # Neutrality: football white/grey should have relatively similar channels.
    chroma = (
        np.maximum(np.maximum(b, g), r)
        - np.minimum(np.minimum(b, g), r)
    ) / 255.0

    neutral = np.clip(
        1.0 - chroma / 0.34,
        0.0,
        1.0,
    )

    low_sat = np.clip(
        1.0 - saturation / 0.55,
        0.0,
        1.0,
    )

    # A weak absolute-brightness term helps in sun, but is deliberately NOT a
    # hard gate so shaded grey lines remain usable.
    bright = np.clip(
        (value - 0.22) / 0.68,
        0.0,
        1.0,
    )

    # Edge support from line borders.
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    grad = cv2.magnitude(gx, gy)
    grad = np.clip(grad / 110.0, 0.0, 1.0)
    edge_support = cv2.GaussianBlur(
        grad,
        (0, 0),
        sigmaX=1.2,
        sigmaY=1.2,
    )

    # Broad turf cue, intentionally tolerant of deep shade.
    exg = (
        2.0 * g - r - b
    ) / 255.0
    raw_grass = np.clip(
        (exg + 0.03) / 0.34,
        0.0,
        1.0,
    )

    grass_support = cv2.GaussianBlur(
        raw_grass,
        (0, 0),
        sigmaX=10.0,
        sigmaY=10.0,
    )

    # Combine rather than threshold. This is the main V2 change.
    evidence = (
        0.34 * local_contrast
        + 0.22 * neutral
        + 0.16 * low_sat
        + 0.10 * bright
        + 0.18 * edge_support
    )

    # Suppress saturated auxiliary lines (yellow/orange) and obvious non-turf
    # structures while retaining shaded white paint.
    evidence *= np.clip(
        0.25 + 0.95 * grass_support,
        0.0,
        1.0,
    )
    evidence *= np.clip(
        1.25 - 1.15 * saturation,
        0.0,
        1.0,
    )

    # Local ridge enhancement helps thin distant painted lines.
    ridge = evidence - cv2.GaussianBlur(
        evidence,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
    )
    evidence = np.clip(
        evidence + 0.55 * np.maximum(ridge, 0.0),
        0.0,
        1.0,
    ).astype(np.float32)

    debug = {
        "local_contrast": local_contrast,
        "neutral": neutral,
        "edge_support": edge_support,
        "grass_support": grass_support,
    }

    return evidence, debug


def sample_evidence(
    evidence: np.ndarray,
    x: float,
    y: float,
) -> float:
    xi = int(round(x))
    yi = int(round(y))
    if not (0 <= xi < RAW_WIDTH and 0 <= yi < RAW_HEIGHT):
        return 0.0

    x0 = max(0, xi - 2)
    x1 = min(RAW_WIDTH, xi + 3)
    y0 = max(0, yi - 2)
    y1 = min(RAW_HEIGHT, yi + 3)

    return float(np.max(evidence[y0:y1, x0:x1]))


def ray_score(
    evidence: np.ndarray,
    seed: np.ndarray,
    direction: np.ndarray,
    lookahead_px: float,
) -> float:
    direction = unit(direction)
    perp = np.asarray([-direction[1], direction[0]])

    values: list[float] = []

    for d in np.arange(6.0, lookahead_px + 1.0, 3.0):
        p = seed + d * direction

        if not (
            2 <= p[0] < RAW_WIDTH - 2
            and 2 <= p[1] < RAW_HEIGHT - 2
        ):
            break

        best = 0.0
        for off in range(-4, 5):
            q = p + off * perp
            best = max(
                best,
                sample_evidence(
                    evidence,
                    q[0],
                    q[1],
                ),
            )
        values.append(best)

    if not values:
        return -1e9

    a = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(a))
        + 0.65 * float(np.mean(a > 0.25))
        + 0.30 * float(np.mean(a > 0.45))
    )


def estimate_touchline_direction(
    evidence: np.ndarray,
    seed: np.ndarray,
    goal_direction: np.ndarray,
    horizontal_sign: float,
    lookahead_px: float,
) -> np.ndarray:
    goal = unit(goal_direction)

    best_score = -1e18
    best: np.ndarray | None = None

    for angle_deg in np.linspace(-82.0, 82.0, 329):
        a = math.radians(float(angle_deg))
        d = np.asarray(
            [
                horizontal_sign * math.cos(a),
                math.sin(a),
            ],
            dtype=np.float64,
        )
        d = unit(d)

        progress = horizontal_sign * d[0]
        if progress < 0.18:
            continue

        goal_alignment = abs(float(np.dot(d, goal)))
        if goal_alignment > 0.91:
            continue

        score = (
            ray_score(
                evidence,
                seed,
                d,
                lookahead_px,
            )
            + 0.10 * progress
            - 0.25 * goal_alignment
        )

        if score > best_score:
            best_score = score
            best = d

    if best is None:
        raise RuntimeError("Could not estimate touchline direction")

    return best


def baseline_from_ray(
    seed: np.ndarray,
    direction: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    direction = unit(direction)

    if abs(direction[0]) < 1e-8:
        return np.full(xs.shape, seed[1], dtype=np.float64)

    slope = direction[1] / direction[0]
    return (
        seed[1]
        + slope * (xs - seed[0])
    )


def baseline_between(
    a: np.ndarray,
    b: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    if abs(b[0] - a[0]) < 1e-8:
        return np.full(xs.shape, 0.5 * (a[1] + b[1]))

    t = (xs - a[0]) / (b[0] - a[0])
    return a[1] + t * (b[1] - a[1])


def global_smooth_path(
    evidence: np.ndarray,
    xs: np.ndarray,
    baseline_y: np.ndarray,
    *,
    start_y: float,
    end_y: float | None,
    corridor_px: float,
    max_dy: int,
    smoothness: float,
    corridor_penalty: float,
) -> np.ndarray:
    """
    Global dynamic-programming path.

    Unlike V1, the path does not stop when the line disappears. A hidden
    section simply contributes weak image evidence; smoothness and the visible
    A/C sections decide where B should pass.
    """
    h = evidence.shape[0]
    n = xs.size

    if n < 2:
        raise RuntimeError("Too few DP columns")

    inf = np.float32(1e9)

    prev = np.full(h, inf, dtype=np.float32)
    back = np.zeros((n, h), dtype=np.int8)

    y0 = int(round(start_y))
    for y in range(max(0, y0 - 2), min(h, y0 + 3)):
        prev[y] = np.float32(abs(y - start_y) * 0.25)

    ys_all = np.arange(h, dtype=np.float32)

    for i in range(1, n):
        x = int(np.clip(round(float(xs[i])), 0, RAW_WIDTH - 1))

        ev = evidence[:, x].astype(np.float32)

        distance_from_prior = np.abs(
            ys_all - np.float32(baseline_y[i])
        )
        soft = np.maximum(
            distance_from_prior - corridor_px,
            0.0,
        ) / max(corridor_px, 1.0)

        # Low evidence is allowed, not fatal. This is what bridges occlusions.
        local_cost = (
            1.0 - ev
            + corridor_penalty * soft * soft
            + 0.002 * distance_from_prior
        ).astype(np.float32)

        expected_dy = (
            baseline_y[i] - baseline_y[i - 1]
        )

        best_cost = np.full(h, inf, dtype=np.float32)
        best_shift = np.zeros(h, dtype=np.int8)

        for dy in range(-max_dy, max_dy + 1):
            shifted = np.full(h, inf, dtype=np.float32)

            if dy < 0:
                shifted[:dy] = prev[-dy:]
            elif dy > 0:
                shifted[dy:] = prev[:-dy]
            else:
                shifted[:] = prev

            transition = (
                smoothness
                * abs(float(dy) - float(expected_dy))
            )

            candidate = shifted + np.float32(transition)
            better = candidate < best_cost

            best_cost[better] = candidate[better]
            best_shift[better] = np.int8(dy)

        curr = best_cost + local_cost
        back[i] = best_shift
        prev = curr

    if end_y is None:
        final_prior = baseline_y[-1]
        endpoint_penalty = (
            np.abs(ys_all - final_prior)
            / max(corridor_px, 1.0)
        ) * 0.04
        y = int(np.argmin(prev + endpoint_penalty))
    else:
        target = int(round(end_y))
        lo = max(0, target - 5)
        hi = min(h, target + 6)
        y = lo + int(np.argmin(prev[lo:hi]))

    path_y = np.zeros(n, dtype=np.float64)
    path_y[-1] = float(y)

    for i in range(n - 1, 0, -1):
        dy = int(back[i, y])
        y = y - dy
        y = int(np.clip(y, 0, h - 1))
        path_y[i - 1] = float(y)

    return np.column_stack(
        [xs.astype(np.float64), path_y]
    )


def path_evidence(
    evidence: np.ndarray,
    path: np.ndarray,
) -> np.ndarray:
    values = np.zeros(path.shape[0], dtype=np.float64)

    for i, (x, y) in enumerate(path):
        values[i] = sample_evidence(evidence, x, y)

    return values


def trace_goal_line(
    name: str,
    lens: LensName,
    a_name: str,
    b_name: str,
    a: np.ndarray,
    b: np.ndarray,
    evidence: np.ndarray,
    args: argparse.Namespace,
) -> BoundaryTrace:
    # DP is x-monotonic. Reverse endpoints if needed.
    if a[0] <= b[0]:
        start, end = a, b
        start_name, end_name = a_name, b_name
    else:
        start, end = b, a
        start_name, end_name = b_name, a_name

    xs = np.arange(
        int(round(start[0])),
        int(round(end[0])) + 1,
        args.x_step,
        dtype=np.float64,
    )

    if xs[-1] != int(round(end[0])):
        xs = np.append(xs, float(round(end[0])))

    baseline = baseline_between(start, end, xs)

    path = global_smooth_path(
        evidence,
        xs,
        baseline,
        start_y=start[1],
        end_y=end[1],
        corridor_px=args.goal_corridor_px,
        max_dy=args.max_dy_per_step,
        smoothness=args.smoothness,
        corridor_penalty=args.corridor_penalty,
    )

    ev = path_evidence(evidence, path)

    return BoundaryTrace(
        name=name,
        lens=lens,
        anchor_names=[start_name, end_name],
        anchors=np.vstack([start, end]),
        path=path,
        evidence_along_path=ev,
        inferred_mask=ev < args.evidence_threshold,
        initial_direction=unit(end - start),
        mode="global_dp_two_anchors",
    )


def trace_touchline(
    name: str,
    lens: LensName,
    anchor_name: str,
    seed: np.ndarray,
    goal_direction: np.ndarray,
    horizontal_sign: float,
    evidence: np.ndarray,
    args: argparse.Namespace,
) -> BoundaryTrace:
    direction = estimate_touchline_direction(
        evidence,
        seed,
        goal_direction,
        horizontal_sign,
        args.initial_lookahead_px,
    )

    if horizontal_sign > 0:
        xs = np.arange(
            int(round(seed[0])),
            RAW_WIDTH,
            args.x_step,
            dtype=np.float64,
        )
    else:
        xs = np.arange(
            int(round(seed[0])),
            -1,
            -args.x_step,
            dtype=np.float64,
        )

    baseline = baseline_from_ray(seed, direction, xs)

    path = global_smooth_path(
        evidence,
        xs,
        baseline,
        start_y=seed[1],
        end_y=None,
        corridor_px=args.touchline_corridor_px,
        max_dy=args.max_dy_per_step,
        smoothness=args.smoothness,
        corridor_penalty=args.corridor_penalty,
    )

    ev = path_evidence(evidence, path)

    return BoundaryTrace(
        name=name,
        lens=lens,
        anchor_names=[anchor_name],
        anchors=np.asarray([seed], dtype=np.float64),
        path=path,
        evidence_along_path=ev,
        inferred_mask=ev < args.evidence_threshold,
        initial_direction=direction,
        mode="global_dp_single_anchor",
    )


def project_halfway_prior(
    calibration: VeoCalibration,
    lens: LensName,
) -> np.ndarray:
    points: list[np.ndarray] = []

    for z in np.linspace(
        -calibration.half_width,
        calibration.half_width,
        401,
    ):
        try:
            p = calibration.field_to_pixel(
                lens,
                np.asarray([0.0, z], dtype=np.float64),
            )
        except Exception:
            continue

        if (
            -120.0 <= p[0] <= RAW_WIDTH + 120.0
            and -120.0 <= p[1] <= RAW_HEIGHT + 120.0
        ):
            points.append(np.asarray(p, dtype=np.float64))

    if len(points) < 8:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def snap_halfway_line(
    evidence: np.ndarray,
    prior: np.ndarray,
    corridor_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagnostic only.

    The .veo halfway projection defines a broad search corridor; actual image
    evidence chooses the fitted cyan curve.
    """
    if prior.shape[0] < 8:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    dx = float(np.ptp(prior[:, 0]))
    dy = float(np.ptp(prior[:, 1]))

    samples: list[list[float]] = []
    weights: list[float] = []

    if dy >= dx:
        # Near-vertical line: scan rows.
        order = np.argsort(prior[:, 1])
        p = prior[order]
        y_min = max(0, int(math.floor(np.min(p[:, 1]))))
        y_max = min(RAW_HEIGHT - 1, int(math.ceil(np.max(p[:, 1]))))

        for y in range(y_min, y_max + 1, 2):
            x_prior = float(np.interp(y, p[:, 1], p[:, 0]))
            x0 = max(0, int(round(x_prior)) - corridor_px)
            x1 = min(RAW_WIDTH, int(round(x_prior)) + corridor_px + 1)

            if x1 <= x0:
                continue

            row = evidence[y, x0:x1]
            x_rel = int(np.argmax(row))
            score = float(row[x_rel])

            if score < 0.14:
                continue

            x = x0 + x_rel
            samples.append([float(x), float(y)])
            weights.append(score)

        if len(samples) < 8:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        pts = np.asarray(samples, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)

        # Robust x=f(y), enough for lens distortion over the visible segment.
        keep = np.ones(pts.shape[0], dtype=bool)

        for _ in range(5):
            coeff = np.polyfit(
                pts[keep, 1],
                pts[keep, 0],
                deg=min(3, np.count_nonzero(keep) - 1),
                w=np.maximum(w[keep], 0.05),
            )
            pred = np.polyval(coeff, pts[:, 1])
            resid = np.abs(pts[:, 0] - pred)
            med = float(np.median(resid[keep]))
            keep = resid <= max(7.0, med * 2.8 + 2.0)

        ys = np.arange(
            int(np.min(pts[keep, 1])),
            int(np.max(pts[keep, 1])) + 1,
            1,
            dtype=np.float64,
        )
        xs = np.polyval(coeff, ys)
        curve = np.column_stack([xs, ys])

    else:
        # Near-horizontal line: scan columns.
        order = np.argsort(prior[:, 0])
        p = prior[order]
        x_min = max(0, int(math.floor(np.min(p[:, 0]))))
        x_max = min(RAW_WIDTH - 1, int(math.ceil(np.max(p[:, 0]))))

        for x in range(x_min, x_max + 1, 2):
            y_prior = float(np.interp(x, p[:, 0], p[:, 1]))
            y0 = max(0, int(round(y_prior)) - corridor_px)
            y1 = min(RAW_HEIGHT, int(round(y_prior)) + corridor_px + 1)

            if y1 <= y0:
                continue

            col = evidence[y0:y1, x]
            y_rel = int(np.argmax(col))
            score = float(col[y_rel])

            if score < 0.14:
                continue

            y = y0 + y_rel
            samples.append([float(x), float(y)])
            weights.append(score)

        if len(samples) < 8:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        pts = np.asarray(samples, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        keep = np.ones(pts.shape[0], dtype=bool)

        for _ in range(5):
            coeff = np.polyfit(
                pts[keep, 0],
                pts[keep, 1],
                deg=min(3, np.count_nonzero(keep) - 1),
                w=np.maximum(w[keep], 0.05),
            )
            pred = np.polyval(coeff, pts[:, 0])
            resid = np.abs(pts[:, 1] - pred)
            med = float(np.median(resid[keep]))
            keep = resid <= max(7.0, med * 2.8 + 2.0)

        xs = np.arange(
            int(np.min(pts[keep, 0])),
            int(np.max(pts[keep, 0])) + 1,
            1,
            dtype=np.float64,
        )
        ys = np.polyval(coeff, xs)
        curve = np.column_stack([xs, ys])

    valid = (
        (curve[:, 0] >= 0)
        & (curve[:, 0] < RAW_WIDTH)
        & (curve[:, 1] >= 0)
        & (curve[:, 1] < RAW_HEIGHT)
    )

    return curve[valid], pts[keep]


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if points.shape[0] < 2:
        return

    p = np.round(points).astype(np.int32)
    p[:, 0] = np.clip(p[:, 0], 0, RAW_WIDTH - 1)
    p[:, 1] = np.clip(p[:, 1], 0, RAW_HEIGHT - 1)

    cv2.polylines(
        image,
        [p.reshape(-1, 1, 2)],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_boundary(
    image: np.ndarray,
    trace: BoundaryTrace,
) -> None:
    draw_polyline(
        image,
        trace.path,
        RED,
        3,
    )

    # Show where the red curve is actually supported vs inferred.
    for i in range(0, trace.path.shape[0], 5):
        p = tuple(np.round(trace.path[i]).astype(int))

        color = (
            MAGENTA
            if bool(trace.inferred_mask[i])
            else GREEN
        )

        cv2.circle(
            image,
            p,
            2,
            color,
            -1,
            cv2.LINE_AA,
        )

    for name, anchor in zip(
        trace.anchor_names,
        trace.anchors,
    ):
        p = tuple(np.round(anchor).astype(int))

        cv2.circle(
            image,
            p,
            7,
            YELLOW,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            name.upper(),
            (p[0] + 8, max(18, p[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            YELLOW,
            1,
            cv2.LINE_AA,
        )


def normalized_debug_image(arr: np.ndarray) -> np.ndarray:
    vis = np.clip(arr, 0.0, 1.0)
    return np.round(vis * 255.0).astype(np.uint8)


def serialize_trace(trace: BoundaryTrace) -> dict[str, Any]:
    return {
        "name": trace.name,
        "lens": trace.lens,
        "anchor_names": trace.anchor_names,
        "anchors_raw": trace.anchors.tolist(),
        "mode": trace.mode,
        "initial_direction": (
            trace.initial_direction.tolist()
            if trace.initial_direction is not None
            else None
        ),
        "path_points": int(trace.path.shape[0]),
        "image_supported_points": int(
            np.count_nonzero(~trace.inferred_mask)
        ),
        "inferred_points": int(
            np.count_nonzero(trace.inferred_mask)
        ),
        "supported_fraction": float(
            np.mean(~trace.inferred_mask)
        ),
        "path_raw": trace.path.tolist(),
        "evidence_along_path": (
            trace.evidence_along_path.tolist()
        ),
    }


def main() -> None:
    args = parse_args()

    if args.x_step < 1:
        raise ValueError("--x-step must be >= 1")

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

    left_ev, left_debug = adaptive_line_evidence(left)
    right_ev, right_debug = adaptive_line_evidence(right)

    cl = require_corner(calibration, "cl", "left")
    fl = require_corner(calibration, "fl", "left")
    fr = require_corner(calibration, "fr", "right")
    cr = require_corner(calibration, "cr", "right")

    traces = [
        trace_goal_line(
            "LEFT_GOAL_LINE",
            "left",
            "cl",
            "fl",
            cl,
            fl,
            left_ev,
            args,
        ),
        trace_goal_line(
            "RIGHT_GOAL_LINE",
            "right",
            "fr",
            "cr",
            fr,
            cr,
            right_ev,
            args,
        ),
        trace_touchline(
            "CLOSE_TOUCHLINE_LEFT",
            "left",
            "cl",
            cl,
            fl - cl,
            +1.0,
            left_ev,
            args,
        ),
        trace_touchline(
            "FAR_TOUCHLINE_LEFT",
            "left",
            "fl",
            fl,
            cl - fl,
            +1.0,
            left_ev,
            args,
        ),
        trace_touchline(
            "CLOSE_TOUCHLINE_RIGHT",
            "right",
            "cr",
            cr,
            fr - cr,
            -1.0,
            right_ev,
            args,
        ),
        trace_touchline(
            "FAR_TOUCHLINE_RIGHT",
            "right",
            "fr",
            fr,
            cr - fr,
            -1.0,
            right_ev,
            args,
        ),
    ]

    left_overlay = left.copy()
    right_overlay = right.copy()

    for trace in traces:
        draw_boundary(
            left_overlay if trace.lens == "left" else right_overlay,
            trace,
        )

    halfway_report: dict[str, Any] = {}

    if args.halfway:
        for lens, image, ev in (
            ("left", left_overlay, left_ev),
            ("right", right_overlay, right_ev),
        ):
            prior = project_halfway_prior(
                calibration,
                lens,
            )

            curve, observed = snap_halfway_line(
                ev,
                prior,
                args.halfway_corridor_px,
            )

            if curve.shape[0] >= 2:
                draw_polyline(
                    image,
                    curve,
                    CYAN,
                    3,
                )

                for p in observed[::4]:
                    cv2.circle(
                        image,
                        tuple(np.round(p).astype(int)),
                        2,
                        GREEN,
                        -1,
                        cv2.LINE_AA,
                    )

            halfway_report[lens] = {
                "prior_points": int(prior.shape[0]),
                "curve_points": int(curve.shape[0]),
                "observed_points": int(observed.shape[0]),
                "curve_raw": curve.tolist(),
                "observed_raw": observed.tolist(),
            }

    stacked_overlay = np.vstack(
        [left_overlay, right_overlay]
    )

    outputs = {
        "left_overlay": args.out_dir / "left_boundary_overlay_v2.jpg",
        "right_overlay": args.out_dir / "right_boundary_overlay_v2.jpg",
        "stacked_overlay": args.out_dir / "stacked_boundary_overlay_v2.jpg",
        "left_evidence": args.out_dir / "left_adaptive_evidence.png",
        "right_evidence": args.out_dir / "right_adaptive_evidence.png",
        "report": args.out_dir / "boundary_trace_v2.json",
    }

    cv2.imwrite(
        str(outputs["left_overlay"]),
        left_overlay,
    )
    cv2.imwrite(
        str(outputs["right_overlay"]),
        right_overlay,
    )
    cv2.imwrite(
        str(outputs["stacked_overlay"]),
        stacked_overlay,
    )
    cv2.imwrite(
        str(outputs["left_evidence"]),
        normalized_debug_image(left_ev),
    )
    cv2.imwrite(
        str(outputs["right_evidence"]),
        normalized_debug_image(right_ev),
    )

    payload = {
        "calibration": str(args.calibration),
        "stacked": str(args.stacked),
        "purpose": (
            "Image-driven white pitch-boundary tracing from .veo corner "
            "anchors, with adaptive shadow handling and global gap bridging."
        ),
        "legend": {
            "red": "final inferred outer boundary",
            "green": "image-supported samples",
            "magenta": "weak/hidden samples inferred by global path",
            "yellow": ".veo selected corner anchors",
            "cyan": "diagnostic halfway line snapped to image evidence",
        },
        "parameters": {
            "x_step": args.x_step,
            "touchline_corridor_px": args.touchline_corridor_px,
            "goal_corridor_px": args.goal_corridor_px,
            "halfway_corridor_px": args.halfway_corridor_px,
            "max_dy_per_step": args.max_dy_per_step,
            "smoothness": args.smoothness,
            "corridor_penalty": args.corridor_penalty,
            "evidence_threshold": args.evidence_threshold,
            "initial_lookahead_px": args.initial_lookahead_px,
        },
        "traces": [
            serialize_trace(t)
            for t in traces
        ],
        "halfway": halfway_report,
    }

    outputs["report"].write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 96)
    print("Veo pitch-boundary tracer V2")
    print("=" * 96)

    for t in traces:
        print(
            f"{t.name:<25} "
            f"{t.lens.upper():<5} "
            f"path={t.path.shape[0]:>4d} "
            f"supported={np.mean(~t.inferred_mask):6.1%} "
            f"inferred={np.mean(t.inferred_mask):6.1%}"
        )

    if args.halfway:
        print()
        for lens in ("left", "right"):
            h = halfway_report[lens]
            print(
                f"HALFWAY {lens.upper():<5} "
                f"observed={h['observed_points']:>4d} "
                f"curve={h['curve_points']:>4d}"
            )

    print()
    print("Written:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
