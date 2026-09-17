#!/usr/bin/env python3
"""
Canonical Veo geometry for VeoTranscodeLab.

This module centralizes the geometry that was previously duplicated across
diagnostic scripts.

Scope
-----
- Load a Veo `.veo` calibration file.
- Represent LEFT/RIGHT lens calibration.
- Convert RAW transcode pixels (2048x1024 per lens) to field coordinates.
- Project field/world points back to RAW transcode pixels.
- Expose field dimensions and selected Veo corners for validation.

Important
---------
This third version preserves the same Veo geometry and uses a portable
Newton refinement for inverse distortion, because some OpenCV Python builds
do not expose `undistortPointsIter`. It does NOT:
- fit or optimize calibration;
- add manual correction offsets;
- estimate an `.oev`;
- alter Veo's camera2world;
- change the inferred 3840x1536 calibration-coordinate interpretation.

Coordinate convention
---------------------
World/field:
    X = field length axis
    Y = vertical axis
    Z = field width axis
    ground plane = Y = 0

RAW lens image:
    width  = 2048
    height = 1024

Effective Veo calibration coordinates:
    width  = 3840
    height = 1536
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


LensName = Literal["left", "right"]

RAW_WIDTH = 2048
RAW_HEIGHT = 1024

CALIBRATION_WIDTH = 3840
CALIBRATION_HEIGHT = 1536

SCALE_X = RAW_WIDTH / CALIBRATION_WIDTH
SCALE_Y = RAW_HEIGHT / CALIBRATION_HEIGHT


def parse_matrix(value: str, rows: int, cols: int) -> np.ndarray:
    values = np.asarray(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    )
    expected = rows * cols
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} values for a {rows}x{cols} matrix, "
            f"got {values.size}"
        )
    return values.reshape(rows, cols)


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    )


def _project_normalized_point(
    normalized_xy: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """
    Apply OpenCV's forward distortion model to one normalized point and
    return calibration-image pixel coordinates.

    Using cv2.projectPoints here guarantees that the inverse refinement below
    targets the exact same distortion model used by world_to_pixel().
    """
    x, y = np.asarray(normalized_xy, dtype=np.float64).reshape(2)

    object_point = np.asarray(
        [[[x, y, 1.0]]],
        dtype=np.float64,
    )

    image_point, _ = cv2.projectPoints(
        object_point,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        intrinsic,
        distortion,
    )

    return image_point.reshape(2)


def _undistort_point_precise(
    distorted_pixel: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    *,
    max_iterations: int = 12,
    tolerance_px: float = 1e-10,
    jacobian_step: float = 1e-7,
) -> np.ndarray:
    """
    Numerically invert OpenCV's distortion for one pixel.

    Some OpenCV Python builds do not expose `undistortPointsIter`, while the
    default `undistortPoints` iteration budget can leave a measurable residual
    with Veo's strong distortion.

    Strategy:
    1. use cv2.undistortPoints() as a good initial guess;
    2. forward-project that normalized point with cv2.projectPoints();
    3. refine the normalized point with a 2D Newton solve until the projected
       calibration pixel matches the original distorted pixel.

    This keeps the inverse mathematically tied to the exact OpenCV forward
    model and avoids depending on an optional Python binding.
    """
    target = np.asarray(
        distorted_pixel,
        dtype=np.float64,
    ).reshape(2)

    initial = cv2.undistortPoints(
        target.reshape(1, 1, 2),
        intrinsic,
        distortion,
    ).reshape(2)

    xy = initial.astype(np.float64, copy=True)

    for _ in range(max_iterations):
        projected = _project_normalized_point(
            xy,
            intrinsic,
            distortion,
        )

        residual = projected - target

        if float(np.linalg.norm(residual)) <= tolerance_px:
            return xy

        # Numerical Jacobian d(pixel)/d(normalized_xy).
        jacobian = np.empty((2, 2), dtype=np.float64)

        for axis in range(2):
            shifted = xy.copy()
            shifted[axis] += jacobian_step

            projected_shifted = _project_normalized_point(
                shifted,
                intrinsic,
                distortion,
            )

            jacobian[:, axis] = (
                projected_shifted - projected
            ) / jacobian_step

        try:
            delta = np.linalg.solve(
                jacobian,
                residual,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Could not invert distortion: singular numerical Jacobian"
            ) from exc

        xy -= delta

        if not np.all(np.isfinite(xy)):
            raise RuntimeError(
                "Could not invert distortion: non-finite Newton iterate"
            )

    final_projected = _project_normalized_point(
        xy,
        intrinsic,
        distortion,
    )
    final_error = float(
        np.linalg.norm(final_projected - target)
    )

    if final_error > 1e-6:
        raise RuntimeError(
            "Could not invert distortion accurately: "
            f"final residual={final_error:.6g} px"
        )

    return xy


@dataclass(frozen=True)
class LensCalibration:
    name: LensName
    intrinsic: np.ndarray
    distortion: np.ndarray
    lens2camera: np.ndarray


@dataclass
class VeoCalibration:
    source_path: Path
    raw_json: dict

    left: LensCalibration
    right: LensCalibration

    camera2world: np.ndarray
    field_length: float
    field_width: float

    calibration_version: str | None = None
    worldmap_version: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "VeoCalibration":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        alignment = data["alignment"]

        left = LensCalibration(
            name="left",
            intrinsic=parse_matrix(
                alignment["intrinsic_left"],
                3,
                3,
            ),
            distortion=parse_vector(
                alignment["distortion_left"],
            ),
            lens2camera=parse_matrix(
                alignment["lens_left2camera"],
                4,
                4,
            ),
        )

        right = LensCalibration(
            name="right",
            intrinsic=parse_matrix(
                alignment["intrinsic_right"],
                3,
                3,
            ),
            distortion=parse_vector(
                alignment["distortion_right"],
            ),
            lens2camera=parse_matrix(
                alignment["lens_right2camera"],
                4,
                4,
            ),
        )

        return cls(
            source_path=path,
            raw_json=data,
            left=left,
            right=right,
            camera2world=parse_matrix(
                alignment["camera2world"],
                4,
                4,
            ),
            field_length=float(alignment["field_length"]),
            field_width=float(alignment["field_width"]),
            calibration_version=data.get("calibration_version"),
            worldmap_version=data.get("worldmap_version"),
        )

    @property
    def half_length(self) -> float:
        return self.field_length / 2.0

    @property
    def half_width(self) -> float:
        return self.field_width / 2.0

    def lens(self, lens_name: LensName) -> LensCalibration:
        if lens_name == "left":
            return self.left
        if lens_name == "right":
            return self.right
        raise ValueError(f"Unknown lens: {lens_name!r}")

    # ------------------------------------------------------------------
    # Coordinate-space conversion
    # ------------------------------------------------------------------

    @staticmethod
    def raw_to_calibration_pixel(
        pixel: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        Convert one RAW 2048x1024 lens pixel into Veo calibration coordinates.
        """
        u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
        return np.asarray(
            [
                u / SCALE_X,
                v / SCALE_Y,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def calibration_to_raw_pixel(
        pixel: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        Convert one Veo calibration-coordinate pixel into RAW 2048x1024.
        """
        u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
        return np.asarray(
            [
                u * SCALE_X,
                v * SCALE_Y,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # RAW pixel -> world / field
    # ------------------------------------------------------------------

    def pixel_to_world_ground(
        self,
        lens_name: LensName,
        pixel: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        RAW lens pixel -> intersection with the Veo ground plane Y=0.

        Returns:
            np.ndarray [X, Y, Z]

        This intentionally preserves the existing project implementation:
        RAW -> calibration coordinates -> cv2.undistortPoints -> lens ray
        -> lens2world -> Y=0 intersection.
        """
        lens = self.lens(lens_name)

        u_cal, v_cal = self.raw_to_calibration_pixel(pixel)

        distorted = np.asarray(
            [[[u_cal, v_cal]]],
            dtype=np.float64,
        )

        normalized = _undistort_point_precise(
            np.asarray([u_cal, v_cal], dtype=np.float64),
            lens.intrinsic,
            lens.distortion,
        )

        ray_lens = np.asarray(
            [
                normalized[0],
                normalized[1],
                1.0,
            ],
            dtype=np.float64,
        )

        lens2world = self.camera2world @ lens.lens2camera

        ray_origin = lens2world[:3, 3]
        ray_direction = lens2world[:3, :3] @ ray_lens

        if abs(ray_direction[1]) < 1e-12:
            raise RuntimeError(
                f"{lens_name}: ray is parallel to ground plane"
            )

        t = -ray_origin[1] / ray_direction[1]

        if t <= 0:
            raise RuntimeError(
                f"{lens_name}: ground intersection lies behind the camera"
            )

        world = ray_origin + t * ray_direction
        return world

    def pixel_to_field(
        self,
        lens_name: LensName,
        pixel: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        RAW lens pixel -> field coordinate [X, Z] in meters.
        """
        world = self.pixel_to_world_ground(lens_name, pixel)
        return world[[0, 2]]

    # ------------------------------------------------------------------
    # World / field -> RAW pixel
    # ------------------------------------------------------------------

    def world_to_pixel(
        self,
        lens_name: LensName,
        world_point: tuple[float, float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        World [X,Y,Z] -> RAW 2048x1024 lens pixel.

        Uses the exact forward model already present in the original
        test_field_projection.py:
            world -> camera -> lens -> cv2.projectPoints -> RAW scaling
        """
        lens = self.lens(lens_name)

        world_point = np.asarray(
            world_point,
            dtype=np.float64,
        ).reshape(3)

        world_h = np.asarray(
            [
                world_point[0],
                world_point[1],
                world_point[2],
                1.0,
            ],
            dtype=np.float64,
        )

        world2camera = np.linalg.inv(self.camera2world)
        camera2lens = np.linalg.inv(lens.lens2camera)

        lens_h = camera2lens @ world2camera @ world_h
        lens_xyz = lens_h[:3]

        if lens_xyz[2] <= 0:
            raise RuntimeError(
                f"{lens_name}: point is behind lens: "
                f"{world_point.tolist()} -> {lens_xyz.tolist()}"
            )

        object_point = lens_xyz.reshape(1, 1, 3)

        image_point, _ = cv2.projectPoints(
            object_point,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            lens.intrinsic,
            lens.distortion,
        )

        calibration_pixel = image_point.reshape(2)

        return self.calibration_to_raw_pixel(
            calibration_pixel,
        )

    def field_to_pixel(
        self,
        lens_name: LensName,
        field_point: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        Field [X,Z] on Y=0 -> RAW 2048x1024 lens pixel.
        """
        x, z = np.asarray(
            field_point,
            dtype=np.float64,
        ).reshape(2)

        return self.world_to_pixel(
            lens_name,
            np.asarray([x, 0.0, z], dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Field / validation helpers
    # ------------------------------------------------------------------

    def is_inside_field(
        self,
        field_point: tuple[float, float] | np.ndarray,
        margin_m: float = 0.0,
    ) -> bool:
        x, z = np.asarray(
            field_point,
            dtype=np.float64,
        ).reshape(2)

        return (
            -self.half_length - margin_m
            <= x
            <= self.half_length + margin_m
            and
            -self.half_width - margin_m
            <= z
            <= self.half_width + margin_m
        )

    def field_corners(self) -> dict[str, np.ndarray]:
        """
        Field corners using the coordinate convention already inferred
        by the existing diagnostics.

        C = close side (negative Z)
        F = far side   (positive Z)
        L = left goal line  (negative X)
        R = right goal line (positive X)
        """
        return {
            "cl": np.asarray(
                [-self.half_length, -self.half_width],
                dtype=np.float64,
            ),
            "fl": np.asarray(
                [-self.half_length, +self.half_width],
                dtype=np.float64,
            ),
            "fr": np.asarray(
                [+self.half_length, +self.half_width],
                dtype=np.float64,
            ),
            "cr": np.asarray(
                [+self.half_length, -self.half_width],
                dtype=np.float64,
            ),
        }

    def selected_corner_raw_pixel(
        self,
        corner_name: str,
        lens_name: LensName,
    ) -> np.ndarray | None:
        """
        Return Veo's selected corner as a local RAW lens pixel.

        Returns None if that corner/lens pair is marked invalid.
        """
        selected = (
            self.raw_json
            .get("selected_corners", {})
            .get("corners_stacked", {})
        )

        corner = selected.get(corner_name)
        if not corner:
            return None

        entry = corner.get(lens_name)
        if not entry or not entry.get("valid"):
            return None

        nx, ny = entry["stacked"]

        x = float(nx) * 2048.0
        y_stacked = float(ny) * 2048.0

        if lens_name == "left":
            y = y_stacked
        else:
            y = y_stacked - 1024.0

        return np.asarray([x, y], dtype=np.float64)

    def selected_corner_observations(
        self,
    ) -> list[dict]:
        """
        Return the valid Veo selected corners as reusable validation records.
        """
        observations: list[dict] = []
        corners = self.field_corners()

        for corner_name, field_point in corners.items():
            for lens_name in ("left", "right"):
                raw_pixel = self.selected_corner_raw_pixel(
                    corner_name,
                    lens_name,
                )
                if raw_pixel is None:
                    continue

                observations.append(
                    {
                        "corner": corner_name,
                        "lens": lens_name,
                        "raw_pixel": raw_pixel,
                        "expected_field": field_point.copy(),
                    }
                )

        return observations


def load_veo_calibration(
    path: str | Path,
) -> VeoCalibration:
    """
    Convenience wrapper for callers that prefer a function-style API.
    """
    return VeoCalibration.from_file(path)


__all__ = [
    "RAW_WIDTH",
    "RAW_HEIGHT",
    "CALIBRATION_WIDTH",
    "CALIBRATION_HEIGHT",
    "SCALE_X",
    "SCALE_Y",
    "LensCalibration",
    "VeoCalibration",
    "load_veo_calibration",
]
