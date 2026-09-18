#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import load_veo_metadata


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stacked", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)

    parser.add_argument(
        "--lens",
        choices=("left", "right"),
        required=True,
    )

    parser.add_argument(
        "--mapping",
        choices=(
            "4k-center-crop",
            "uniform-1920",
            "legacy-1536",
        ),
        default="4k-center-crop",
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Offset horizontal desde el eje de la lente, en grados.",
    )

    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Offset vertical desde el eje de la lente, en grados. Positivo = arriba.",
    )

    parser.add_argument(
        "--hfov",
        type=float,
        default=100.0,
        help="FOV horizontal de la cámara virtual rectilínea.",
    )

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/veo_virtual_camera"),
    )

    return parser.parse_args()


def split_stacked(image):
    h, w = image.shape[:2]

    if h % 2:
        raise ValueError("Stacked image height must be even")

    hh = h // 2

    return {
        "left": image[:hh].copy(),
        "right": image[hh:].copy(),
    }


def build_raw_intrinsic(
    K_veo: np.ndarray,
    raw_width: int,
    raw_height: int,
    mapping: str,
) -> tuple[np.ndarray, dict]:

    K = K_veo.astype(np.float64).copy()

    if mapping == "4k-center-crop":
        # Hypothesis:
        #
        # calibrated lens image: 3840x2160
        #              ↓
        # centered vertical crop: 3840x1920
        #              ↓
        # downloaded RAW lens: 2048x1024
        #
        source_width = 3840.0
        source_height = 2160.0

        crop_width = 3840.0
        crop_height = 1920.0

        crop_x = (source_width - crop_width) / 2.0
        crop_y = (source_height - crop_height) / 2.0

        sx = raw_width / crop_width
        sy = raw_height / crop_height

        K[0, 0] *= sx
        K[1, 1] *= sy

        K[0, 2] = (K_veo[0, 2] - crop_x) * sx
        K[1, 2] = (K_veo[1, 2] - crop_y) * sy

        info = {
            "source_size": [3840, 2160],
            "crop": [crop_x, crop_y, crop_width, crop_height],
            "scale": [sx, sy],
        }

    elif mapping == "uniform-1920":
        source_width = 3840.0
        source_height = 1920.0

        sx = raw_width / source_width
        sy = raw_height / source_height

        K[0, 0] *= sx
        K[1, 1] *= sy

        K[0, 2] *= sx
        K[1, 2] *= sy

        info = {
            "source_size": [3840, 1920],
            "crop": None,
            "scale": [sx, sy],
        }

    elif mapping == "legacy-1536":
        source_width = 3840.0
        source_height = 1536.0

        sx = raw_width / source_width
        sy = raw_height / source_height

        K[0, 0] *= sx
        K[1, 1] *= sy

        K[0, 2] *= sx
        K[1, 2] *= sy

        info = {
            "source_size": [3840, 1536],
            "crop": None,
            "scale": [sx, sy],
        }

    else:
        raise ValueError(mapping)

    return K, info


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector")

    return v / n


def lens_axis_in_presentation(
    lens2camera: np.ndarray,
    camera2presentation: np.ndarray,
) -> np.ndarray:

    optical_axis_lens = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    R_l2c = lens2camera[:3, :3]
    R_c2p = camera2presentation[:3, :3]

    axis_camera = R_l2c @ optical_axis_lens
    axis_presentation = R_c2p @ axis_camera

    return normalize(axis_presentation)


def direction_to_yaw_pitch(direction):
    x, y, z = normalize(direction)

    yaw = np.arctan2(x, z)

    # Presentation Y points down, so positive mathematical
    # pitch/up corresponds to negative Y.
    pitch = np.arctan2(
        -y,
        np.sqrt(x * x + z * z),
    )

    return yaw, pitch


def yaw_pitch_to_direction(yaw, pitch):
    return normalize(
        np.array(
            [
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float64,
        )
    )


def virtual_to_presentation_rotation(
    yaw: float,
    pitch: float,
) -> np.ndarray:

    forward = yaw_pitch_to_direction(
        yaw,
        pitch,
    )

    presentation_down = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    right = np.cross(
        presentation_down,
        forward,
    )

    if np.linalg.norm(right) < 1e-8:
        raise ValueError(
            "Virtual camera direction is too close to vertical"
        )

    right = normalize(right)

    down = normalize(
        np.cross(
            forward,
            right,
        )
    )

    # Columns are the virtual camera axes expressed
    # in presentation coordinates.
    return np.column_stack(
        [right, down, forward]
    )


def virtual_intrinsic(
    width: int,
    height: int,
    hfov_deg: float,
) -> np.ndarray:

    hfov = np.deg2rad(hfov_deg)

    f = width / (
        2.0 * np.tan(hfov / 2.0)
    )

    return np.array(
        [
            [f, 0.0, (width - 1) / 2.0],
            [0.0, f, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distort_normalized_points(
    x,
    y,
    distortion,
):
    d = np.zeros(14, dtype=np.float64)
    d[: len(distortion)] = distortion

    (
        k1,
        k2,
        p1,
        p2,
        k3,
        k4,
        k5,
        k6,
        s1,
        s2,
        s3,
        s4,
        tau_x,
        tau_y,
    ) = d

    if abs(tau_x) > 1e-12 or abs(tau_y) > 1e-12:
        raise NotImplementedError(
            "Tilted-sensor distortion is not implemented "
            "because current Veo files use tauX=tauY=0."
        )

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    numerator = (
        1.0
        + k1 * r2
        + k2 * r4
        + k3 * r6
    )

    denominator = (
        1.0
        + k4 * r2
        + k5 * r4
        + k6 * r6
    )

    radial = numerator / denominator

    xd = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x * x)
        + s1 * r2
        + s2 * r4
    )

    yd = (
        y * radial
        + p1 * (r2 + 2.0 * y * y)
        + 2.0 * p2 * x * y
        + s3 * r2
        + s4 * r4
    )

    return xd, yd


def build_inverse_map(
    *,
    lens2camera,
    camera2presentation,
    K_raw,
    distortion,
    K_virtual,
    R_virtual2presentation,
    output_width,
    output_height,
    raw_width,
    raw_height,
):

    yy, xx = np.indices(
        (output_height, output_width),
        dtype=np.float64,
    )

    fx = K_virtual[0, 0]
    fy = K_virtual[1, 1]
    cx = K_virtual[0, 2]
    cy = K_virtual[1, 2]

    xv = (xx - cx) / fx
    yv = (yy - cy) / fy

    rays_virtual = np.stack(
        [
            xv,
            yv,
            np.ones_like(xv),
        ],
        axis=-1,
    )

    # virtual -> presentation
    rays_presentation = (
        rays_virtual
        @ R_virtual2presentation.T
    )

    R_c2p = camera2presentation[:3, :3]
    R_l2c = lens2camera[:3, :3]

    # presentation -> camera -> lens
    rays_camera = (
        rays_presentation
        @ R_c2p
    )

    rays_lens = (
        rays_camera
        @ R_l2c
    )

    X = rays_lens[..., 0]
    Y = rays_lens[..., 1]
    Z = rays_lens[..., 2]

    valid_front = Z > 1e-8

    xn = np.zeros_like(X)
    yn = np.zeros_like(Y)

    xn[valid_front] = (
        X[valid_front]
        / Z[valid_front]
    )

    yn[valid_front] = (
        Y[valid_front]
        / Z[valid_front]
    )

    xd, yd = distort_normalized_points(
        xn,
        yn,
        distortion,
    )

    fx_raw = K_raw[0, 0]
    fy_raw = K_raw[1, 1]
    cx_raw = K_raw[0, 2]
    cy_raw = K_raw[1, 2]

    u = fx_raw * xd + cx_raw
    v = fy_raw * yd + cy_raw

    valid = (
        valid_front
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < raw_width - 1)
        & (v >= 0)
        & (v < raw_height - 1)
    )

    map_x = np.full(
        (output_height, output_width),
        -1,
        dtype=np.float32,
    )

    map_y = np.full(
        (output_height, output_width),
        -1,
        dtype=np.float32,
    )

    map_x[valid] = u[valid].astype(
        np.float32
    )

    map_y[valid] = v[valid].astype(
        np.float32
    )

    return map_x, map_y, valid


def main():
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

    raw_images = split_stacked(
        stacked
    )

    raw = raw_images[
        args.lens
    ]

    raw_h, raw_w = raw.shape[:2]

    metadata = load_veo_metadata(
        args.calibration
    )

    lens = metadata.lens(
        args.lens
    )

    K_raw, mapping_info = (
        build_raw_intrinsic(
            lens.intrinsic,
            raw_w,
            raw_h,
            args.mapping,
        )
    )

    optical_axis = (
        lens_axis_in_presentation(
            lens.lens2camera,
            metadata.camera2presentation,
        )
    )

    base_yaw, base_pitch = (
        direction_to_yaw_pitch(
            optical_axis
        )
    )

    yaw = (
        base_yaw
        + np.deg2rad(args.yaw)
    )

    pitch = (
        base_pitch
        + np.deg2rad(args.pitch)
    )

    K_virtual = virtual_intrinsic(
        args.width,
        args.height,
        args.hfov,
    )

    R_v2p = (
        virtual_to_presentation_rotation(
            yaw,
            pitch,
        )
    )

    map_x, map_y, valid = (
        build_inverse_map(
            lens2camera=lens.lens2camera,
            camera2presentation=metadata.camera2presentation,
            K_raw=K_raw,
            distortion=lens.distortion,
            K_virtual=K_virtual,
            R_virtual2presentation=R_v2p,
            output_width=args.width,
            output_height=args.height,
            raw_width=raw_w,
            raw_height=raw_h,
        )
    )

    rendered = cv2.remap(
        raw,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = (
        valid.astype(np.uint8)
        * 255
    )

    prefix = (
        f"{args.lens}_"
        f"{args.mapping}_"
        f"hfov{args.hfov:g}"
    )

    image_path = (
        args.out
        / f"{prefix}.jpg"
    )

    mask_path = (
        args.out
        / f"{prefix}_mask.png"
    )

    summary_path = (
        args.out
        / f"{prefix}.json"
    )

    cv2.imwrite(
        str(image_path),
        rendered,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    cv2.imwrite(
        str(mask_path),
        mask,
    )

    summary = {
        "lens": args.lens,
        "mapping": args.mapping,
        "source": {
            "stacked": str(args.stacked),
            "calibration": str(
                args.calibration
            ),
        },
        "raw_size": [
            raw_w,
            raw_h,
        ],
        "mapping_info": mapping_info,
        "K_veo": (
            lens.intrinsic.tolist()
        ),
        "K_raw": K_raw.tolist(),
        "distortion": (
            lens.distortion.tolist()
        ),
        "lens2camera": (
            lens.lens2camera.tolist()
        ),
        "camera2presentation": (
            metadata
            .camera2presentation
            .tolist()
        ),
        "va_presentation": (
            metadata.va_presentation
        ),
        "lens_axis_presentation": (
            optical_axis.tolist()
        ),
        "base_yaw_deg": float(
            np.rad2deg(base_yaw)
        ),
        "base_pitch_deg": float(
            np.rad2deg(base_pitch)
        ),
        "virtual_yaw_deg": float(
            np.rad2deg(yaw)
        ),
        "virtual_pitch_deg": float(
            np.rad2deg(pitch)
        ),
        "hfov_deg": args.hfov,
        "output_size": [
            args.width,
            args.height,
        ],
        "K_virtual": (
            K_virtual.tolist()
        ),
        "valid_fraction": float(
            np.mean(valid)
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Lens: {args.lens.upper()}"
    )

    print(
        f"Mapping: {args.mapping}"
    )

    print(
        "Lens optical axis in presentation:"
    )

    print(
        f"  yaw   = "
        f"{np.rad2deg(base_yaw):.2f}°"
    )

    print(
        f"  pitch = "
        f"{np.rad2deg(base_pitch):.2f}°"
    )

    print()

    print(
        "Virtual camera:"
    )

    print(
        f"  yaw   = "
        f"{np.rad2deg(yaw):.2f}°"
    )

    print(
        f"  pitch = "
        f"{np.rad2deg(pitch):.2f}°"
    )

    print(
        f"  HFOV  = {args.hfov:.2f}°"
    )

    print()

    print("Effective RAW K:")
    print(K_raw)

    print()

    print(
        "Valid output: "
        f"{100.0 * np.mean(valid):.1f}%"
    )

    print()

    print(
        f"Image:   {image_path}"
    )

    print(
        f"Mask:    {mask_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
