#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_PATH = (
    ROOT
    / "data"
    / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
)

LEFT_PATH = ROOT / "samples" / "left_frame.jpg"
RIGHT_PATH = ROOT / "samples" / "right_frame.jpg"

OUTPUT_DIR = ROOT / "output" / "ground_aware_panorama"

OUTPUT_WIDTH = 3072
OUTPUT_HEIGHT = 810

CALIBRATION_WIDTH = 3840.0
CALIBRATION_HEIGHT = 1536.0

SOURCE_WIDTH = 2048.0
SOURCE_HEIGHT = 1024.0

SCALE_X = SOURCE_WIDTH / CALIBRATION_WIDTH
SCALE_Y = SOURCE_HEIGHT / CALIBRATION_HEIGHT

# Allow a few metres around the pitch so that the boundary
# itself is not aggressively clipped.
FIELD_MARGIN_METERS = 15.0


def parse_matrix(value, rows, cols):
    return np.array(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    ).reshape(rows, cols)


def parse_vector(value):
    return np.array(
        [float(x) for x in value.split(",")],
        dtype=np.float64,
    )


def build_ground_map(
    lens2camera,
    camera2world,
    camera2presentation,
    K,
    distortion,
    va,
    field_length,
    field_width,
    width,
    height,
):
    """
    Build panorama -> source-lens maps using the actual ground plane.

    For every panorama pixel:

        presentation ray
            ->
        central Veo camera ray
            ->
        Veo world ray
            ->
        intersection with Y = 0
            ->
        same physical world point projected into the real lens

    This explicitly accounts for the physical translation between
    the left and right lenses, unlike rotation-only stitching.
    """

    map_x = np.full(
        (height, width),
        -1.0,
        dtype=np.float32,
    )

    map_y = np.full(
        (height, width),
        -1.0,
        dtype=np.float32,
    )

    quality = np.full(
        (height, width),
        -1.0,
        dtype=np.float32,
    )

    half_length = field_length / 2.0
    half_width = field_width / 2.0

    yaw_values = np.linspace(
        va["left"],
        va["right"],
        width,
        dtype=np.float64,
    )

    presentation2camera = np.linalg.inv(
        camera2presentation
    )

    world2camera = np.linalg.inv(
        camera2world
    )

    camera2lens = np.linalg.inv(
        lens2camera
    )

    world2lens = (
        camera2lens
        @ world2camera
    )

    R_presentation2camera = (
        presentation2camera[:3, :3]
    )

    R_camera2world = (
        camera2world[:3, :3]
    )

    camera_origin_world = (
        camera2world[:3, 3]
    )

    zero_rotation = np.zeros(
        3,
        dtype=np.float64,
    )

    zero_translation = np.zeros(
        3,
        dtype=np.float64,
    )

    for y in range(height):
        t_row = y / max(height - 1, 1)

        pitch = (
            va["top"]
            + t_row
            * (va["bottom"] - va["top"])
        )

        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)

        sin_yaw = np.sin(yaw_values)
        cos_yaw = np.cos(yaw_values)

        rays_presentation = np.stack(
            [
                sin_yaw * cos_pitch,
                np.full_like(
                    yaw_values,
                    -sin_pitch,
                ),
                cos_yaw * cos_pitch,
            ],
            axis=1,
        )

        rays_camera = (
            rays_presentation
            @ R_presentation2camera.T
        )

        rays_world = (
            rays_camera
            @ R_camera2world.T
        )

        ray_y = rays_world[:, 1]

        # The Veo camera origin has Y < 0 and the pitch lies
        # on Y = 0. Therefore useful downward rays have Y > 0.
        can_hit_ground = ray_y > 1e-9

        ground_t = np.full(
            width,
            np.nan,
            dtype=np.float64,
        )

        ground_t[can_hit_ground] = (
            -camera_origin_world[1]
            / ray_y[can_hit_ground]
        )

        in_front = (
            can_hit_ground
            & (ground_t > 0.0)
        )

        if not np.any(in_front):
            continue

        world_points = (
            camera_origin_world[None, :]
            + ground_t[:, None] * rays_world
        )

        x_world = world_points[:, 0]
        z_world = world_points[:, 2]

        inside_working_plane = (
            in_front
            & (
                np.abs(x_world)
                <= half_length + FIELD_MARGIN_METERS
            )
            & (
                np.abs(z_world)
                <= half_width + FIELD_MARGIN_METERS
            )
        )

        indices = np.where(
            inside_working_plane
        )[0]

        if len(indices) == 0:
            continue

        selected_world = world_points[
            indices
        ]

        world_h = np.concatenate(
            [
                selected_world,
                np.ones(
                    (len(indices), 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        lens_h = (
            world_h
            @ world2lens.T
        )

        lens_xyz = lens_h[:, :3]

        lens_front = (
            lens_xyz[:, 2] > 1e-6
        )

        if not np.any(lens_front):
            continue

        front_indices = indices[
            lens_front
        ]

        front_xyz = lens_xyz[
            lens_front
        ]

        projected, _ = cv2.projectPoints(
            front_xyz.reshape(-1, 1, 3),
            zero_rotation,
            zero_translation,
            K,
            distortion,
        )

        projected = projected.reshape(
            -1,
            2,
        )

        u = projected[:, 0] * SCALE_X
        v = projected[:, 1] * SCALE_Y

        source_valid = (
            (u >= 0)
            & (u < SOURCE_WIDTH - 1)
            & (v >= 0)
            & (v < SOURCE_HEIGHT - 1)
        )

        final_indices = front_indices[
            source_valid
        ]

        if len(final_indices) == 0:
            continue

        map_x[
            y,
            final_indices,
        ] = u[source_valid].astype(
            np.float32
        )

        map_y[
            y,
            final_indices,
        ] = v[source_valid].astype(
            np.float32
        )

        valid_xyz = front_xyz[
            source_valid
        ]

        norm = np.linalg.norm(
            valid_xyz,
            axis=1,
        )

        q = (
            valid_xyz[:, 2]
            / np.maximum(norm, 1e-12)
        )

        quality[
            y,
            final_indices,
        ] = q.astype(
            np.float32
        )

    return map_x, map_y, quality


def warp_image(image, map_x, map_y):
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_mask(path, mask):
    cv2.imwrite(
        str(path),
        mask.astype(np.uint8) * 255,
    )


def crop_valid(image, valid_mask, padding=8):
    ys, xs = np.where(valid_mask)

    if len(xs) == 0:
        return image

    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(
        int(xs.max()) + padding + 1,
        image.shape[1],
    )

    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(
        int(ys.max()) + padding + 1,
        image.shape[0],
    )

    return image[y1:y2, x1:x2]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    left = cv2.imread(str(LEFT_PATH))
    right = cv2.imread(str(RIGHT_PATH))

    if left is None:
        raise RuntimeError(
            f"Could not load {LEFT_PATH}"
        )

    if right is None:
        raise RuntimeError(
            f"Could not load {RIGHT_PATH}"
        )

    with open(
        CALIBRATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        calibration = json.load(f)

    alignment = calibration["alignment"]

    K_left = parse_matrix(
        alignment["intrinsic_left"],
        3,
        3,
    )

    K_right = parse_matrix(
        alignment["intrinsic_right"],
        3,
        3,
    )

    distortion_left = parse_vector(
        alignment["distortion_left"]
    )

    distortion_right = parse_vector(
        alignment["distortion_right"]
    )

    lens_left2camera = parse_matrix(
        alignment["lens_left2camera"],
        4,
        4,
    )

    lens_right2camera = parse_matrix(
        alignment["lens_right2camera"],
        4,
        4,
    )

    camera2world = parse_matrix(
        alignment["camera2world"],
        4,
        4,
    )

    camera2presentation = parse_matrix(
        alignment["camera2presentation"],
        4,
        4,
    )

    field_length = float(
        alignment["field_length"]
    )

    field_width = float(
        alignment["field_width"]
    )

    va = alignment["va_presentation"]

    print(
        f"Field: "
        f"{field_length:.3f} x "
        f"{field_width:.3f} m"
    )

    print(
        "Building LEFT ground-aware map..."
    )

    (
        left_map_x,
        left_map_y,
        left_quality,
    ) = build_ground_map(
        lens_left2camera,
        camera2world,
        camera2presentation,
        K_left,
        distortion_left,
        va,
        field_length,
        field_width,
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
    )

    print(
        "Building RIGHT ground-aware map..."
    )

    (
        right_map_x,
        right_map_y,
        right_quality,
    ) = build_ground_map(
        lens_right2camera,
        camera2world,
        camera2presentation,
        K_right,
        distortion_right,
        va,
        field_length,
        field_width,
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
    )

    left_warp = warp_image(
        left,
        left_map_x,
        left_map_y,
    )

    right_warp = warp_image(
        right,
        right_map_x,
        right_map_y,
    )

    left_valid = left_map_x >= 0
    right_valid = right_map_x >= 0

    overlap = (
        left_valid
        & right_valid
    )

    valid_any = (
        left_valid
        | right_valid
    )

    # Hard selection is especially important here because
    # players are 3D objects and blending them may create ghosts.
    use_left = (
        left_valid
        & (
            ~right_valid
            | (
                left_quality
                >= right_quality
            )
        )
    )

    use_right = (
        right_valid
        & ~use_left
    )

    panorama_hard = np.zeros_like(
        left_warp
    )

    panorama_hard[
        use_left
    ] = left_warp[
        use_left
    ]

    panorama_hard[
        use_right
    ] = right_warp[
        use_right
    ]

    # Blend version only as a diagnostic.
    wl = np.where(
        left_valid,
        np.maximum(
            left_quality,
            0.0,
        ) ** 6,
        0.0,
    )

    wr = np.where(
        right_valid,
        np.maximum(
            right_quality,
            0.0,
        ) ** 6,
        0.0,
    )

    weight_sum = wl + wr

    panorama_blend = np.zeros_like(
        left_warp,
        dtype=np.float32,
    )

    blend_valid = (
        weight_sum > 1e-8
    )

    panorama_blend[
        blend_valid
    ] = (
        left_warp[
            blend_valid
        ].astype(np.float32)
        * wl[
            blend_valid,
            None,
        ]
        + right_warp[
            blend_valid
        ].astype(np.float32)
        * wr[
            blend_valid,
            None,
        ]
    ) / weight_sum[
        blend_valid,
        None,
    ]

    panorama_blend = np.clip(
        panorama_blend,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ground_left_warp.jpg"
        ),
        left_warp,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ground_right_warp.jpg"
        ),
        right_warp,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ground_panorama_hard.jpg"
        ),
        panorama_hard,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ground_panorama_blend.jpg"
        ),
        panorama_blend,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    hard_crop = crop_valid(
        panorama_hard,
        valid_any,
    )

    blend_crop = crop_valid(
        panorama_blend,
        valid_any,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ground_panorama_hard_crop.jpg"
        ),
        hard_crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ground_panorama_blend_crop.jpg"
        ),
        blend_crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    save_mask(
        OUTPUT_DIR
        / "ground_left_valid.png",
        left_valid,
    )

    save_mask(
        OUTPUT_DIR
        / "ground_right_valid.png",
        right_valid,
    )

    save_mask(
        OUTPUT_DIR
        / "ground_overlap.png",
        overlap,
    )

    total = (
        OUTPUT_WIDTH
        * OUTPUT_HEIGHT
    )

    print()
    print(
        "Left ground coverage:  "
        f"{100 * left_valid.sum() / total:.2f}%"
    )

    print(
        "Right ground coverage: "
        f"{100 * right_valid.sum() / total:.2f}%"
    )

    print(
        "Ground overlap:        "
        f"{100 * overlap.sum() / total:.2f}%"
    )

    print()
    print(
        f"Saved outputs to "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
