#!/usr/bin/env python3

"""Dual-lens OpenCV viewer for inspecting Veo pixel-to-field geometry."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from test_pixel_to_field import parse_matrix, parse_vector, pixel_to_field


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "data" / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
LEFT_PATH = ROOT / "samples" / "left_frame.jpg"
RIGHT_PATH = ROOT / "samples" / "right_frame.jpg"
WINDOW_NAME = "Veo dual-lens field debug"
MARGIN, HEADER_HEIGHT, LENS_WIDTH, LENS_HEIGHT, PANEL_GAP, PITCH_GAP = 20, 34, 760, 380, 20, 26
PITCH_MARGIN_METERS = 8.0


def load_calibration(calibration_path: Path):
    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        alignment = json.load(calibration_file)["alignment"]
    camera2world = parse_matrix(alignment["camera2world"], 4, 4)
    lenses = {
        name: {
            "K": parse_matrix(alignment[f"intrinsic_{name}"], 3, 3),
            "distortion": parse_vector(alignment[f"distortion_{name}"]),
            "lens2camera": parse_matrix(alignment[f"lens_{name}2camera"], 4, 4),
        }
        for name in ("left", "right")
    }
    return lenses, camera2world, float(alignment["field_length"]), float(alignment["field_width"])


def load_lens_images(stacked_path: Path | None, left_path: Path, right_path: Path):
    if stacked_path is not None:
        stacked = cv2.imread(str(stacked_path))
        if stacked is None:
            raise RuntimeError(f"Could not load {stacked_path}")
        height = stacked.shape[0]
        if height % 2:
            raise RuntimeError("Stacked frame height is not divisible by two")
        return {"left": stacked[:height // 2].copy(), "right": stacked[height // 2:].copy()}

    images = {"left": cv2.imread(str(left_path)), "right": cv2.imread(str(right_path))}
    for name, image in images.items():
        if image is None:
            raise RuntimeError(f"Could not load {left_path if name == 'left' else right_path}")
    return images


class PitchTransform:
    """Metric X/Z field coordinates mapped into the lower pitch panel."""

    def __init__(self, x: int, y: int, width: int, height: int, length: float, field_width: float):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.length, self.field_width = length, field_width
        self.scale = min(width / (length + 2 * PITCH_MARGIN_METERS), height / (field_width + 2 * PITCH_MARGIN_METERS))
        self.center_x, self.center_y = x + width / 2.0, y + height / 2.0

    def point(self, field_x: float, field_z: float) -> tuple[int, int]:
        return int(round(self.center_x + field_x * self.scale)), int(round(self.center_y - field_z * self.scale))


def draw_metric_rectangle(canvas, transform, min_x, max_x, min_z, max_z, color, thickness=2):
    cv2.rectangle(canvas, transform.point(min_x, max_z), transform.point(max_x, min_z), color, thickness, cv2.LINE_AA)


def draw_penalty_arc(canvas, transform, penalty_x, box_edge_x, right_goal):
    radius = 9.15
    limit = math.acos(abs(box_edge_x - penalty_x) / radius)
    angles = np.linspace(math.pi - limit, math.pi + limit, 40) if right_goal else np.linspace(-limit, limit, 40)
    points = np.array([transform.point(penalty_x + radius * math.cos(angle), radius * math.sin(angle)) for angle in angles], dtype=np.int32)
    cv2.polylines(canvas, [points], False, (235, 235, 235), 2, cv2.LINE_AA)


def draw_pitch(canvas: np.ndarray, transform: PitchTransform, markers: list[dict]):
    length, width = transform.length, transform.field_width
    half_length, half_width = length / 2.0, width / 2.0
    white = (235, 235, 235)
    cv2.rectangle(canvas, (transform.x, transform.y), (transform.x + transform.width, transform.y + transform.height), (28, 92, 28), -1)
    draw_metric_rectangle(canvas, transform, -half_length, half_length, -half_width, half_width, white, 3)
    cv2.line(canvas, transform.point(0.0, -half_width), transform.point(0.0, half_width), white, 2, cv2.LINE_AA)
    center = transform.point(0.0, 0.0)
    cv2.circle(canvas, center, int(round(9.15 * transform.scale)), white, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 4, white, -1, cv2.LINE_AA)

    penalty_half_width, goal_area_half_width = min(20.16, half_width), min(9.16, half_width)
    left_penalty_x, right_penalty_x = -half_length + 11.0, half_length - 11.0
    left_box_edge, right_box_edge = -half_length + 16.5, half_length - 16.5
    draw_metric_rectangle(canvas, transform, -half_length, left_box_edge, -penalty_half_width, penalty_half_width, white)
    draw_metric_rectangle(canvas, transform, right_box_edge, half_length, -penalty_half_width, penalty_half_width, white)
    draw_metric_rectangle(canvas, transform, -half_length, -half_length + 5.5, -goal_area_half_width, goal_area_half_width, white)
    draw_metric_rectangle(canvas, transform, half_length - 5.5, half_length, -goal_area_half_width, goal_area_half_width, white)
    for penalty_x in (left_penalty_x, right_penalty_x):
        cv2.circle(canvas, transform.point(penalty_x, 0.0), 4, white, -1, cv2.LINE_AA)
    draw_penalty_arc(canvas, transform, left_penalty_x, left_box_edge, False)
    draw_penalty_arc(canvas, transform, right_penalty_x, right_box_edge, True)
    goal_half_width = min(3.66, half_width)
    draw_metric_rectangle(canvas, transform, -half_length - 2.44, -half_length, -goal_half_width, goal_half_width, white)
    draw_metric_rectangle(canvas, transform, half_length, half_length + 2.44, -goal_half_width, goal_half_width, white)

    corner_radius = int(round(transform.scale))
    for field_x, field_z, start, end in (
        (-half_length, +half_width, 0, 90),
        (+half_length, +half_width, 90, 180),
        (+half_length, -half_width, 180, 270),
        (-half_length, -half_width, 270, 360),
    ):
        cv2.ellipse(canvas, transform.point(field_x, field_z),
                    (corner_radius, corner_radius), 0, start, end,
                    white, 1, cv2.LINE_AA)

    omitted = []
    for marker in markers:
        point = transform.point(marker["x"], marker["z"])
        visible = transform.x <= point[0] < transform.x + transform.width and transform.y <= point[1] < transform.y + transform.height
        color = (0, 220, 255) if marker["inside"] else (0, 70, 255)
        if visible:
            cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, 20, 3)
            cv2.putText(canvas, f"#{marker['id']}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        elif not marker["inside"]:
            omitted.append(f"#{marker['id']}")
    cv2.putText(canvas, f"Pitch: X={length:.2f} m, Z={width:.2f} m", (transform.x + 12, transform.y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 2, cv2.LINE_AA)
    if omitted:
        cv2.putText(canvas, "Outside pitch view: " + ", ".join(omitted), (transform.x + 12, transform.y + transform.height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2, cv2.LINE_AA)


def lens_panel_rect(lens_name: str) -> tuple[int, int, int, int]:
    return (MARGIN if lens_name == "left" else MARGIN + LENS_WIDTH + PANEL_GAP, HEADER_HEIGHT, LENS_WIDTH, LENS_HEIGHT)


def raw_to_panel(raw_pixel, image, panel):
    x, y, width, height = panel
    raw_height, raw_width = image.shape[:2]
    return int(round(x + raw_pixel[0] * width / raw_width)), int(round(y + raw_pixel[1] * height / raw_height))


def panel_to_raw(click_x, click_y, image, panel):
    x, y, width, height = panel
    if not (x <= click_x < x + width and y <= click_y < y + height):
        return None
    raw_height, raw_width = image.shape[:2]
    return np.array([(click_x - x) * raw_width / width, (click_y - y) * raw_height / height], dtype=np.float64)


def render(images, field_length, field_width, markers):
    canvas_width = 2 * MARGIN + 2 * LENS_WIDTH + PANEL_GAP
    pitch_width = canvas_width - 2 * MARGIN
    pitch_height = int(round(pitch_width * (field_width + 2 * PITCH_MARGIN_METERS) / (field_length + 2 * PITCH_MARGIN_METERS)))
    pitch_y = HEADER_HEIGHT + LENS_HEIGHT + PITCH_GAP
    canvas_height = pitch_y + pitch_height + MARGIN
    canvas = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)
    for lens_name in ("left", "right"):
        panel = lens_panel_rect(lens_name)
        x, y, width, height = panel
        canvas[y:y + height, x:x + width] = cv2.resize(images[lens_name], (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (235, 235, 235), 2)
        cv2.putText(canvas, lens_name.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        for marker in markers:
            if marker["lens"] == lens_name:
                point = raw_to_panel(marker["raw"], images[lens_name], panel)
                color = (0, 220, 255) if marker["inside"] else (0, 70, 255)
                cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, 20, 3)
                cv2.putText(canvas, f"#{marker['id']}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    transform = PitchTransform(MARGIN, pitch_y, pitch_width, pitch_height, field_length, field_width)
    draw_pitch(canvas, transform, markers)
    cv2.putText(canvas, "Click either lens | C clear | Z undo | Q/Esc quit", (MARGIN, canvas_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--left", type=Path, default=LEFT_PATH)
    parser.add_argument("--right", type=Path, default=RIGHT_PATH)
    parser.add_argument("--stacked", type=Path, help="Optional stacked frame; top and bottom halves become left and right")
    parser.add_argument("--lens", choices=("left", "right"), help="Accepted for compatibility; both lenses are displayed")
    args = parser.parse_args()

    images = load_lens_images(args.stacked, args.left, args.right)
    lenses, camera2world, field_length, field_width = load_calibration(args.calibration)
    markers: list[dict] = []

    def refresh():
        cv2.imshow(WINDOW_NAME, render(images, field_length, field_width, markers))

    def on_click(event, click_x, click_y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for lens_name in ("left", "right"):
            raw_pixel = panel_to_raw(click_x, click_y, images[lens_name], lens_panel_rect(lens_name))
            if raw_pixel is None:
                continue
            try:
                world = pixel_to_field(raw_pixel, lenses[lens_name]["K"], lenses[lens_name]["distortion"], lenses[lens_name]["lens2camera"], camera2world)
            except RuntimeError as error:
                print(f"{lens_name.upper()} raw pixel: ({raw_pixel[0]:.1f}, {raw_pixel[1]:.1f})")
                print(f"projection error: {error}\n")
                return
            field_x, field_z = float(world[0]), float(world[2])
            inside = -field_length / 2.0 <= field_x <= field_length / 2.0 and -field_width / 2.0 <= field_z <= field_width / 2.0
            marker = {"id": len(markers) + 1, "lens": lens_name, "raw": (float(raw_pixel[0]), float(raw_pixel[1])), "x": field_x, "z": field_z, "inside": inside}
            markers.append(marker)
            print(f"#{marker['id']:02d} {lens_name.upper()}")
            print(f"raw pixel: ({raw_pixel[0]:.1f}, {raw_pixel[1]:.1f})")
            print(f"field: X={field_x:.3f} m, Z={field_z:.3f} m")
            print(f"inside field: {'YES' if inside else 'NO'}\n")
            refresh()
            return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_click)
    refresh()
    print("Click either lens. Press C to clear, Z to undo, or Q/Escape to quit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            markers.clear()
            refresh()
        if key == ord("z") and markers:
            markers.pop()
            refresh()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
