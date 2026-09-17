#!/usr/bin/env python3

"""Dual-lens OpenCV viewer for inspecting Veo pixel-to-field geometry."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.veo_geometry import VeoCalibration, load_veo_calibration


CALIBRATION_PATH = ROOT / "data" / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
LEFT_PATH = ROOT / "samples" / "left_frame.jpg"
RIGHT_PATH = ROOT / "samples" / "right_frame.jpg"
WINDOW_NAME = "Veo dual-lens field debug"
MARGIN, HEADER_HEIGHT, LENS_WIDTH, LENS_HEIGHT, PANEL_GAP, PITCH_GAP = 20, 34, 1040, 520, 20, 26
PITCH_HEIGHT = 270
PITCH_MARGIN_METERS = 8.0
PANEL_BORDER = 2
MAGNIFIER_RAW_SIZE = 31
MAGNIFIER_SCALE = 10
MAGNIFIER_MARGIN = 8


def load_calibration(calibration_path: Path) -> VeoCalibration:
    return load_veo_calibration(calibration_path)


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
    if not (
        x + PANEL_BORDER <= click_x < x + width - PANEL_BORDER
        and y + PANEL_BORDER <= click_y < y + height - PANEL_BORDER
    ):
        return None
    raw_height, raw_width = image.shape[:2]
    return np.array([(click_x - x) * raw_width / width, (click_y - y) * raw_height / height], dtype=np.float64)


def resolve_lens_at_canvas(canvas_x, canvas_y, images):
    """Return the lens and raw pixel under a composed-canvas position."""
    for lens_name in ("left", "right"):
        panel = lens_panel_rect(lens_name)
        raw_pixel = panel_to_raw(canvas_x, canvas_y, images[lens_name], panel)
        if raw_pixel is not None:
            return lens_name, raw_pixel, panel
    return None


def draw_precision_crosshair(canvas, canvas_x, canvas_y, lens_name, raw_pixel, panel):
    x, y = int(round(canvas_x)), int(round(canvas_y))
    color = (255, 220, 0)
    gap, arm = 4, 14
    cv2.line(canvas, (x - arm, y), (x - gap, y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x + gap, y), (x + arm, y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - arm), (x, y - gap), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y + gap), (x, y + arm), color, 1, cv2.LINE_AA)

    label = f"{lens_name.upper()} u={raw_pixel[0]:.0f} v={raw_pixel[1]:.0f}"
    label_x = x + 12
    if label_x + 180 > panel[0] + panel[2]:
        label_x = x - 180
    label_y = y + 22 if y < panel[1] + 38 else y - 10
    cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 1, cv2.LINE_AA)


def magnifier_position(canvas_x, canvas_y, panel, size):
    """Choose a panel corner that does not cover the current target."""
    panel_x, panel_y, panel_width, panel_height = panel
    candidates = (
        (panel_x + panel_width - size - MAGNIFIER_MARGIN, panel_y + MAGNIFIER_MARGIN),
        (panel_x + MAGNIFIER_MARGIN, panel_y + MAGNIFIER_MARGIN),
        (panel_x + panel_width - size - MAGNIFIER_MARGIN, panel_y + panel_height - size - MAGNIFIER_MARGIN),
        (panel_x + MAGNIFIER_MARGIN, panel_y + panel_height - size - MAGNIFIER_MARGIN),
    )
    for x, y in candidates:
        if not (x - 16 <= canvas_x <= x + size + 16 and y - 16 <= canvas_y <= y + size + 16):
            return x, y
    return candidates[0]


def draw_magnifier(canvas, image, panel, lens_name, raw_pixel, canvas_pixel):
    """Draw a nearest-neighbor view of the original RAW pixels."""
    raw_height, raw_width = image.shape[:2]
    center_u, center_v = int(round(raw_pixel[0])), int(round(raw_pixel[1]))
    half = MAGNIFIER_RAW_SIZE // 2
    x0, y0 = center_u - half, center_v - half
    x1, y1 = x0 + MAGNIFIER_RAW_SIZE, y0 + MAGNIFIER_RAW_SIZE

    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(raw_width, x1), min(raw_height, y1)
    crop = image[source_y0:source_y1, source_x0:source_x1]
    crop = cv2.copyMakeBorder(
        crop,
        source_y0 - y0,
        y1 - source_y1,
        source_x0 - x0,
        x1 - source_x1,
        cv2.BORDER_REPLICATE,
    )
    size = MAGNIFIER_RAW_SIZE * MAGNIFIER_SCALE
    enlarged = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    magnifier_x, magnifier_y = magnifier_position(canvas_pixel[0], canvas_pixel[1], panel, size)
    canvas[magnifier_y:magnifier_y + size, magnifier_x:magnifier_x + size] = enlarged
    cv2.rectangle(canvas, (magnifier_x, magnifier_y),
                  (magnifier_x + size, magnifier_y + size), (255, 255, 255), 2)

    cross_x = magnifier_x + int(round((raw_pixel[0] - x0) * MAGNIFIER_SCALE))
    cross_y = magnifier_y + int(round((raw_pixel[1] - y0) * MAGNIFIER_SCALE))
    color = (0, 255, 255)
    gap, arm = 5, 24
    cv2.line(canvas, (cross_x - arm, cross_y), (cross_x - gap, cross_y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cross_x + gap, cross_y), (cross_x + arm, cross_y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cross_x, cross_y - arm), (cross_x, cross_y - gap), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cross_x, cross_y + gap), (cross_x, cross_y + arm), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{lens_name.upper()}  u={raw_pixel[0]:.1f} v={raw_pixel[1]:.1f}",
                (magnifier_x, magnifier_y + size + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


def render(images, field_length, field_width, markers, hover=None):
    canvas_width = 2 * MARGIN + 2 * LENS_WIDTH + PANEL_GAP
    pitch_width = int(round(
        PITCH_HEIGHT * (field_length + 2 * PITCH_MARGIN_METERS)
        / (field_width + 2 * PITCH_MARGIN_METERS)
    ))
    pitch_x = (canvas_width - pitch_width) // 2
    pitch_height = PITCH_HEIGHT
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
    if hover is not None:
        draw_precision_crosshair(
            canvas,
            hover["canvas"][0],
            hover["canvas"][1],
            hover["lens"],
            hover["raw"],
            hover["panel"],
        )
        draw_magnifier(
            canvas,
            images[hover["lens"]],
            hover["panel"],
            hover["lens"],
            hover["raw"],
            hover["canvas"],
        )
    transform = PitchTransform(pitch_x, pitch_y, pitch_width, pitch_height, field_length, field_width)
    draw_pitch(canvas, transform, markers)
    cv2.putText(canvas, "Click either lens | C clear | Z undo | Q/Esc quit", (MARGIN, canvas_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def adjust_hover(hover, images, delta_u, delta_v):
    """Move the active precision cursor by exact RAW-pixel increments."""
    if hover is None:
        return None
    image = images[hover["lens"]]
    raw_height, raw_width = image.shape[:2]
    raw_u = max(0.0, min(raw_width - 1.0, hover["raw"][0] + delta_u))
    raw_v = max(0.0, min(raw_height - 1.0, hover["raw"][1] + delta_v))
    raw_pixel = (raw_u, raw_v)
    panel_pixel = raw_to_panel(raw_pixel, image, hover["panel"])
    return {**hover, "raw": raw_pixel, "canvas": panel_pixel}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--left", type=Path, default=LEFT_PATH)
    parser.add_argument("--right", type=Path, default=RIGHT_PATH)
    parser.add_argument("--stacked", type=Path, help="Optional stacked frame; top and bottom halves become left and right")
    parser.add_argument("--lens", choices=("left", "right"), help="Accepted for compatibility; both lenses are displayed")
    parser.add_argument("--debug-clicks", action="store_true", help="Print canvas and raw click coordinates")
    args = parser.parse_args()

    images = load_lens_images(args.stacked, args.left, args.right)
    calibration = load_calibration(args.calibration)
    field_length, field_width = calibration.field_length, calibration.field_width
    markers: list[dict] = []
    hover = None
    viewer_frame = render(images, field_length, field_width, markers, hover)

    def refresh():
        nonlocal viewer_frame
        viewer_frame = render(images, field_length, field_width, markers, hover)
        cv2.imshow(WINDOW_NAME, viewer_frame)

    def on_click(event, click_x, click_y, _flags, _userdata):
        nonlocal hover
        if event == cv2.EVENT_MOUSEMOVE:
            resolved = resolve_lens_at_canvas(click_x, click_y, images)
            next_hover = None
            if resolved is not None:
                lens_name, raw_pixel, panel = resolved
                next_hover = {
                    "lens": lens_name,
                    "raw": (float(raw_pixel[0]), float(raw_pixel[1])),
                    "panel": panel,
                    "canvas": (click_x, click_y),
                }
            if next_hover != hover:
                hover = next_hover
                refresh()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        resolved = resolve_lens_at_canvas(click_x, click_y, images)
        if resolved is None:
            return
        lens_name, raw_pixel, panel = resolved
        if hover is not None and hover["lens"] == lens_name:
            # Arrow-key refinement changes the active precision cursor without
            # moving the native mouse, so confirmation uses that exact RAW point.
            raw_pixel = np.array(hover["raw"], dtype=np.float64)
        if args.debug_clicks:
            print(f"canvas click: ({click_x:.1f}, {click_y:.1f}) -> {lens_name.upper()} raw: ({raw_pixel[0]:.1f}, {raw_pixel[1]:.1f})")
        try:
            field = calibration.pixel_to_field(lens_name, raw_pixel)
        except RuntimeError as error:
            print(f"{lens_name.upper()} raw pixel: ({raw_pixel[0]:.1f}, {raw_pixel[1]:.1f})")
            print(f"projection error: {error}\n")
            return
        field_x, field_z = float(field[0]), float(field[1])
        inside = calibration.is_inside_field(field)
        marker = {"id": len(markers) + 1, "lens": lens_name, "raw": (float(raw_pixel[0]), float(raw_pixel[1])), "x": field_x, "z": field_z, "inside": inside}
        markers.append(marker)
        print(f"#{marker['id']:02d} {lens_name.upper()}")
        print(f"raw pixel: ({raw_pixel[0]:.1f}, {raw_pixel[1]:.1f})")
        print(f"field: X={field_x:.3f} m, Z={field_z:.3f} m")
        print(f"inside field: {'YES' if inside else 'NO'}\n")
        refresh()

    window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(WINDOW_NAME, window_flags)
    cv2.resizeWindow(WINDOW_NAME, 1500, 800)
    cv2.setMouseCallback(WINDOW_NAME, on_click)
    refresh()
    print("Click either lens. Press C to clear, Z to undo, or Q/Escape to quit.")
    while True:
        key = cv2.waitKeyEx(20)
        key_low = key & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key_low in (ord("c"), ord("C")):
            markers.clear()
            refresh()
        if key_low in (ord("z"), ord("Z")) and markers:
            markers.pop()
            refresh()
        arrow_delta = {
            81: (-1, 0), 83: (1, 0), 82: (0, -1), 84: (0, 1),
            65361: (-1, 0), 65363: (1, 0), 65362: (0, -1), 65364: (0, 1),
            2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1),
        }.get(key)
        if arrow_delta is not None and hover is not None:
            hover = adjust_hover(hover, images, *arrow_delta)
            refresh()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
