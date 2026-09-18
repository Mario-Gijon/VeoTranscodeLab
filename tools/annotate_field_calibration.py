#!/usr/bin/env python3
"""Manually collect lens-local RAW annotations for fixed-camera calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.veo_metadata import VeoMetadata, load_veo_metadata


LINE_LABELS = (
    "touchline_near", "touchline_far", "goal_line_left", "goal_line_right",
    "halfway_line", "penalty_area_left_front", "penalty_area_right_front",
    "goal_area_left_front", "goal_area_right_front", "penalty_area_left_near_side",
    "penalty_area_left_far_side", "penalty_area_right_near_side",
    "penalty_area_right_far_side", "goal_area_left_near_side",
    "goal_area_left_far_side", "goal_area_right_near_side", "goal_area_right_far_side",
)
LANDMARK_LABELS = (
    "center", "corner_left_near", "corner_left_far", "corner_right_near",
    "corner_right_far", "halfway_near", "halfway_far", "penalty_spot_left",
    "penalty_spot_right", "penalty_area_left_near", "penalty_area_left_far",
    "penalty_area_right_near", "penalty_area_right_far", "goal_area_left_near",
    "goal_area_left_far", "goal_area_right_near", "goal_area_right_far",
    "center_circle_halfway_near", "center_circle_halfway_far",
)
MODES = ("lines", "landmarks", "correspondences", "center_circle")
WINDOW_NAME = "Veo field calibration annotations"
MARGIN = 18
HEADER_HEIGHT = 62
FOOTER_HEIGHT = 24
PANEL_HEIGHT = 500
PANEL_GAP = 18
MAGNIFIER_SIZE = 25
MAGNIFIER_SCALE = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked", type=Path, required=True, help="Stacked LEFT-over-RIGHT Veo frame")
    parser.add_argument("--calibration", type=Path, required=True, help="Veo .veo metadata file")
    parser.add_argument("--out", type=Path, required=True, help="Annotation JSON path; existing files are resumed")
    return parser.parse_args()


def load_stacked_lenses(path: Path) -> tuple[dict[str, np.ndarray], int, int]:
    stacked = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if stacked is None:
        raise RuntimeError(f"Could not read stacked image: {path}")
    stacked_height, stacked_width = stacked.shape[:2]
    if stacked_height % 2:
        raise ValueError("Stacked image height must be divisible by two")
    lens_height = stacked_height // 2
    return {
        "left": stacked[:lens_height].copy(),
        "right": stacked[lens_height:].copy(),
    }, stacked_width, stacked_height


def empty_annotations() -> dict:
    return {
        "lines": {"left": {}, "right": {}},
        "landmarks": {"left": {}, "right": {}},
        "center_circle": {"left": [], "right": []},
        "correspondences": [],
    }


def anchors_json(metadata: VeoMetadata, width: int, height: int) -> dict:
    anchors = {"left": {}, "right": {}}
    for corner in metadata.selected_corners(stacked_width=width, stacked_height=height):
        anchors[corner.lens][corner.name] = [
            int(round(corner.raw_pixel[0])),
            int(round(corner.raw_pixel[1])),
        ]
    return anchors


def new_document(metadata: VeoMetadata, stacked_path: Path, width: int, height: int) -> dict:
    return {
        "version": 1,
        "source": {
            "stacked_image": str(stacked_path),
            "calibration": str(metadata.source_path),
            "stacked_width": width,
            "stacked_height": height,
            "lens_width": width,
            "lens_height": height // 2,
        },
        "field": {"length": metadata.field_length, "width": metadata.field_width},
        "veo_anchors": anchors_json(metadata, width, height),
        "annotations": empty_annotations(),
    }


def load_or_create_document(out_path: Path, metadata: VeoMetadata, stacked_path: Path, width: int, height: int) -> dict:
    if not out_path.exists():
        return new_document(metadata, stacked_path, width, height)
    with out_path.open("r", encoding="utf-8") as source_file:
        document = json.load(source_file)
    if document.get("version") != 1:
        raise ValueError(f"Unsupported annotation version in {out_path}")
    annotations = document.setdefault("annotations", empty_annotations())
    annotations.setdefault("lines", {"left": {}, "right": {}})
    annotations.setdefault("landmarks", {"left": {}, "right": {}})
    annotations.setdefault("center_circle", {"left": [], "right": []})
    annotations.setdefault("correspondences", [])
    for group in ("lines", "landmarks", "center_circle"):
        annotations[group].setdefault("left", {} if group != "center_circle" else [])
        annotations[group].setdefault("right", {} if group != "center_circle" else [])
    document.setdefault("veo_anchors", anchors_json(metadata, width, height))
    return document


def atomic_save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary_file:
        json.dump(document, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_name = temporary_file.name
    os.replace(temporary_name, path)


def panel_layout(images: dict[str, np.ndarray]) -> dict[str, tuple[int, int, int, int]]:
    width = int(round(images["left"].shape[1] * PANEL_HEIGHT / images["left"].shape[0]))
    return {
        "left": (MARGIN, HEADER_HEIGHT, width, PANEL_HEIGHT),
        "right": (MARGIN + width + PANEL_GAP, HEADER_HEIGHT, width, PANEL_HEIGHT),
    }


def raw_to_panel(pixel: tuple[int, int], image: np.ndarray, panel: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, width, height = panel
    image_height, image_width = image.shape[:2]
    return (
        int(round(x + pixel[0] * width / image_width)),
        int(round(y + pixel[1] * height / image_height)),
    )


def panel_to_raw(canvas_x: int, canvas_y: int, image: np.ndarray, panel: tuple[int, int, int, int]) -> tuple[int, int] | None:
    x, y, width, height = panel
    if not (x <= canvas_x < x + width and y <= canvas_y < y + height):
        return None
    image_height, image_width = image.shape[:2]
    raw_x = int(round((canvas_x - x) * image_width / width))
    raw_y = int(round((canvas_y - y) * image_height / height))
    return min(max(raw_x, 0), image_width - 1), min(max(raw_y, 0), image_height - 1)


def find_lens(canvas_x: int, canvas_y: int, images: dict[str, np.ndarray], panels: dict[str, tuple[int, int, int, int]]):
    for lens in ("left", "right"):
        raw = panel_to_raw(canvas_x, canvas_y, images[lens], panels[lens])
        if raw is not None:
            return lens, raw
    return None


def draw_crosshair(canvas: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], size: int = 12) -> None:
    x, y = point
    cv2.line(canvas, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)


def draw_magnifier(canvas: np.ndarray, image: np.ndarray, raw: tuple[int, int], panel: tuple[int, int, int, int]) -> None:
    raw_height, raw_width = image.shape[:2]
    half = MAGNIFIER_SIZE // 2
    x0, y0 = raw[0] - half, raw[1] - half
    x1, y1 = x0 + MAGNIFIER_SIZE, y0 + MAGNIFIER_SIZE
    source = image[max(0, y0):min(raw_height, y1), max(0, x0):min(raw_width, x1)]
    source = cv2.copyMakeBorder(source, max(0, -y0), max(0, y1 - raw_height), max(0, -x0), max(0, x1 - raw_width), cv2.BORDER_REPLICATE)
    size = MAGNIFIER_SIZE * MAGNIFIER_SCALE
    enlarged = cv2.resize(source, (size, size), interpolation=cv2.INTER_NEAREST)
    panel_x, panel_y, panel_width, _ = panel
    origin = (panel_x + panel_width - size - 8, panel_y + 8)
    canvas[origin[1]:origin[1] + size, origin[0]:origin[0] + size] = enlarged
    cv2.rectangle(canvas, origin, (origin[0] + size, origin[1] + size), (255, 255, 255), 2)
    cross = (origin[0] + int(round((raw[0] - x0) * MAGNIFIER_SCALE)), origin[1] + int(round((raw[1] - y0) * MAGNIFIER_SCALE)))
    draw_crosshair(canvas, cross, (0, 255, 255), 18)


def draw_annotations(canvas: np.ndarray, images: dict[str, np.ndarray], panels: dict[str, tuple[int, int, int, int]], document: dict) -> None:
    annotations = document["annotations"]
    anchors = document.get("veo_anchors", {})
    for lens in ("left", "right"):
        panel = panels[lens]
        for name, point in anchors.get(lens, {}).items():
            pixel = raw_to_panel(tuple(point), images[lens], panel)
            cv2.drawMarker(canvas, pixel, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(canvas, name.upper(), (pixel[0] + 8, pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        for label, points in annotations["lines"][lens].items():
            for point in points:
                cv2.circle(canvas, raw_to_panel(tuple(point), images[lens], panel), 4, (0, 220, 255), -1, cv2.LINE_AA)
        for label, point in annotations["landmarks"][lens].items():
            pixel = raw_to_panel(tuple(point), images[lens], panel)
            cv2.drawMarker(canvas, pixel, (0, 255, 0), cv2.MARKER_DIAMOND, 14, 2)
            cv2.putText(canvas, label, (pixel[0] + 7, pixel[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        for point in annotations["center_circle"][lens]:
            cv2.circle(canvas, raw_to_panel(tuple(point), images[lens], panel), 4, (255, 180, 0), -1, cv2.LINE_AA)
    for pair in annotations["correspondences"]:
        for lens in ("left", "right"):
            pixel = raw_to_panel(tuple(pair[lens]), images[lens], panels[lens])
            cv2.drawMarker(canvas, pixel, (0, 120, 255), cv2.MARKER_STAR, 15, 2)
            cv2.putText(canvas, pair["id"], (pixel[0] + 7, pixel[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 255), 1, cv2.LINE_AA)


def next_pair_id(document: dict) -> str:
    ids = [pair.get("id", "") for pair in document["annotations"]["correspondences"]]
    numbers = [int(value.split("_")[-1]) for value in ids if value.startswith("pair_") and value.split("_")[-1].isdigit()]
    return f"pair_{max(numbers, default=0) + 1:03d}"


class AnnotationApp:
    def __init__(self, images: dict[str, np.ndarray], metadata: VeoMetadata, document: dict, out_path: Path):
        self.images = images
        self.metadata = metadata
        self.document = document
        self.out_path = out_path
        self.panels = panel_layout(images)
        self.mode_index = 0
        self.line_index = 0
        self.landmark_index = 0
        self.hover = None
        self.pending_pair = None
        self.history: list[tuple] = []

    @property
    def mode(self) -> str:
        return MODES[self.mode_index]

    @property
    def label(self) -> str | None:
        if self.mode == "lines":
            return LINE_LABELS[self.line_index]
        if self.mode == "landmarks":
            return LANDMARK_LABELS[self.landmark_index]
        return None

    def add_point(self, lens: str, point: tuple[int, int]) -> None:
        annotations = self.document["annotations"]
        if self.mode == "lines":
            points = annotations["lines"][lens].setdefault(self.label, [])
            points.append(list(point))
            self.history.append(("list", points))
        elif self.mode == "landmarks":
            landmarks = annotations["landmarks"][lens]
            previous = landmarks.get(self.label)
            landmarks[self.label] = list(point)
            self.history.append(("landmark", landmarks, self.label, previous))
        elif self.mode == "center_circle":
            points = annotations["center_circle"][lens]
            points.append(list(point))
            self.history.append(("list", points))
        else:
            self.add_correspondence_point(lens, point)

    def add_correspondence_point(self, lens: str, point: tuple[int, int]) -> None:
        if self.pending_pair is None:
            self.pending_pair = {"lens": lens, "pixel": list(point)}
            return
        if self.pending_pair["lens"] == lens:
            print("Pending correspondence needs a point from the other lens.")
            return
        pair = {"id": next_pair_id(self.document), lens: list(point), self.pending_pair["lens"]: self.pending_pair["pixel"]}
        self.document["annotations"]["correspondences"].append(pair)
        self.history.append(("pair", self.document["annotations"]["correspondences"]))
        self.pending_pair = None

    def undo(self) -> None:
        if not self.history:
            print("Nothing added in this session to undo.")
            return
        action = self.history.pop()
        if action[0] in ("list", "pair"):
            action[1].pop()
        else:
            _, landmarks, label, previous = action
            if previous is None:
                landmarks.pop(label, None)
            else:
                landmarks[label] = previous

    def cycle_label(self, direction: int) -> None:
        if self.mode == "lines":
            self.line_index = (self.line_index + direction) % len(LINE_LABELS)
        elif self.mode == "landmarks":
            self.landmark_index = (self.landmark_index + direction) % len(LANDMARK_LABELS)

    def render(self) -> np.ndarray:
        panel_width = self.panels["left"][2]
        canvas_width = 2 * MARGIN + 2 * panel_width + PANEL_GAP
        canvas_height = HEADER_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT + MARGIN
        canvas = np.full((canvas_height, canvas_width, 3), 22, dtype=np.uint8)
        for lens in ("left", "right"):
            x, y, width, height = self.panels[lens]
            canvas[y:y + height, x:x + width] = cv2.resize(self.images[lens], (width, height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (235, 235, 235), 2)
            cv2.putText(canvas, lens.upper(), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
        draw_annotations(canvas, self.images, self.panels, self.document)
        if self.pending_pair is not None:
            lens = self.pending_pair["lens"]
            point = raw_to_panel(tuple(self.pending_pair["pixel"]), self.images[lens], self.panels[lens])
            cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_STAR, 24, 3)
        if self.hover is not None:
            lens, raw, canvas_point = self.hover
            draw_crosshair(canvas, canvas_point, (255, 220, 0))
            draw_magnifier(canvas, self.images[lens], raw, self.panels[lens])
        version = self.metadata.raw_json.get("calibration_version", "unknown")
        label_text = self.label if self.label is not None else "no semantic label"
        cv2.putText(canvas, f"MODE: {self.mode} | LABEL: {label_text}", (MARGIN, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Field {self.metadata.field_length:.2f}m x {self.metadata.field_width:.2f}m | Veo calibration: {version}", (MARGIN, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        if self.pending_pair is not None:
            pending = self.pending_pair
            pending_text = f"PENDING {pending['lens'].upper()} ({pending['pixel'][0]}, {pending['pixel'][1]}) | click the other lens, C cancel"
            cv2.putText(canvas, pending_text, (MARGIN + 720, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1, cv2.LINE_AA)
        if self.hover is not None:
            lens, raw, _ = self.hover
            cv2.putText(canvas, f"{lens.upper()} RAW u={raw[0]} v={raw[1]}", (MARGIN + 720, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 0), 1, cv2.LINE_AA)
        help_text = "1 lines | 2 landmarks | 3 pairs | 4 circle | N/P label | U undo | C cancel pair | S save | Q/Esc quit"
        cv2.putText(canvas, help_text, (MARGIN, canvas_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        return canvas

    def refresh(self) -> None:
        cv2.imshow(WINDOW_NAME, self.render())

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        found = find_lens(x, y, self.images, self.panels)
        if event == cv2.EVENT_MOUSEMOVE:
            next_hover = None if found is None else (found[0], found[1], (x, y))
            if next_hover != self.hover:
                self.hover = next_hover
                self.refresh()
        elif event == cv2.EVENT_LBUTTONDOWN and found is not None:
            self.add_point(found[0], found[1])
            self.refresh()

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 1600, 800)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.refresh()
        print("Annotate lens-local RAW pixels. Press S to save; Q/Escape to quit.")
        while True:
            key = cv2.waitKeyEx(20)
            low = key & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if low in (ord("1"), ord("2"), ord("3"), ord("4")):
                self.mode_index = low - ord("1")
            elif low in (ord("n"), ord("N")):
                self.cycle_label(1)
            elif low in (ord("p"), ord("P")):
                self.cycle_label(-1)
            elif low in (ord("u"), ord("U")):
                self.undo()
            elif low in (ord("c"), ord("C")):
                self.pending_pair = None
            elif low in (ord("s"), ord("S")):
                atomic_save(self.out_path, self.document)
                print(f"Saved annotations to {self.out_path}")
            self.refresh()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    images, width, height = load_stacked_lenses(args.stacked)
    metadata = load_veo_metadata(args.calibration)
    document = load_or_create_document(args.out, metadata, args.stacked, width, height)
    if args.out.exists():
        print(f"Resuming annotations from {args.out}")
    AnnotationApp(images, metadata, document, args.out).run()


if __name__ == "__main__":
    main()
