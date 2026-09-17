# VeoTranscodeLab

## Interactive field projection debug

Run the lightweight OpenCV experiment from the repository root:

```bash
python scripts/interactive_field_debug.py
```

It shows the left and right raw lens images independently above a metric
top-down football pitch. Click either lens to inspect its raw pixel and
recovered field point. A fine transient crosshair follows the mouse over the
actual lens image and displays its RAW coordinates. A nearest-neighbor RAW
pixel magnifier appears beside the current target; use the arrow keys to move
the precision cursor by one RAW pixel at a time before clicking. Press `c` to
clear, `z` to undo, or `q`/Escape to quit.

It can also split a stacked Veo frame internally:

```bash
python scripts/interactive_field_debug.py --stacked samples/stacked_frame.jpg
```

The pitch uses calibrated field length on world X and field width on world Z.
This validates the existing pixel-to-ground projection interactively; it does
not change the calibration math or Veo coordinate conventions.

## Multiple matches

Match 1 keeps the original defaults:

```bash
python scripts/test_pixel_to_field.py
python scripts/test_field_projection.py
```

Run Match 2 by supplying its calibration and lens files. The inspection
outputs are prefixed so Match 1 files are preserved:

```bash
python scripts/inspect_calibration.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --stacked samples/match2_stacked_frame.jpg \
  --prefix match2

python scripts/test_field_projection.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --left samples/match2_left_frame.jpg \
  --right samples/match2_right_frame.jpg \
  --prefix match2

python scripts/test_pixel_to_field.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo

python scripts/interactive_field_debug.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --left samples/match2_left_frame.jpg \
  --right samples/match2_right_frame.jpg \
  --lens left
```

Use `--lens right` for the right-lens interactive view.
