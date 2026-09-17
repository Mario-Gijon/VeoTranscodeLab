# VeoTranscodeLab

## Interactive field projection debug

Run the lightweight OpenCV experiment from the repository root:

```bash
python scripts/interactive_field_debug.py --lens left
python scripts/interactive_field_debug.py --lens right
```

Click pixels in the selected Veo lens image to print their Veo world and
field coordinates. The companion top-down view plots each projected point,
with the calibrated field length on world X and field width on world Z.
Press `c` to clear points or `q`/Escape to quit. This validates the existing
pixel-to-ground projection interactively; it does not change the calibration
math or the Veo coordinate conventions.

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
