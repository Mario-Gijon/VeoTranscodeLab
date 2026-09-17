# VeoTranscodeLab

An experimental lab for inspecting Veo calibration, projection, and panorama
behavior. The repository is organized by purpose:

- `analysis/` — reserved for future core geometry, calibration, optimization,
  and analysis algorithms.
- `tools/` — interactive development utilities.
- `diagnostics/` — calibration, projection, and undistortion inspection
  scripts.
- `experiments/panorama/` — experimental panorama and stitching scripts.
- `data/` — Veo calibration files.
- `samples/` — input frames and inspection images.
- `output/` — generated diagnostic and panorama output.

## Interactive field projection debug

Run the dual-lens viewer from the repository root:

```bash
python tools/interactive_field_debug.py
```

It displays independent left and right lens images above a metric top-down
pitch. Hovering over a lens shows a precision crosshair, RAW coordinate label,
and nearest-neighbor magnifier. Use the arrow keys for one-RAW-pixel
adjustments before clicking. Press `c` to clear, `z` to undo, or `q`/Escape to
quit.

Validation Mode

Press `v` to toggle Validation Mode. The UI shows `VALIDATION ON`, the selected
landmark, and its expected `X,Z` coordinate. Use `n`/`p` to move to the
next/previous landmark, then click the exact landmark location in either RAW
lens panel using the magnifier. A click creates a temporary candidate; its RAW
pixel, canonical recovered coordinate, expected coordinate, error, image
marker, and pitch segment are shown immediately. Use the arrow keys to adjust
the candidate by one RAW pixel at a time, then press `Enter` to confirm it.
Only confirmed candidates are recorded and printed. Press `s` to save all
confirmed observations as JSON; provide the output path when starting the
viewer:

```bash
python tools/interactive_field_debug.py \
  --validation-out output/validation/match2.json
```

Pressing `n`/`p` clears an unconfirmed candidate before changing landmarks.
Validation observations are append-only, so repeated selections of the same
landmark and lens are retained. The on-screen help is:
`VALIDATION | N/P landmark | click candidate | arrows adjust | ENTER confirm | S save | V exit`.

To split a stacked Veo frame internally:

```bash
python tools/interactive_field_debug.py \
  --stacked samples/stacked_frame.jpg
```

## Diagnostics

Match 1 uses the original default paths:

```bash
python diagnostics/inspect_calibration.py
python diagnostics/test_field_projection.py
python diagnostics/test_pixel_to_field.py
python diagnostics/test_undistortion.py
```

For another recording, provide its calibration and input paths. For example,
Match 2 inspection and projection validation are:

```bash
python diagnostics/inspect_calibration.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --stacked samples/match2_stacked_frame.jpg \
  --prefix match2

python diagnostics/test_field_projection.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --left samples/match2_left_frame.jpg \
  --right samples/match2_right_frame.jpg \
  --prefix match2

python diagnostics/test_pixel_to_field.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo
```

The Match 2 interactive viewer can use either lens explicitly:

```bash
python tools/interactive_field_debug.py \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --left samples/match2_left_frame.jpg \
  --right samples/match2_right_frame.jpg \
  --lens left
```

Use `--lens right` for the right-lens compatibility option. The viewer always
shows both independent lens panels.

## Panorama experiments

```bash
python experiments/panorama/build_panorama.py
python experiments/panorama/build_ground_aware_panorama.py
```

These remain experimental and separate from the reserved `analysis/` area.
