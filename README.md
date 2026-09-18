# VeoTranscodeLab

VeoTranscodeLab is a focused research workspace for building a fixed-camera,
football-pitch calibration from a manually selected frame. The target is a
shared metric pitch coordinate system for the two RAW Veo lens images.

The current repository intentionally does not provide a player-positioning
pipeline or treat Veo's end-to-end optical projection as metric ground truth.
The original `.veo` files are retained as initialization and metadata: field
dimensions, selected-corner anchors, lens transforms, intrinsic/distortion
metadata, and presentation data for the panorama visualisation.

## Layout

- `analysis/veo_metadata.py` — minimal `.veo` metadata reader; it does not
  project pixels or fit geometry.
- `data/` — original Veo calibration files.
- `samples/` — genuine RAW lens and stacked calibration frames.
- `experiments/panorama/build_panorama.py` — retained approximate LEFT/RIGHT
  panorama and blend visualisation.
- `docs/calibration-roadmap.md` — architecture, constraints, and established
  conclusions for the next calibration phase.

Generated files belong in `output/`, which is ignored by Git.

## Manual calibration annotations

Use the OpenCV annotation tool to collect precise, lens-local RAW pixels from
one stacked frame. It only records observations; it performs no projection,
distortion correction, or calibration.

```bash
python tools/annotate_field_calibration.py \
  --stacked samples/match2_stacked_frame.jpg \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --out annotations/match2.json
```

The viewer shows both lenses, a cursor crosshair, lens-local RAW coordinates,
and a nearest-neighbour magnifier, plus a persistent 2D Spanish-labelled pitch
minimap. The minimap highlights the currently selected line or landmark and
shows the stable orientation: far side at the top, near side at the bottom.
Saved coordinates are integer `(u, v)` RAW
pixels: the RIGHT lens always uses its local `0..lens_height-1` vertical range,
never the stacked-frame Y coordinate. Existing `--out` files are resumed.

Modes: `1` straight-line samples, `2` named landmarks, `3` LEFT/RIGHT
correspondences, and `4` centre-circle samples. Use `N`/`P` to cycle line or
landmark labels, `U` to undo the most recent session edit, `C` to cancel a
pending correspondence, `S` to save, and `Q`/Escape to quit. Correspondences
require one point from each lens before a `pair_###` record is created. The
header shows both the stable English identifier and a human-friendly Spanish
target name; JSON identifiers remain unchanged.

## Veo RAW rectification mapping search

`analysis/refine_veo_rectification.py` searches the RAW-to-Veo
calibration-coordinate mapping while keeping the `.veo` intrinsics and
distortion coefficients fixed. It tests uniform scale/offset, anisotropic
scale/offset, and larger-source-canvas crop hypotheses. Manual straight-line
annotations are only an evaluation ruler: they score rectified line
straightness and never define an image warp or homography.

```bash
python analysis/refine_veo_rectification.py \
  --stacked samples/match2_stacked_frame.jpg \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --annotations annotations/match2_calibration.json \
  --out-dir output/refine_veo_rectification_match2
```

The output directory contains the best rectified lens images, transformed
annotation overlays, valid masks, a tested-family comparison panel,
`report.json`, and `ranking.csv`.

## Interactive Veo rectification tuning

`tools/tune_veo_rectification.py` is a local OpenCV HighGUI utility for
visually tuning how a stacked RAW frame is mapped into the coordinate space
described by the `.veo` intrinsics and distortion. The mapping scale and
offset are shared by LEFT and RIGHT; small per-lens principal-point deltas,
rectified horizontal FOV, and a radial distortion gain are exposed as
trackbars. This is a practical exploration tool, not a calibration solver.

```bash
python tools/tune_veo_rectification.py \
  --stacked samples/match2_stacked_frame.jpg \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --annotations annotations/match2_calibration.json \
  --view both \
  --out output/tune_veo_rectification_match2
```

Controls: `S` saves the current settings, rectified images, and annotation
summary; `W` writes only the current rectified images; `R` resets all
trackbars; `Q` or `Escape` quits. The optional annotation file is read-only:
its line points are drawn on the previews and used only as a straightness
score, never to fit or force a warp. Invalid or missing annotations produce a
readable warning and the rectification preview remains usable.

The trackbars mean: `scale x/y` map RAW pixels into the `.veo` calibration
canvas; `offset x/y` shift that shared mapping; `HFOV deg` controls the
virtual rectified camera view; `dist gain x100` scales only the radial Veo
distortion terms; and the LEFT/RIGHT `pp dx/dy` controls make small
per-lens principal-point adjustments around the `.veo` values.

## Streamlit development lab

Install the small local UI dependency, then start the Lens configuration page:

```bash
pip install streamlit
streamlit run dev_app/app.py
```

The page accepts an uploaded stacked JPG/PNG or a repository-local stacked
frame path, and either a bundled or uploaded `.veo` calibration file. Select
LEFT or RIGHT to inspect its RAW and rectified images side by side at their
native aspect ratio. Its controls use the same defaults and OpenCV
rectification helpers as `tools/tune_veo_rectification.py`: shared
RAW-to-calibration mapping scale/offset, radial distortion gain, selected
lens principal-point shifts, and virtual HFOV. `Save configuration` and `Save
rectified image` write development artifacts under ignored `output/dev_app/`.

## Retained panorama visualisation

From the repository root, run:

```bash
python experiments/panorama/build_panorama.py
```

It writes the LEFT/RIGHT warps, masks, hard selection, and blended panorama to
`output/panorama/`. This is a useful visual aid for the approximate lens
relationship; it is not a field-calibration result.

## Next direction

Future calibration will derive one calibration per fixed-camera match from a
manually annotated frame. It will combine painted-line constraints, known pitch
landmarks, LEFT/RIGHT same-point correspondences, and later circle constraints.
See [the calibration roadmap](docs/calibration-roadmap.md) for the intended
coordinate flow and the decisions behind it.
