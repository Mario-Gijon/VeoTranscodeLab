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
