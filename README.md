# VeoTranscodeLab

VeoTranscodeLab reproduces Veo's Panoramic WebGL renderer from a dual-lens
2048×2048 source transcode and its matching `camera_alignment` `.veo` metadata.
The source transcode stacks the LEFT lens in the upper half and the RIGHT lens
in the lower half. The implementation follows the captured Veo Panoramic
shader behavior; the original shader and uniforms captures are preserved in
`reference/`.

## Validated renderer

The current entry point is `analysis/reproduce_veo_panorama.py`. Pair each
transcode with the calibration from the same match:

```bash
python analysis/reproduce_veo_panorama.py \
  --video transcode-69bb54b9-7256-47d7-a4a2-4ae96b075e51.mp4 \
  --calibration data/camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo \
  --time 1694.444874 \
  --width 3840 \
  --height 2160 \
  --out-dir output/match1
```

```bash
python analysis/reproduce_veo_panorama.py \
  --video transcode-a4ff37d5-aa30-457d-86d1-8bdb5e77278f.mp4 \
  --calibration data/camera_alignment-d07ae0aa-86a6-4c57-9fa9-c6c34977b616.veo \
  --time 1081.880737 \
  --width 3840 \
  --height 2160 \
  --out-dir output/match2
```

Choose the desired panorama output resolution with `--width` and `--height`.
The current `3840×2160` setting is useful for high-quality development
renders. Results are written to the selected output directory.

`analysis/veo_metadata.py` is the small reader for the calibration metadata.
The two available matches have both been reproduced and validated with their
corresponding `.veo` files.

## Local geometry development app

`dev_app/` is a local FastAPI + React/Vite workspace for validating
panorama-to-pitch geometry. It consumes the validated Python renderer and
ground-intersection helpers without changing or duplicating their math.

Start the backend from the repository root:

```bash
.venv/bin/pip install -r dev_app/backend/requirements.txt
.venv/bin/uvicorn dev_app.backend.app.main:app --reload
```

Then start the frontend in another terminal:

```bash
cd dev_app/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. FastAPI's local API documentation is available
at `http://127.0.0.1:8000/docs`.

Create a match workspace, associate its `.veo` calibration, then add reusable
media. An existing local path is recorded by reference; browser uploads and
extracted video frames are stored under the Git-ignored `dev_app/runtime/`
directory. Supported media are JPG/PNG/WebP images and MP4/MOV/MKV/AVI/WebM
video or clips. Select a 2048×2048 stacked frame to render a 1920×1080 or
3840×2160 panorama immediately. For video, position the player at a timestamp
and use **Use current frame** before rendering.

The normal workflow is **match → media → frame → panorama → click → pitch**.
The React client only visualizes returned results: its responsive panorama
viewer maps CSS coordinates to the rendered natural pixel for the backend,
and its 2×–10× cursor magnifier uses that same source pixel. Python remains
the authority for rendering, landmark validation, and panorama-to-ground
intersection. Landmark error is a measurement only; the app does not fit,
tune, or alter the calibration.

## Repository contents and next work

- `analysis/` contains the validated renderer and metadata reader.
- `data/` contains the original `camera_alignment` calibration files.
- `samples/` contains representative source frames; local transcode MP4s may
  also be present at the repository root and are ignored by Git.
- `reference/` contains the captured Veo Panoramic shaders, uniforms, and
  reference panorama image.
- `dev_app/backend/` contains the local FastAPI API, filesystem workspace
  store, media adapters, render cache, and API schemas.
- `dev_app/frontend/` contains the Material UI React/Vite client: Library,
  Workspace, panorama viewer, SVG pitch, and validation table.
- `tests/` contains focused checks for the panorama click-to-ground helper.
- `output/` is intentionally preserved as historical visual evidence and is
  ignored by Git.

The next planned phase is reproducing Veo's Interactive renderer. Later work
will reuse the recovered geometry for player-to-pitch mapping.
