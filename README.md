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

## Veo Match Lab

`dev_app/` is the local FastAPI + React/Vite/TypeScript + Material UI match
workspace. Native video playback drives Tracking and Dual lenses views;
Panoramic playback feeds that same dual-lens video to the captured Veo WebGL
shaders. Python remains authoritative for all metric click-to-pitch results.
The four validated renderer/metadata/ground/landmark modules are unchanged.

Start from the repository root (Python with NumPy/OpenCV wheels and Node
22.18+ are recommended; the frontend coordinate tests use Node type stripping):

```bash
python -m venv .venv  # only if a virtual environment does not already exist
.venv/bin/pip install -r dev_app/backend/requirements-dev.txt
.venv/bin/playwright install chromium
.venv/bin/uvicorn dev_app.backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd dev_app/frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173>; API docs are at
<http://127.0.0.1:8000/docs>. Use one backend worker. This is a trusted local
tool with filesystem-path import, not an authenticated Internet-facing service.
Do not expose either server to untrusted networks.

### Manual workflow

1. In the Match Library, choose **New match**. Assets are optional.
2. Open **Assets & calibration**. Upload media or reference a local path;
   assign videos as **Dual lenses**, **Tracking**, or **Clip**. Select/upload
   the matching `.veo`. Names, dimensions, fps, duration, size, calibration
   version and field dimensions are shown. Existing media roles are editable.
3. Switch between **Tracking**, **Panoramic**, and **Dual lenses**. The shared
   timeline preserves elapsed seconds across views, with play/pause, frame
   stepping and time entry. Dual lenses offers stacked, LEFT and RIGHT crops.
4. Panoramic playback supports 1920×1080 and 3840×2160. Click pauses playback
   and measures the selected pixel. The 180-pixel magnifier samples that same
   canvas; wheel zoom is 2×–10×. Use Free point or a selected landmark.
5. The SVG pitch and table show numbered measurements, expected landmarks,
   connecting error segments, status, deltas and aggregate error statistics.
   **Clear points** clears these in-memory observations; they are not calibration
   corrections and are not saved as match assets.

Dual-lens video must be 2048×2048. Tracking can be used without calibration;
panorama and RAW metric clicks need calibration. Images can be previewed, and
2048×2048 stacked images can use **Render frame**. **Extract frame / CPU render**
retains the existing Python still-render workflow; it never processes an entire
match just to play it. Browser codec support may be narrower than OpenCV's
MP4/MOV/MKV/AVI/WebM support; H.264 MP4 is recommended for playback.

Tracking click-to-pitch is deliberately disabled: available captures and local
videos do not supply a validated time-varying virtual-camera pose, FOV and crop.
See the [tracking investigation](docs/TRACKING_GEOMETRY.md).

### Persistence

Matches live in `dev_app/runtime/matches/<match-id>/match.json`, with sibling
`calibration/`, `media/`, `frames/`, `cache/` and `diagnostics/` directories
created as needed. Uploads/downloads preserve original bytes; local paths are
referenced, not copied. Legacy `runtime/workspaces.json` is migrated once
without moving assets or deleting the recovery manifest. Refreshing the Library
reloads match state. Historical `output/` is never used by the app.

### Importing an authorized Veo match

The Library's **Import from Veo** flow uses a separate persistent Playwright
profile at `dev_app/runtime/veo-profile/`; it never reads cookies from Chrome
or Brave. Install the optional browser support once:

```bash
.venv/bin/pip install -r dev_app/backend/requirements.txt
.venv/bin/playwright install chromium
```

Choose **Connect Veo**, log in yourself in the headed browser, then close that
browser and paste an authorized Veo match URL into **Import from Veo**. The
inspection observes player network resources rather than relying on static HTML
scraping. Review discovered files, evidence/confidence and available metadata;
explicitly assign ambiguous videos or leave them skipped. Streaming manifests
are shown as evidence, not downloaded as fake MP4 files. Metadata inspection can
fetch bounded file prefixes, but full assets download only after **Import match**.

Downloads stream to the new match's runtime directory. The dialog displays job
status, current asset, bytes, available progress and cancellation. Completed
assets survive cancellation/failure. Jobs do not resume after backend restart;
reinspect expired URLs before retrying. **Disconnect Veo** removes the dedicated
profile after active work is finished/cancelled. No passwords are requested by
this app, and normal browser profiles are never read. Cookies and full signed
URLs stay out of public diagnostics. Profile/session data stays Git-ignored.

Live-account importing is implemented but has not been verified against a real
authorized session in this task. Lazily requested assets may be absent from an
inspection. See the [manual integration checklist and engineering report](docs/ENGINEERING_REPORT.md).

### Validation

From the repository root:

```bash
.venv/bin/python -m compileall -q analysis dev_app/backend tests
.venv/bin/python -m pytest -q
.venv/bin/python tests/browser_check.py
git diff --check
```

From `dev_app/frontend/`:

```bash
npm run typecheck
npm run lint
npm test
npm run build
```

The standalone browser check requires Chromium, system `ffmpeg`, sample frames
and both calibrations. It starts isolated localhost servers on 8005/5175,
checks FastAPI startup/OpenAPI, compares WebGL to Python for both matches, and
exercises workspace playback/measurement. It uses temporary storage, not your
matches. `ffprobe` is optional for bounded Veo video metadata probing.

## Repository contents and next work

- `analysis/` contains the validated renderer, metadata, panorama geometry,
  landmark references, and the RAW inverse geometry adapter.
- `data/` contains the original `camera_alignment` calibration files.
- `samples/` contains representative source frames; local transcode MP4s may
  also be present at the repository root and are ignored by Git.
- `reference/` contains the captured Veo Panoramic shaders, uniforms, and
  reference panorama image.
- `dev_app/backend/` contains the local FastAPI API, filesystem workspace
  store, media adapters, render cache, and API schemas.
- `dev_app/frontend/` contains the Material UI React/Vite client: Library,
  Workspace, panorama viewer, SVG pitch, and validation table.
- `tests/` contains geometry, persistence, import, download, HTTP API and local
  Chromium integration checks.
- `docs/` contains the full engineering report and tracking-geometry findings.
- `output/` is intentionally preserved as historical visual evidence and is
  ignored by Git.

The next step is an authorized end-to-end Veo import verification and capture
of Interactive/tracking camera metadata. Later work can reuse the proven
geometry for player-to-pitch mapping; no player detection or tracking is added.
