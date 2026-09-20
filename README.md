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

### Matches and files

Choose **New Match**, enter a name, then optionally add files in **Manage files**.
Upload Tracking, Dual lens, Calibration, clips or images, or expand **Use a local
file reference** to avoid copying a large local video. Tracking works alone;
Dual lens + matching .veo enables Panorama. No asset is required to create a match.

Library cards show the three primary resources. The menu offers Rename,
Manage files, Delete match and (for Veo imports) Resume Veo import. Manage files
supports adding, replacing, removing and changing video roles. Replacing or
removing an existing file asks for confirmation. Uploads/downloads are managed
copies; external references are unlinked without deleting their source files.

Deleting a match removes its runtime directory, including managed media,
calibration, extracted frames, cache, diagnostics and analysis artifacts.
External files and historical `output/` are never deleted. Active imports must
finish or be cancelled before file changes or deletion.

### Workspace

**Tracking | Panorama | Dual lens** share elapsed playback time. The viewer takes
roughly 70% of a large desktop layout, with the scalable pitch alongside it;
narrower screens stack. Play/pause, timeline, seconds entry and one-frame steps
remain available. Source switches pause and preserve time, clamped to the new
source duration. This does not infer offsets between differently trimmed clips.

- Tracking plays the follow-camera video. Metric clicks remain disabled because
  its time-varying camera geometry has not been validated.
- Panorama uses the unchanged captured Veo WebGL shaders with the dual-lens
  source and calibration; choose 1920×1080 or 3840×2160.
- Dual lens offers Stacked, LEFT and RIGHT. The source must be 2048×2048.

Panorama and RAW clicks still use Python geometry. Free point, landmarks,
numbered points, pitch error segments and the 180-pixel cursor magnifier remain.
The mouse wheel adjusts magnification from 2× to 10×. Measurements live in a
collapsed section after clicking. They are session-only evaluation points,
never calibration corrections. Manage files contains image preview and stacked
frame rendering; the playback toolbar retains extraction / CPU rendering.

### Persistence and analysis foundation

The existing JSON store remains authoritative:

```text
dev_app/runtime/matches/<match-id>/
    match.json
    media/          # imported tracking.mp4 and dual_lens.mp4; uploads
    calibration/    # imported camera_alignment.veo; uploads
    clips/          # available for future clip artifacts
    frames/
    cache/
    diagnostics/
    analysis/
```

Directories are created as needed; clips currently remain normal media records
with a clip role. Ownership and logical roles are stored independently of original
filenames. Matches and completed assets survive restart. Legacy workspaces remain
compatible. Use one backend worker.

`dev_app/backend/app/analysis_layers.py` defines small stage, run and artifact
contracts: stage/version, required/produced artifact types, configuration, status,
Match and source media. Detection, tracking and team-role definitions illustrate
dependencies only. There is no runner, model, fake output or unfinished analysis UI.

### Importing an authorized Veo match

1. Choose **Import from Veo** → **Connect Veo**, sign in in the dedicated browser,
   then close that browser. A saved session is not proof of current access.
2. Paste an authorized match URL and choose **Inspect**.
3. The dialog resolves exactly **Tracking**, **Dual lens**, and **Calibration**.
   All three must be found before **Import match** becomes available. If one is
   missing, use **Retry inspection**. No manual role classification is needed.
4. Import streams the three files directly to disk. Each file shows Waiting,
   Downloading, Complete, Failed or Cancelled, transferred bytes and percentage
   when the server supplies a size. Overall progress is byte-weighted when all
   sizes are known; otherwise it shows completed-file count and indeterminate
   activity. Close the dialog to continue in the background; reopen it to monitor.
5. Open the saved match. For a failed/cancelled import, retry the remaining files,
   or use **Resume Veo import** from the Library menu. Reinspection refreshes
   expired signed links and keeps the same Match and completed valid resources.

Inspection observes the default player first (`video.mp4` → Tracking), then
activates the actual `button.interactive-player` and continues listening for
`transcode-*.mp4` and valid `camera_alignment-*.veo` / `.veo` calibration.
It combines network requests/responses, resource timing, video sources and
bounded JSON responses. DOM changes alone never confirm successful activation.
Shaders, scripts, fonts and other application assets are not import choices.
Phases, selectors, evidence, sanitized URLs and screenshots belong in the
collapsed **Developer diagnostics** section.

The persistent Playwright profile is local to `dev_app/runtime/veo-profile/`.
Passwords are entered in Veo, not in this app. Signed media URLs and scoped
cookies remain private in memory; downloads use the original authenticated URLs,
never redacted diagnostic URLs. Files stream in 1 MiB chunks to partial/staging
paths and are published to Match metadata only after download and validation.

Jobs do not survive backend restart: unfinished imports are reported interrupted.
Completed assets remain; reinspect to restart the remaining files. Partial files
are not resumed byte-for-byte. Unknown-length transfers show byte counts, not
invented percentages. A real authenticated multi-gigabyte Veo import still needs
manual verification; local checks do not establish live-account success.

See [engineering report and manual checklist](docs/ENGINEERING_REPORT.md) and
[tracking geometry investigation](docs/TRACKING_GEOMETRY.md).

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
