# Veo Match Lab — engineering report

## Continuation audit

The implementation continues the existing FastAPI/React working tree, including
the pending Veo importer, rather than replacing it with a second application.
The original renderer, metadata reader, panorama-ground and landmark modules
were inspected and retained byte-for-byte. Original calibrations, reference
captures, source frames/MP4s, and historical `output/` were not edited or removed.
No commit or push was made.

The inherited incomplete changes were reviewed for integration and safety, not
assumed correct because they compiled. Specific issues corrected include:

- Import job serialization attempted to deep-copy a `threading.Event`; public
  state now excludes synchronization objects.
- Import/download failure messages and inspection URLs could disclose signed
  tokens. All URL query values are redacted, cookies stay private, and network
  errors exposed to the client are generic. Pydantic errors omit submitted values.
- Downloads now restrict direct/redirect URLs to public HTTPS, scope cookies
  using a cookie jar, check content length, refuse overwrites, and preserve a
  pre-existing `.part` file. Failure removes only this invocation's partial file.
- Concurrent match updates are locked; legacy metadata is migrated once without
  moving original assets. Downloaded assets now join normal Match persistence.
- Manual roles, calibration version reporting, stale Library state, native
  media Range serving, and import-job polling/review were integrated.
- Shared seeking and RAW crop coordinates are exercised in Chromium, rather
  than assuming the three viewers' individual implementations integrate.
- Real HTTP geometry tests found a NumPy boolean serialization failure missed
  by direct-handler tests; the new RAW adapter now returns a Python boolean.
- The previous root-only TypeScript command did not check referenced projects;
  `typecheck` now uses `tsc -b` and uncovered/fixed actual interface mismatches.
- Still-image previews disable video controls, frame observations retain their
  extraction timestamp, stale click replies are ignored after context changes,
  and both magnifiers use non-passive wheel listeners.

There is no reliable per-edit model attribution in the working tree; these are
concrete findings from the continuation audit, not guesses about which model
authored a particular line.

## 1. Architecture

One FastAPI process serves Match metadata, source media, uploads, extracted
frames, CPU-rendered frames, authoritative geometry, and local import jobs.
React/Vite/TypeScript/MUI remains the only frontend. Browser-native video drives
Tracking and RAW views; a WebGL video texture drives Panoramic visualization.
Python supplies shader uniforms and all metric geometry results.

Veo-specific code is isolated in `backend/app/importers/veo/`. Existing route,
schema, media-service and filesystem-store modules are evolved in place. No
database, queue service, frontend framework replacement or fitting subsystem
was introduced.

## 2. Final repository structure

Generated/ignored directories are abbreviated below; original assets remain.

```text
VeoTranscode/
├── README.md
├── .gitignore
├── analysis/
│   ├── reproduce_veo_panorama.py     # unchanged validated renderer
│   ├── veo_metadata.py              # unchanged reader
│   ├── panorama_ground.py           # unchanged panorama geometry
│   ├── panorama_landmarks.py        # unchanged landmark references
│   └── raw_ground.py                # inverse of existing forward sampling
├── data/                            # both original camera_alignment .veo files
├── samples/                         # all existing source/calibration frames
├── reference/
│   ├── veo_panoramic.vert.glsl
│   ├── veo_panoramic.frag.glsl
│   ├── veo_panoramic_uniforms.json
│   └── veo_panoramic_reference.png
├── dev_app/
│   ├── backend/
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   └── app/
│   │       ├── main.py, schemas.py, services.py, storage.py
│   │       ├── view_geometry.py
│   │       └── importers/veo/
│   │           ├── browser.py, inspector.py, classifier.py
│   │           └── downloader.py, models.py, service.py
│   ├── frontend/
│   │   ├── package.json, package-lock.json, Vite/TS/ESLint configuration
│   │   ├── src/
│   │   │   ├── App.tsx, main.tsx, api/client.ts, types/index.ts
│   │   │   ├── pages/LibraryPage.tsx, WorkspacePage.tsx
│   │   │   └── components/
│   │   │       ├── MatchViewer.tsx, panoramaGL.ts, viewCoordinates.ts
│   │   │       ├── PanoramaViewer.tsx, VeoImportDialog.tsx
│   │   │       └── PitchSvg.tsx, ValidationTable.tsx
│   │   └── tests/coordinates.test.mjs
│   └── runtime/                     # ignored; matches, sessions, diagnostics
├── docs/ENGINEERING_REPORT.md, TRACKING_GEOMETRY.md
├── tests/
│   ├── test_panorama_ground.py, test_panorama_landmarks.py
│   ├── test_raw_ground.py, test_match_storage.py
│   ├── test_dev_app_api.py, test_match_api.py
│   ├── test_veo_importer.py, test_veo_downloads.py
│   └── browser_check.py
├── output/                          # ignored, historical evidence preserved
└── local *.mp4                      # ignored, unchanged
```

## 3. Match persistence model

`runtime/matches/<match-id>/match.json` is authoritative. Each Match has name,
description, timestamps, calibration, media records, metadata and a future
artifacts slot. Media records include stable ID, source provenance, path, role,
dimensions, timing and size. Subdirectories `calibration`, `media`, `frames`,
`cache`, `diagnostics` are created as needed. Clips use a media role; no separate
storage hierarchy is required for playback.

Writes replace a temporary JSON file atomically and updates use an RLock.
This is a single-process store, not multi-worker storage. A `.migrated` marker
prevents repeated legacy imports. `workspaces.json` and its referenced files
remain intact for recovery. Context/render registries, live observations and
job state are memory-only; saved matches/assets survive process restart.

## 4. Manual import workflow

Create a Match with any subset of assets. Choose a repository calibration,
upload `.veo`, or reference a local `.veo` path. Upload media or reference paths;
choose Tracking, Dual lenses, Clip or Other/image and adjust video roles later.
OpenCV validates decoding and reports resolution/fps/duration; records show size
and original filename. Dual-lens roles require 2048×2048.

Referenced local files are not copied; uploads retain original bytes in the
Match directory. Frame extraction creates a new derived image and records its
actual decoded frame time/source. Ordinary images preview directly; stacked
images can use the unchanged CPU panorama renderer. Partial-asset Matches work:
Tracking requires no calibration, while panorama needs dual source + calibration.

## 5. Veo URL import workflow

Library → Import from Veo → Connect Veo → user login → close Connect browser →
Inspect authorized HTTPS Veo URL → review candidates → assign/select roles →
Import match. Nothing selects an ambiguous video silently. The review includes
filename, inferred role/confidence/evidence, MIME and available dimensions,
fps/duration/calibration dimensions/version. Manifests remain evidence, not
downloadable video selections. The job can be cancelled and its partial Match
opened after completion/failure; Library reload sees imported assets.

Automated tests mock Veo network/session responses. A real-account end-to-end
import remains a manual integration check, not a claimed successful live test.

## 6. Browser/session handling

Playwright launches a dedicated persistent Chromium profile under ignored
`runtime/veo-profile/`, mode 0700. Connect is a headed subprocess for normal
user authentication. No password input exists in the app; normal desktop
browser profiles are not read. Inspection reuses this profile headlessly only
after the Connect browser closes. Session state reports browser availability,
profile presence, running connection and readable launch errors; profile
presence is not represented as proof of authentication.

Disconnect blocks during active import, closes the owned process, deletes only
the dedicated profile and clears retained inspection credentials. Cookies are
scoped by host/path/secure rules for HTTP fetches and never returned in JSON.
Full signed URLs are kept only in memory while needed; diagnostics redact all
query values and fragments. Do not commit runtime data or share browser profiles.

## 7. Asset discovery/classification

The inspector observes successful browser network responses and nested resource
URLs actually returned in bounded JSON, not guessed IDs/private endpoints.
It combines filename/path, MIME, and optional 2048×2048 probe evidence.
Known sample IDs are not embedded in importer logic.

Inspection limits: roughly nine seconds of page observation, 100 resources,
JSON bodies with declared length ≤2 MB, calibration reads ≤2 MB, and video
prefix probes ≤8 MiB on up to eight early resource positions. `ffprobe` operates
on a temporary prefix, so signed URLs never enter its process arguments. A
trailing MP4 `moov` atom or absent ffprobe leaves metadata unknown. Metadata is
conservative: only explicit scalar match title/date/team fields are retained.
Resources requested only after a player mode changes may not be discovered.

## 8. Download architecture

One local daemon worker thread imports at a time. Jobs expose ID, state, current
asset, bytes/total when known, completed assets, progress, error and Match ID.
Polling continues using a session-stored job ID while that backend is alive.
Progress combines completed assets with the active asset fraction; it is not
an estimate of total download bytes when all sizes are unknown.

Downloads stream in 1 MiB chunks through scoped authentication cookies to an
exclusive `.part`, verify declared size, and rename on success. Only public
HTTPS direct and redirect URLs are accepted. Complete original files are never
held in RAM. Cancellation is checked between reads (network timeout 20 s);
completed files remain. Existing destination/partial files are not overwritten
or deleted. An abrupt process exit can leave recoverable partials; there is no
restart-resume engine. No Redis/Celery/external service is required.

## 9. Match Library

The Library lists persisted matches with Tracking, Dual-lens, Calibration and
Frames/clips availability, New match, Import from Veo and Open workspace.
Returning from a workspace refreshes server state. Empty/partial Matches are
valid. Styling is restrained dark MUI without gradients or decorative dashboards.

## 10. Match Workspace

The main area prioritizes a large media viewer and persistent SVG pitch, using
roughly 68/32 width on desktop and stacking on smaller screens. There is no
narrow fixed maximum width on ultrawide displays. Assets/calibration sit in a
compact accordion. The view toggle, source selector, timeline, diagnostics and
landmark controls form one workflow. Readable errors and a CPU still-render
fallback remain available if the browser codec/WebGL cannot display a source.

The pitch uses loaded L/W, real aspect ratio and original football markings.
Blue expected diamonds and numbered measured markers are joined by error
segments. The table retains pixel/UV, view/time, expected/measured X/Z, deltas,
Euclidean error and inside/outside/no-ground status. Aggregate diagnostics show
mean, median, RMS, maximum and signed mean ΔX/ΔZ. They never change calibration.

## 11. Shared playback/timeline

Workspace owns one elapsed-seconds timestamp. Panoramic/RAW modes keep the same
video element; switching to another source mounts its video at the common time,
clamped to its duration. Source switches pause; timeline has play/pause, numeric
time, seek slider and ±1 frame using fps when available. Native time events
update state without pushing per-frame React seek loops. Clicks pause before
measurement, and clicks during an unfinished seek are rejected.

This is approximate shared elapsed time, not proven synchronization of unrelated
files. Clip offsets are not inferred. Frame stepping uses nominal fps and browser
seeking, not a frame-accurate NLE or variable-frame-rate indexing engine.

## 12. Tracking viewer

Normal HTML video playback, HTTP Range support and the shared timeline. Tracking
can play without geometry. Metric clicks are disabled with the explicit missing
time-varying-camera explanation; the API cannot be asked to approximate them.

## 13. Dual-lens viewer

The original 2048×2048 video is decoded once and drawn as stacked or exact
2048×1024 LEFT/RIGHT crops. LEFT is top, RIGHT is bottom. A RIGHT crop click adds
1024 to natural y before calling Python. No resampling is written to disk and
the source video is unchanged. Magnifier and markers share these coordinates.

## 14. Panoramic/WebGL viewer

The actual captured vertex and fragment files are delivered unchanged to
WebGL1. Python computes the uniforms from metadata, including the validated
inverse `camera2presentation`, and column-major matrix serialization. Native
HTML video frames become the stacked texture; LINEAR/CLAMP sampling and the
source-y convention follow the captured shader. Watermark texture is transparent
for the geometry comparison. The renderer is not rebuilt on measurement clicks;
video texture updates occur only when decoded playback time changes.

CPU/WebGL pixel comparisons used identical decoded sample pixels supplied as
PNG, at 640×360 with Chromium/SwiftShader:

| Calibration | Mean absolute RGB difference (0–255) | 99th percentile |
| --- | ---: | ---: |
| `67a3f0b3…` / match 1 | 0.112483 | 1.0 |
| `d07ae0aa…` / match 2 | 0.071924 | 1.0 |

This validates the shader integration on both samples, not performance on every
GPU or every frame of a match. Native 1920 preview playback/seek was exercised;
3840×2160 is supported but its sustained frame rate depends on the local GPU.
The unchanged CPU renderer remains a still-frame baseline.

## 15. Panorama click → pitch

Browser CSS coordinates are explicitly floored/clamped to canvas natural
pixels. Python's existing helper uses pixel-center UV, the captured panorama
angular bounds, the effective camera transform, and Y=0 ground intersection.
No camera math moved to frontend hit testing.

The 180×180 magnifier samples the same rendered canvas/image centered at
`(x+0.5, y+0.5)`; its crosshair and request therefore reference the same texel at
any responsive scale. Wheel zoom is 2×–10× with scrolling prevented only over
the viewer. It does not intercept clicks. Numbered markers survive multiple
measurements; stale async replies are discarded after clearing/changing context.

## 16. RAW dual-lens click → pitch

`analysis/raw_ground.py` reverses the existing source sampling: stacked texel
center → lens selection → inverse piecewise vertical source compression →
3840×2160 calibration pixel → inversion of the canonical rational distortion
function → calibrated lens ray → `camera2world @ lens2camera` → Y=0 intersection.
The optical lens-origin translation is retained. Damped Newton iteration solves
the existing forward model with a checked calibration-pixel residual; it is not
parameter fitting or a new calibration. Nonconvergence is a readable error.

Tests round-trip known field landmarks through the original forward projector
for both lenses of both matches (tolerance 0.00006 m including float32 forward
map precision), and compare panorama ground points projected into RAW and
inverted (0.0001 m tolerance). These are numerical consistency checks, not a
claim of submillimetre physical accuracy from manual clicks.

## 17. Tracking click → pitch findings

See [TRACKING_GEOMETRY.md](TRACKING_GEOMETRY.md) for probes and missing evidence.
The available static `.veo` transforms and Panoramic shader captures do not give
the tracking video's time-indexed virtual-camera pose, FOV/zoom/crop or timebase.
Local encoded videos have no discovered camera-data stream. No authenticated
network investigation was possible without a user session. The correct current
state is working playback with disabled metric clicks, not a fabricated map.

## 18. Tests and validation

- Python compilation: all source files under `analysis`, backend and `tests`.
- Full pytest: geometry/landmarks, both-match RAW inversion, migration/concurrent
  persistence, manual imports, Range serving, classification/redaction,
  download integrity/cookie scope/cancellation, job serialization, real HTTP
  API and mocked Veo routes. Existing tests retained.
- Chromium check: isolated FastAPI startup/health/OpenAPI, two-match CPU/WebGL
  comparison, Library/roles, shared 1.2-second seek across all views, RIGHT RAW
  and panorama clicks, playback, responsive pitch and no page exceptions.
- Frontend: actual TypeScript project build-check, zero-warning lint, focused
  responsive coordinate/crop tests, production build.
- `git diff --check`; no diff in the four protected modules or original assets.

Final run: compilation passed; **39 pytest tests passed**; FastAPI startup and
OpenAPI passed; Chromium parity/workflow passed; TypeScript, lint, coordinate
assertions and production build passed; tracked and new-file whitespace checks
passed. SHA-256 verification and Git diff confirm all four protected modules
are unchanged. Non-fatal tooling warnings: installed Starlette/AnyIO deprecation
and Vite's approximately 548 kB main-chunk advisory. No live Veo credentials or
servers are required by tests.

### Authorized live-account manual integration checklist

1. Run both servers locally. Install Playwright Chromium, with a working desktop
   display for Connect. Keep runtime private and Git-ignored.
2. Connect, authenticate yourself, close Chromium, reopen/reuse Connect to verify
   the dedicated session persists. Never paste passwords into this app.
3. Inspect a permitted match URL. Check listed resource evidence against the
   browser's player, especially 2048×2048 dual source versus tracking. Missing
   metadata must show unknown, and ambiguous videos must require selection.
4. Review/select `.veo`, dual source and tracking if available. Start import;
   observe bytes/status without buffering full files. Cancel one large import:
   current partial should be removed, completed assets should remain usable.
5. Reinspect and import successfully. Verify calibration dimensions/version,
   media metadata, Library badges and persisted Match JSON after restart.
6. Scrub/switch Tracking, LEFT, RIGHT and Panoramic at several timestamps.
   Compare a CPU still to WebGL, and measure visible known landmarks in RAW and
   panorama. Do not interpret Tracking playback as supported ground mapping.
7. Inspect saved diagnostics for redacted query values and no cookies/headers.
   Expired/inaccessible resources must fail readably with completed files kept.
8. Disconnect when no job is active; verify dedicated profile removal and no
   change to the user's regular browser profile or original assets.

## 19. Dependencies

Backend keeps FastAPI, Uvicorn and python-multipart. Playwright provides the
authorized browser session; NumPy/OpenCV are declared explicitly for the existing
renderer/media adapters. Development requirements add pytest and httpx2 for the
installed Starlette TestClient path. The local Chromium integration test uses
system ffmpeg; optional ffprobe enriches bounded inspection metadata. Browser
installation is separate from pip dependencies.

No new npm library, database, broker or frontend test framework was introduced.
Node's built-in test runner/type stripping runs the coordinate assertions.

## 20. Exact commands

From the repository root (create `.venv` only if absent):

```bash
python -m venv .venv
.venv/bin/pip install -r dev_app/backend/requirements-dev.txt
.venv/bin/playwright install chromium
.venv/bin/uvicorn dev_app.backend.app.main:app --host 127.0.0.1 --port 8000
```

Second terminal:

```bash
cd dev_app/frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. Use one backend process/worker. For custom origins
set `VITE_API_BASE_URL` and backend `VEO_FRONTEND_ORIGINS` explicitly; tests use
an isolated `VEO_RUNTIME_ROOT`. Defaults remain the normal local runtime.

Validation from root:

```bash
.venv/bin/python -m compileall -q analysis dev_app/backend tests
.venv/bin/python -m pytest -q
.venv/bin/python tests/browser_check.py
git diff --check
```

Validation from `dev_app/frontend`:

```bash
npm run typecheck
npm run lint
npm test
npm run build
```

The browser check uses temporary storage and localhost ports 8005/5175; leave
those ports free. It requires the repository samples/calibrations and ffmpeg.
The original CLI renderer commands for both matches remain in README.

## 21. Known limitations

- Real authenticated Veo discovery/download is not live-verified. Bounded passive
  observation can miss lazy resources or JSON without Content-Length; unfamiliar
  API/header-based authorization may require a future observed integration.
- Manifests are not assembled into downloads; only discovered complete files
  are supported. Unknown metadata is not invented.
- Jobs/inspection/context IDs do not survive backend restart. Imported assets
  and Match JSON do. There is no cross-process lock or automatic job resume.
- Metric observations are in-memory, not a durable analysis export yet.
- Native playback depends on codec/GPU support; 4K performance is not benchmarked.
  WebGL uses preserved drawing buffers for precision inspection.
- Common elapsed seconds do not automatically align trimmed clips or different
  encodings. Browser frame stepping is not a frame-accurate editing system.
- RAW inverse behavior outside convergent usable lens regions returns an error;
  tested numerical round trips do not replace physical calibration validation.
- Tracking ground mapping remains disabled pending the missing trajectory.
- This is a trusted single-user localhost tool, not a hardened public service.
  It intentionally supports local filesystem paths and persistent credentials;
  do not expose it remotely. URL checks are defense in depth, not a replacement
  for a network sandbox against hostile pages/DNS behavior.

## 22. Recommended next step

Run the documented authorized real-match import checklist, then capture and
validate time-indexed Interactive/follow-camera metadata. Preserve the current
two-match renderer/RAW numerical baselines while investigating that evidence.
Only after exact tracking geometry is demonstrated should Tracking metric clicks
be enabled. Player detection, identity, tracking, tactical analytics and fitting
remain explicitly out of scope.
