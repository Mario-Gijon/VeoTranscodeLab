# Veo Match Lab — engineering report

## Scope and architecture

This iteration evolves the existing FastAPI + React/Vite/TypeScript/MUI app.
The filesystem Match store and local background importer remain the single
persistence/job systems. No database, external queue, dependencies, computer
vision algorithms or deployment were added. No commit or push was made.

The four protected modules are unchanged:
`analysis/reproduce_veo_panorama.py`, `analysis/veo_metadata.py`,
`analysis/panorama_ground.py`, `analysis/panorama_landmarks.py`.
The RAW geometry, captured shaders, WebGL sampling, original media/calibrations
and historical `output/` also remain unchanged.

The initial working tree contained pending Interactive instrumentation in
inspector.py, the import dialog/types, and importer tests. Those changes were
reviewed and continued. In particular, DOM-only activation confirmation was
replaced with useful post-Interactive asset evidence, screenshot paths are now
passed to activation, and final video/resource snapshots are collected after
the observation interval.

## Match lifecycle and ownership

A Match is created without mandatory files. The next dialog optionally associates
Tracking, Dual lens, Calibration, clips and images. Upload and local-reference
paths enter the same store used by Veo imports. Files can later be added, replaced,
removed or assigned another video role. Replacements are validated before the
old record/copy is removed. There is one primary Tracking and Dual lens resource;
clips/images remain collections.

New resources record `ownership: managed | external`. Older resources infer
ownership from their existing source provenance. Calibration has the same rules.
The JSON store uses atomic replacement and locks. File mutations serialize
against import creation and deletion; an active import blocks file changes.

Match deletion requires a simple dialog, not name re-entry. It recursively removes
only the validated runtime Match directory, including media, calibration, frames,
cache, diagnostics and analysis data. Metadata paths are never used as recursive
delete targets; child symlinks are not followed. External files are unlinked from
metadata only. Historical output, source MP4s and original .veo files are untouched.
Deletion is permanent for managed copies; there is no trash/undo function.

The existing layout remains:

```text
dev_app/runtime/
  matches/<match-id>/
    match.json
    media/                  # tracking.mp4, dual_lens.mp4, uploads, clips
    calibration/            # camera_alignment.veo or uploaded calibration
    frames/
    cache/
    diagnostics/
    analysis/
    clips/                  # supported category, created only when needed
  veo-inspections/
```

Logical roles do not depend on original filenames. Imported primary files use
predictable names. Original filenames are retained for display. A store reload
retains Matches; the legacy migration marker prevents deleted matches from
being recreated from the old manifest.

## Veo discovery and readiness

The authorized persistent Chromium session is reused. Inspection captures
requests and responses before navigation, waits for the default video/player,
and resolves initial `video.mp4` as Tracking. It then targets
`button.interactive-player`, hovers when necessary, checks visibility/disabled
state and center obstruction, and uses a physical pointer click. Accessible/text/menu
fallbacks remain only when the semantic button is absent; native DOM click is not
used by the production interaction path.

Observation combines request/response events, bounded JSON links, resource timing
and all video currentSrc/src values. Final snapshots are taken after the additional
post-activation wait. Calibration candidates must parse successfully through the
existing metadata reader to be importable. A complete MP4 is required for video
import; dual-lens candidates require the transcode name or 2048×2048 evidence.

For this workflow, a returned click or changing controls/canvases is not success.
Confirmation requires post-Interactive transcode, clearly identified dual-lens
media, or a parsed .veo resource. Import readiness separately requires all three
roles. Backend validation enforces exactly one resource for each role; neither
JavaScript nor streaming manifests can be submitted as normal downloads.

The normal dialog shows URL, session state and three file rows. Inspection phases,
selectors, confidence, MIME, redacted URLs, snapshots and screenshot paths are
only in collapsed Developer diagnostics. No asset IDs are hardcoded. Session
presence is described as saved, not proof of authentication.

## Download behavior and recovery

One background thread streams the three selected original authenticated URLs
using the existing scoped cookie jar. The downloader reads 1 MiB chunks into
exclusive .part files. Complete transfers are decoded/validated in staging before
being renamed and published to Match metadata. Entire MP4s are never buffered
in memory. Public API/diagnostics omit cookies and redact signed URL query values.

Each role exposes waiting/downloading/complete/failed/cancelled status, bytes and
known total. HTTP Content-Length or full Content-Range sizes provide estimates
at inspection; live download sizes take precedence. Overall percentage is
byte-weighted only when every size is known. Otherwise the UI shows an
indeterminate bar and completed-file count. No fabricated ETA or percentage.

On failure or cancellation, valid completed resources remain available. Retry
keeps the same Match, validates existing resources and skips their download.
A new inspection can refresh expired URLs. Failed/cancelled status and the last
summary are persisted. After backend restart, unfinished jobs are shown as
interrupted; fresh inspection restarts remaining files, not partial byte ranges.
A hard process exit may leave ignored staging/partial files until match deletion.

The Library refreshes while open. Background progress can be reopened from
Import from Veo or a Match's Resume Veo import menu, including after a page reload
while the backend is alive. Import jobs/credentials themselves remain in memory.

## Library and workspace

The Library has compact availability cards, New Match, Import from Veo, Open,
and contextual Rename/Manage files/Delete. A restrained dark palette uses one
mint accent. File operations live in a reusable dialog, keeping the viewer clear.

The workspace uses approximately 70% viewer / 30% pitch on desktop and stacks
on smaller screens. Tracking, Panorama and Dual lens share elapsed seconds;
switching source pauses and carries the current timestamp, clamped to duration.

- Tracking: native playback, seeking, duration and nominal frame steps. Metric
  clicks remain disabled pending validated time-varying camera geometry.
- Panorama: unchanged WebGL renderer, 1920×1080 or 3840×2160, magnifier, clicks,
  Free point/landmarks and Python metric results.
- Dual lens: stacked, LEFT or RIGHT, original RAW coordinate conversion,
  magnifier and Python metric results.

The pitch retains .veo dimensions, numbered/expected/measured points and error
segments. Measurements are collapsed until wanted. Frame extraction, CPU render,
image preview and stacked-image rendering are retained in compact controls.
Measurements remain session-only; no hidden correction or homography was added.

## Future analysis foundation

`backend/app/analysis_layers.py` supplies Pydantic contracts for stage definitions,
stage runs, analysis runs and artifacts. They describe id/version, required and
produced artifact types, configuration/status, and Match/source-media/run
ownership. Three illustrative dependency definitions cover person detection,
person tracking and team/referee role classification. They do not execute.

Future code can store run manifests/artifacts through the existing analysis
directory category. There is no workflow engine, CV model, fake output or normal
frontend exposure of unfinished stages.

## Files changed

- Backend: `main.py`, `schemas.py`, `services.py`, `storage.py`;
  `importers/veo/inspector.py`, `models.py`, `service.py`.
- New backend module: `analysis_layers.py`.
- Frontend: `App.tsx`, `main.tsx`, `api/client.ts`, `types/index.ts`;
  `pages/LibraryPage.tsx`, `WorkspacePage.tsx`;
  `components/VeoImportDialog.tsx`.
- New frontend component: `components/ManageFilesDialog.tsx`.
- Existing validation: `tests/test_match_api.py` now reflects all-three readiness;
  `tests/browser_check.py` uses new labels and optional UI screenshots.
  The pending `tests/test_veo_importer.py` changes were preserved.
- Documentation: `README.md` and this report.

## Validation performed

- Python compile checks passed.
- Existing Python suite: 45 passed, one installed Starlette/AnyIO deprecation warning.
- FastAPI startup, health and OpenAPI passed with isolated runtime storage.
- Existing Chromium smoke: all three views, shared seek, RAW/panorama clicks,
  playback and responsive pitch passed.
- CPU/WebGL comparison remained unchanged on both matches:
  mean RGB error 0.12847 / 0.09205; 99th percentile 1 / 2.
- TypeScript typecheck, ESLint, existing coordinate tests and production build passed.
  Vite reports its existing approximately 551 kB bundle-size advisory.
- Direct temporary lifecycle smoke passed creation, rename, persisted reload,
  managed removal, external preservation, import deletion guard and match deletion.
- Direct temporary importer smoke passed three-role resolution, JS exclusion,
  DOM-only activation rejection, byte-weighted progress, simulated download failure,
  same-Match retry and skipping the completed Tracking download.
- Library, workspace and Manage files screenshots were captured for local visual review.
- `git diff --check` passed; protected geometry/shader files have no diff.

No large new test suite was added. Local importer checks use simulated resource
access and do not establish successful authenticated Veo discovery or multi-GB transfer.

## Run commands

From the repository root, using the existing environment:

```bash
.venv/bin/uvicorn dev_app.backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd dev_app/frontend
npm run dev -- --host 127.0.0.1
```

Open http://127.0.0.1:5173. First-time dependency/browser setup remains in README.
Use one backend worker and keep the app local. Existing checks:

```bash
.venv/bin/python -m compileall -q analysis dev_app/backend tests
.venv/bin/python -m pytest -q
.venv/bin/python tests/browser_check.py
git diff --check
cd dev_app/frontend
npm run typecheck
npm run lint
npm test
npm run build
```

## Required manual real-Veo verification

1. Inspect an accessible/public match; the importer uses a clean unauthenticated browser context.
2. Inspect an authorized match. Verify Tracking resolves video.mp4, Dual lens
   resolves transcode-*.mp4 and Calibration resolves a valid .veo. A missing file
   must disable Import match; use Retry inspection and diagnostics when necessary.
3. Import and observe per-file transferred bytes/percentages over real large
   downloads. Cancel or interrupt once; confirm completed files remain and a
   retry fills only missing resources in the same Match.
4. Open the Match and seek/switch all three views at several times. Check
   LEFT/RIGHT crops, magnification, a landmark click and the 2D pitch.
5. Restart both servers. Confirm the Match still opens; an interrupted import
   should require reinspection, not appear Ready.
6. Exercise rename, managed replacement and reference removal. Delete a disposable
   Match and verify its managed directory disappears while the originals and
   historical output remain intact.

## Remaining limitations

Real authenticated Veo discovery and multi-gigabyte downloads were not exercised
in this environment. Lazy resources, expired authorization, profile locks or a
future player DOM change can still require investigation. The bounded inspector
does not invent endpoints or implement streaming-manifest assembly.

Jobs and signed credentials are not durable; there is no byte-range resumption
after failure/restart. No multi-worker coordination exists. Unknown-size streams
can only report transferred bytes. Common timestamps do not align trimmed media
automatically, and frame stepping uses nominal fps/browser seeking. Playback
depends on browser codecs/GPU. Tracking metric geometry and all CV algorithms
remain intentionally unavailable.
