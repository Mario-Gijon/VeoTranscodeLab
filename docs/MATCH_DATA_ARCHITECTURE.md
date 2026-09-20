# Match datasets, events and analysis artifacts

## Boundaries

The previous app stored media in match records, bounded Veo evidence in diagnostic
manifests, and unused illustrative analysis contracts. Dataset payloads had no
durable domain/API/viewer boundary. This foundation adds that boundary without
changing source selection, four-view geometry, pitch points or landmark behavior.

- Core assets: `tracking`, `dual_lens`, calibration and `camera_directions`.
  Tracking remains the initial ready visible Veo player's `currentSrc`;
  directions remain paired to that exact normalized URL. This layer never selects
  media, including when replaying a legacy standalone capture's asset inventory.
- Optional media: an explicitly identified or previously assigned `veo_highlights`
  reel still uses the existing separate video dialog. Individual clips are not reels.
- Datasets: `highlights`, `events`, `lineup`, `roster`, `player_tracking`,
  `player_moments`, `heatmaps`, `physical_metrics`, `match_report`.
- Veo permissions/SDK flags remain diagnostic **platform support**. Match data
  capabilities derive from stored payloads/normalized events/published artifacts,
  never from the stage catalogue or permissions.

## Capture and import

`StructuredDiscovery.observe_json()` is shared by the live app inspector and
`replay_capture()` used by the standalone discovery tool/offline dataset importer.
The diagnostic manifest is still bounded. Separately, complete sanitized payloads
are written under the capture's `datasets/` directory (no 200-row truncation).
Existing HTTP capture limits still apply: 2 MB in the inspector, 5 MiB in the
standalone recorder. Unreadable/oversized successful bodies are `not_observed`,
not empty. The byte-limited capture is not intended for future bulk tracking.

Dataset statuses: `observed`, `empty`, `forbidden` (401/403), `unavailable` (404),
`not_observed`. A missing body is not proof of an empty dataset. Endpoint identity,
source match slug, phase, observation UTC time, capture context, HTTP status,
payload-observed flag and schema/parser version are retained internally.
Only observations for the exact workspace Veo match are published. Importing a
capture into an unrelated match is rejected. This does not assert that unrelated
API responses or asset URLs belong to the inspected match.

`VeoImportService` persists small datasets before downloading selected media, so
an interrupted video download does not lose the dataset. Retrying observations
is idempotent. History is retained; the latest successful snapshot is active.
Failed reinspection does not erase a previous successful snapshot. A later
successful empty response does clear its current events. Unknown schemas retain
their raw sanitized body and parser report, without invented normalized events.

`tools/veo_probe.py` remains a diagnostic media probe, not a dataset importer.
Its legacy resource guesses are not authoritative inputs to this data layer.
`classifier.py` and the initial-player/source association logic are unchanged.

## Actual Highlights evidence

Capture: `20260913-real-jaen-vs-hispania-v90ba645-20260920-152110-9caa8a`.
Two saved `/highlights/` responses contain the same bare array of 37 records:
11 `goal`, 26 `shot_on_goal`; starts range from 302 to 6642 seconds; all recorded
durations are 25 seconds. The parser accepts this observed bare-array schema only.

Fields used: `id` (string), `type` (string), `start` (numeric match seconds),
`duration` (numeric seconds), `thumbnail` (URL), `videos[]` with `url`, `mime_type`,
`width`, `height`. Safe `created` and `is_ai_generated` are retained in internal
event metadata. Labels: goal → Gol; shot_on_goal → Tiro a puerta; unknown types
retain their original escaped text. Invalid id/type/start rows are skipped and
counted; valid duplicate IDs collapse within a snapshot.

Also observed, retained in raw artifacts but not assigned invented semantics:
`ai_resolution`, `annotations`, `camera_directions`, `comment`, `creator`,
`has_camera_directions`, `involved_players`, `modified`, `permissions`, `privacy`,
`reply_count`, `should_render`, `tags`, `team_association`.
All current `team_association` values are null and `involved_players` are empty.
`ai_resolution` is **not** a confidence score. Team/player/X/Z/confidence remain
unset. Clip MP4s remain references; no clip download or autoplay is performed.
Sanitized signed/credential-redacted URLs are not used as display thumbnails.
Remote public thumbnails load lazily and disappear on load failure.

There is no confirmed standalone reel: `/videos/` has Standard/Panorama sources,
not a compilation. No reel asset is created from these Highlights records.

The existing Pre Linares–Real Jaén workspace is a different match. Its media and
points were not reassigned. The 37 events are persisted in the correctly identified
Real Jaén–Hispania dataset-only workspace; import that match's core media before
video seeking can work there. Offline replay never trusts legacy asset guesses.

```bash
.venv/bin/python tools/veo_import_datasets.py \
  dev_app/runtime/veo-discovery/20260913-real-jaen-vs-hispania-v90ba645-20260920-152110-9caa8a \
  --create-match
```

Use `--match-id` to explicitly select an existing matching workspace and
`--runtime-root` for an isolated runtime. No media are downloaded by this command.

## Domain, storage and APIs

`match_data.py`: versioned `MatchEvent`, `Provenance`, `DatasetObservation`.
An event carries id/match/source/source_dataset/run/type/label/start, optional
duration/end, team/player, canonical x/z, confidence and optional resources.
Veo and our events share presentation contracts, never source identity. Event IDs
are stable per match/provider/dataset/original ID; future generated events must
also be unique per run. Internal provenance links normalized outputs to raw inputs.

```text
matches/<match_id>/
  match.json                 # media + small structured_data index references
  datasets/index.json        # dataset status, observations, active artifact refs
  artifacts/index.json       # small published artifact descriptors
  artifacts/<artifact_id>/
    manifest.json            # descriptor and optional chunk time index
    payload.json             # small sanitized raw or normalized event dataset
    000000.jsonl             # alternatively: large timestamp-sorted row chunks
  analysis/index.json        # run IDs
  analysis/<run_id>.json     # run state/configuration/versions/input/output refs
```

Old matches without indexes return empty datasets/events/runs; no migration or
rewrite is required. `LocalArtifactStore` uses content-addressed immutable JSON
artifacts and atomic index publication. Dataset history/active references can
change; published raw outputs do not. Interrupted chunk writes leave unindexed
directories, never visible partial artifacts. There is no cleanup daemon yet.
Local writes share the workspace lock: this is a single-process store, not a
multi-worker database or crash-durable transaction across all indexes.

Read-only endpoints under `/api/matches/{id}`:

- `/datasets`: statuses/counts and capabilities from actual stored data.
- `/events?offset=0&limit=500`: normalized UI events, max page size 1000;
  optional `start_s`, `end_s`, `source`, `event_type`. Time ranges overlap event
  intervals. Current snapshots only for Veo; our outputs retain distinct run IDs.
- `/analysis`: run/artifact metadata, without storage paths, configurations,
  technical errors or raw provenance. No start/cancel/fake progress APIs.

Normal UI never consumes raw Veo JSON or filesystem paths from these endpoints.
Small event snapshots are sorted/filtered on the backend before pagination. This
is not a high-volume observation API; don't route millions of detections through it.

## One clock and compact UI

`useMatchEvents` fetches by match/revision, not per frame. `MatchTimeline` keeps the
ordinary slider and adds lightweight SVG start markers. Markers have accessible
names, a roving tab stop, Left/Right/Home/End navigation and Enter/Space activation.
`EventsPanel` is a compact scrollable Spanish list, omitted for no events. Pages
load 500 events; “Mostrar más” makes additional markers and rows available.

Row/marker → `WorkspacePage.seekEvent(event)` → pause → clear image-only preview
→ existing shared seek at `event.start_s`. No individual clip player opens.
The existing video `timeupdate` keeps the timeline synchronized. Switching from
Tracking to the shared dual-lens decoder restores the match timestamp on metadata
load; Panorama, Interactive and Dual Lens continue using the same decoder/clock.
Pending seeks use the latest requested timestamp, including before metadata is
ready. All seeks are bounded by source duration. Points/landmarks are untouched.
There is no event overlay on the moving Tracking/Interactive image.

The event model already supports intervals; markers initially show their starts.
`matchEvents.ts` provides identity, ordering, filtering and adjacent-event helpers
without introducing unused navigation/filter panels.

## Analysis foundation and the next detector

`analysis_layers.py` defines `AnalysisRun` (queued/running/completed/failed/cancelled,
pipeline/stage/implementation/version, source media, config, timestamps, upstream
and output refs, error) and versioned artifact/observation contracts. Nothing runs
automatically and no synthetic run is created by declaring stage support.

Proposed flow (dependencies describe data requirements, not a fixed scheduler):

```text
media → person_detection → person_tracking → team_classification / role_classification
                        ↘ identity (jersey recognition + player association)
media → ball_detection → ball_tracking
tracks + calibration → field_positions → events / metrics / heatmaps
```

`Detection`, `TrackObservation`, `EntityClassification`, `FieldObservation` and
`IdentityAssignment` use match seconds, optional frame index and source media.
Image bboxes are source-pixel `(x, y, width, height)`, not resized-model coordinates.
Field positions reuse the existing pitch convention: X along length, Z across
width, origin at midfield, metres; don't flip signs or tune the validated geometry.
Role classification can describe player/referee/goalkeeper independently of team.
Model confidence is optional. Identity can be automatic or manually sourced.

To add RF-DETR (or YOLO/D-FINE) next:

1. Implement a backend stage adapter outside routes/viewer. Take explicit source
   media and sampling/time range, implementation/model version and configuration.
2. Map actual decoded frame timestamps onto match seconds (record any source-time
   mapping in run configuration). Return validated `Detection` rows in match-time
   order; undo input resizing so bboxes use native source pixels.
3. Create/save a real `AnalysisRun` through `LocalArtifactStore.save_run`; populate
   lifecycle times/status/error only from actual execution. No runner exists yet.
4. Stream rows into `write_records(..., 'detections.person', ..., Provenance(
   source='our_analysis', run_id=...), source_media_ids=[...])`. Record artifact IDs
   in the completed run. A new model/version creates a new run and immutable output.
5. Tracking consumes those artifact IDs using `read_window`, producing its own
   `tracks.person` artifacts and explicit upstream provenance, not reading detector
   internals or overwriting Veo tracking.
6. Later expose bounded time-window observation reads and a cache/prefetching hook
   for `current_time ± window`; an overlay should consume this boundary, not raw
   files or per-frame requests. Add execution/cancellation only with a real runner.

## Volume, provenance and corrections

`ArtifactStore` is the replaceable storage boundary. The local implementation
streams sorted JSONL chunks (default max 60 seconds or 10,000 rows). Each chunk
has time bounds, row count and checksum in its manifest. `read_window` selects
overlapping chunks first and streams only matching rows. Payload memory is bounded
by chunk size, not match length. Manifests are small relative to observations.
For larger deployments, replace this implementation with indexed Parquet/DuckDB
or another store without changing event/observation contracts. Compression,
server/client caching and a network observation endpoint are future work.

Artifacts carry source, dataset or run, schema/parser version and upstream IDs;
runs carry stage/implementation versions. Dataset observations add capture-specific
provenance. Veo and own-analysis results never silently overwrite or deduplicate
one another. Capability checks require nonempty, published outputs, not model
support or configured jobs. Veo-specific capabilities are separately named.

`ManualCorrection` describes an append-only overlay referencing raw artifacts,
entities, optional time bounds, author/time and superseded correction. Team/role,
jersey/identity, merge and split corrections can be stored as separate artifacts
with source `manual`. Future resolution must apply these explicitly; no correction
UI or destructive rewrite of automatic results is included here.

## Current limits

Only actual Highlights have a domain parser; other dataset bodies are retained but
not normalized. Capture is not paginated bulk ingestion. Public thumbnails may
expire. No media were downloaded for the dataset-only workspace, and live seeking
across real media still needs browser acceptance testing. No CV algorithms, job
runner, observation network API or future analytics panels are implemented.
