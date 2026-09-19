# Tracking click-to-pitch investigation

## Decision

Tracking video playback is implemented. Tracking metric clicks are disabled in
both the UI and the view-point API schema. No fixed homography, guessed FOV, or
automatic fitting is substituted for the missing virtual-camera trajectory.

This is a conclusion about the evidence available in this checkout, not a claim
that Veo never supplies the required data. No authenticated live Veo match was
available for this investigation.

## Evidence inspected

- `reference/veo_panoramic.vert.glsl`, `veo_panoramic.frag.glsl`,
  `veo_panoramic_uniforms.json`, and the reference PNG are Panoramic captures.
  The fragment shader declares `camera2logic` but does not use it. There is no
  complete Standard/Interactive shader or timestamped `virtualcam`/`fov`
  sequence in these files.
- Both original `.veo` files report calibration version `6.2`. They contain
  static lens/camera transforms, including `camera2logic`, but not an encoded
  tracking video's per-frame virtual-camera trajectory.
- `ffprobe` of both local dual-lens transcodes shows H.264 video at 2048×2048
  and AAC audio. Their durations are approximately 6744.004 s and 7249.042 s.
- The local `veo_standard_machine.mp4` and `video.mp4` contain 1920×1080 H.264
  and AAC audio, approximately 6744.004 s. Container/stream metadata shows
  ordinary creation/encoder tags and no camera-data stream or pose/FOV tags.
  Matching duration alone does not prove per-frame projection or time alignment.
- The importer observes authorized player network responses and resource links
  in bounded JSON. It is not a tracker-metadata reverse-engineering scraper.
  Without a live authorized session, unseen network resources cannot be ruled
  out or reported as verified.

The local probes used:

```bash
ffprobe -v error \
  -show_entries format=filename,duration:stream=index,codec_type,codec_name,width,height:stream_tags:format_tags \
  -of json veo_standard_machine.mp4
```

The same read-only probe was applied to `video.mp4` and both `transcode-*.mp4`
files. No source media or reference captures were changed.

## What is missing

An exact tracking pixel-to-ray transform needs all of:

1. The virtual-camera rotation/pose for the selected presentation timestamp,
   with its interpolation rule and timebase.
2. The view's FOV or intrinsics/zoom and aspect ratio, including any crop,
   letterboxing, stabilization or resampling in the encoded tracking video.
3. The transform convention tying that virtual camera to the static calibrated
   camera/world coordinates (`camera2logic` by itself is insufficient).
4. Verified synchronization between encoded tracking timestamps, source
   transcode timestamps and the camera trajectory, including clip offsets.

Only then can a tracking pixel produce a calibrated world ray, followed by the
existing Y=0 intersection and X/Z field classification. Static calibration
cannot determine which part of the panorama an unknown moving camera displayed.

## Proven / inferred / unresolved

- **Proven by local tests:** faithful Panoramic WebGL visualization on both
  samples; unchanged Python panorama-to-ground; numerical RAW inversion of the
  same forward lens mapping on both matches and lenses.
- **Inferred:** the tracking video is a rendered moving view of the recording;
  its dimensions, duration and appearance do not establish an invertible mapping.
- **Unresolved:** where Veo supplies the tracking trajectory, whether it matches
  the downloaded encoding exactly, and its matrix/time/FOV conventions.

## Next evidence to capture

In a user-authorized match session, inspect the network while actually switching
between Panoramic, Standard/Interactive and follow-camera modes. Capture shader
sources and representative uniforms, identify any time-indexed camera resources,
and retain sanitized format examples and timestamp associations. Do not publish
cookies, headers or signed URLs. Validate the discovered transform against known
landmarks and corresponding panorama rays across several times/zooms before
enabling metric clicks. This is future work, not an enabled approximation.
