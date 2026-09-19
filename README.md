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

## Repository contents and next work

- `analysis/` contains the validated renderer and metadata reader.
- `data/` contains the original `camera_alignment` calibration files.
- `samples/` contains representative source frames; local transcode MP4s may
  also be present at the repository root and are ignored by Git.
- `reference/` contains the captured Veo Panoramic shaders, uniforms, and
  reference panorama image.
- `output/` is intentionally preserved as historical visual evidence and is
  ignored by Git.

The next planned phase is reproducing Veo's Interactive renderer. Later work
will reuse the recovered geometry for player-to-pitch mapping.
