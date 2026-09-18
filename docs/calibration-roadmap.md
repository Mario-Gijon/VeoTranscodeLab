# Calibration roadmap

## Goal

Build a match-level calibration for Veo's fixed dual-lens camera from one
manually selected football-pitch frame. The calibration will map both lenses to
one metric pitch coordinate system, where `X` follows pitch length and `Z`
follows pitch width.

Conceptually, each lens will have its own image model and field-plane transform:

```text
LEFT RAW pixel  -> LEFT undistortion model  -> LEFT field-plane transform  -> shared pitch (X, Z)
RIGHT RAW pixel -> RIGHT undistortion model -> RIGHT field-plane transform -> shared pitch (X, Z)
```

The same shared pitch point must result from matched LEFT/RIGHT observations of
the same physical point.

## Inputs retained from Veo

The `.veo` data is useful initialization and reference metadata, not a trusted
end-to-end metric camera model. The retained metadata reader exposes:

- `field_length` and `field_width`;
- Veo's selected field-corner anchors in local RAW lens coordinates;
- `lens_left2camera` and `lens_right2camera`, including the approximate
  7.8 cm stereo baseline;
- intrinsic and distortion metadata for both lenses;
- presentation metadata used solely by the panorama visualisation.

Original `.veo` files and genuine calibration/source frames remain versioned.

## Calibration constraints

The future objective will combine complementary image evidence:

1. Straight painted football-pitch lines to estimate each lens's distortion and
   field-plane mapping.
2. Known metric pitch landmarks, with Veo selected corners as useful anchors.
3. LEFT/RIGHT correspondences so both lenses map the same scene point to the
   same `X, Z` coordinate.
4. Centre-circle and other circle constraints after the line/landmark model is
   established.

One manually calibrated frame should be reusable across the match because the
camera is fixed.

## Established conclusions

- Downloaded transcodes contain two RAW `2048x1024` lens images stacked into a
  `2048x2048` frame.
- The camera is fixed during a match.
- Veo selected corners and field dimensions are valuable anchors/metadata.
- Optimising `camera2world` from four selected corners was tested and rejected.
- The previous interpretation of Veo distortion and RAW-to-calibration scaling
  was not accurate enough for metric player positioning.
- Projected-field overlays, pixel-mapping hypotheses, cross-lens ray checks,
  pose refinement, and validation UI served their investigative purpose but are
  not the product pipeline.
- Sequential white-line tracing experiments were rejected as the calibration
  strategy.
- The simple LEFT/RIGHT panorama blend remains useful as an approximate visual
  relationship between the lenses and may support later annotation work.

## Deliberately out of scope for this phase

This cleanup does not add a calibration solver, correction offsets, camera pose
optimisation, automatic line tracing, or player-positioning application. It
keeps the repository limited to durable inputs, metadata access, the panorama
visualisation, and the roadmap for the next implementation phase.
