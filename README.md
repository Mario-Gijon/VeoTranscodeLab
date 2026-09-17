# VeoTranscodeLab

## Interactive field projection debug

Run the lightweight OpenCV experiment from the repository root:

```bash
python scripts/interactive_field_debug.py --lens left
python scripts/interactive_field_debug.py --lens right
```

Click pixels in the selected Veo lens image to print their Veo world and
field coordinates. The companion top-down view plots each projected point,
with the calibrated field length on world X and field width on world Z.
Press `c` to clear points or `q`/Escape to quit. This validates the existing
pixel-to-ground projection interactively; it does not change the calibration
math or the Veo coordinate conventions.
