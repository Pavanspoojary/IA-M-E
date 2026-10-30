# Digit Templates Library

This directory holds the reference digit template crops (0 to 9) used by the classical OpenCV normalized cross-correlation matching pipeline.

## Structure

- Templates are stored as 32x32 binary PNG images (`0_style0.png`, `1_style0.png`, ..., `9_styleN.png`).
- Foreground digit strokes are white (`255`), and background pixels are black (`0`).
- You can place manually labelled digit crops from actual college bluebooks directly into this directory using the naming pattern:
  `<digit>_custom_<id>.png` (for example: `5_custom_01.png`, `8_custom_bluebook.png`).

## Automatic Synthesis

If this directory contains no template images on initialization, `DigitRecognizer` will automatically synthesize diverse reference exemplars across standard typographical weights, line thicknesses, and styles using OpenCV vector fonts.
