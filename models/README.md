# Local ONNX Digit Classification Model

This directory houses the compact, local handwritten digit classifier used for question-wise marks extraction via ONNX Runtime on the CPU.

## Model Specifications

- **File Path**: `models/digit_model.onnx`
- **Architecture**: Compact Convolutional Neural Network (Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Dense -> Dropout -> Dense)
- **Input Tensor**:
  - Name: `input`
  - Shape: `[batch_size, 1, 28, 28]`
  - Data Type: `float32`
  - Normalization: Grayscale intensity scaled to `[0.0, 1.0]`. Digits are centered white strokes (`1.0`) on a black background (`0.0`), consistent with standard digit classification pipelines.
- **Output Tensor**:
  - Name: `output`
  - Shape: `[batch_size, 10]`
  - Data Type: `float32`
  - Interpretation: Logits or unnormalized log-probabilities corresponding to digits `0` through `9`. Argmax yields the predicted digit class, and Softmax provides the candidate confidence.
- **Execution Provider**:
  - Strictly `CPUExecutionProvider` configured through ONNX Runtime (`onnxruntime.InferenceSession(..., providers=['CPUExecutionProvider'])`).
  - No GPU or external accelerator required.
  - Zero external API calls, zero telemetry.

## Fallback & Missing Model Behavior

If `digit_model.onnx` is not present in this directory:
- The web application and extraction pipeline automatically fall back to **Manual-Only Verification Mode**.
- The application will not crash or fail silently.
- Preprocessing, perspective correction, table region cropping, and validation rules remain completely functional.
- The web interface displays a clear notification banner indicating that local CPU model inference is unavailable, allowing evaluators to manually inspect cell crops and input marks directly.

## Training and Model Export

To train and export a customized model using labelled bluebook samples or synthetic variations, run:

```bash
python scripts/train_digit_model.py
python scripts/export_onnx.py
python scripts/verify_model.py
```
