"""Export utility for IA Bluebook ONNX Digit Classifier."""

from pathlib import Path
from scripts.train_digit_model import train_and_export_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"

if __name__ == "__main__":
    print(f"Exporting compact ONNX digit model to {MODEL_PATH}...")
    train_and_export_model(MODEL_PATH)
    print("Export complete.")
