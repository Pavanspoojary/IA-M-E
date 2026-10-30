"""Unit tests for local ONNX Runtime digit recognition engine."""

import pytest
from pathlib import Path
import numpy as np
import cv2

from src.digit_recognizer import DigitRecognizer, DigitRecognitionResult

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"


@pytest.fixture
def recognizer():
    return DigitRecognizer(model_path=MODEL_PATH)


def test_model_loading_and_cpu_provider(recognizer):
    """Test ONNX model loads with CPUExecutionProvider."""
    assert recognizer.model_available is True
    assert recognizer.session is not None
    assert "CPUExecutionProvider" in recognizer.session.get_providers()
    assert recognizer.input_name == "input"
    assert recognizer.output_name == "output"


def test_missing_model_handling():
    """Test graceful fallback when ONNX model file is missing."""
    rec = DigitRecognizer(model_path="models/non_existent_model.onnx")
    assert rec.model_available is False
    assert rec.session is None
    assert "not found" in rec.load_error_message

    # Inference on missing model should return unclassified without crashing
    crop = np.zeros((40, 40), dtype=np.uint8)
    crop[10:30, 15:25] = 255  # Fake stroke
    res = rec.recognize_region(crop)
    assert res.is_ambiguous is True
    assert res.value_str is None
    assert "Model unavailable" in res.explanation or "manual" in res.explanation.lower()


def test_blank_field_detection(recognizer):
    """Test blank fields are detected without calling the model."""
    empty_crop = np.zeros((50, 100), dtype=np.uint8)
    res = recognizer.recognize_region(empty_crop)
    assert res.is_blank is True
    assert res.value_str is None
    assert res.confidence >= 0.95


def test_input_tensor_preprocessing(recognizer):
    """Test tensor preprocessing generates correct shape (1, 1, 28, 28) and range [0, 1]."""
    stroke = np.zeros((50, 40), dtype=np.uint8)
    cv2.putText(stroke, "5", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

    tensor = recognizer._preprocess_for_onnx(stroke)
    assert tensor is not None
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_single_digit_recognition(recognizer):
    """Test recognition of clean single digits."""
    test_digits = ["3", "5", "7", "8", "9"]
    for d in test_digits:
        img = np.zeros((60, 60), dtype=np.uint8)
        cv2.putText(img, d, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
        res = recognizer.recognize_region(img, max_marks=10.0)
        assert not res.is_blank
        assert res.value_str == d
        assert res.numeric_value == float(d)
        assert res.confidence > 0.60


def test_multi_digit_recognition(recognizer):
    """Test multi-digit marks such as 10."""
    img = np.zeros((60, 110), dtype=np.uint8)
    cv2.putText(img, "10", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
    res = recognizer.recognize_region(img, max_marks=10.0)
    assert not res.is_blank
    assert res.value_str == "10"
    assert res.numeric_value == 10.0


def test_decimal_mark_recognition(recognizer):
    """Test decimal marks with dot such as 4.5."""
    img = np.zeros((60, 110), dtype=np.uint8)
    cv2.putText(img, "4 . 5", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    res = recognizer.recognize_region(img, max_marks=10.0)
    assert not res.is_blank
    assert "." in res.value_str
    assert res.numeric_value == 4.5
