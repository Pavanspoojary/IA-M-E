"""Unit tests for image preprocessing pipeline."""

import pytest
import numpy as np
import cv2

from src.image_preprocessor import ImagePreprocessor, PreprocessingResult


@pytest.fixture
def preprocessor():
    return ImagePreprocessor(default_target_width=1240, default_target_height=1754)


def test_empty_image_handling(preprocessor):
    """Test graceful handling of empty or None images."""
    res = preprocessor.preprocess(None)
    assert not res.success
    assert res.alignment_status == "failed"

    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = preprocessor.preprocess(empty)
    assert not res_empty.success


def test_orientation_correction(preprocessor):
    """Test landscape images are rotated to portrait."""
    # Create landscape image (width > height)
    landscape = np.full((600, 900, 3), 255, dtype=np.uint8)
    # Put a marker near top-left of original
    cv2.circle(landscape, (100, 100), 20, (0, 0, 0), -1)

    res = preprocessor.preprocess(landscape)
    assert res.success
    # Output must match target portrait resolution
    assert res.aligned_bgr.shape[0] == 1754
    assert res.aligned_bgr.shape[1] == 1240


def test_resolution_standardization(preprocessor):
    """Test input images of different resolutions are standardized."""
    resolutions = [(800, 600), (1920, 1080), (3000, 4000), (500, 700)]
    for h, w in resolutions:
        img = np.full((h, w, 3), 240, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (w - 20, h - 20), (50, 50, 50), 2)
        res = preprocessor.preprocess(img)
        assert res.success
        assert res.aligned_bgr.shape == (1754, 1240, 3)
        assert res.thresholded_inv.shape == (1754, 1240)


def test_skew_correction(preprocessor):
    """Test minor skew angle detection and correction."""
    img = np.full((1754, 1240, 3), 255, dtype=np.uint8)
    # Draw clear horizontal table lines
    for y in range(400, 1200, 100):
        cv2.line(img, (100, y), (1140, y), (0, 0, 0), 3)

    # Rotate by 4 degrees
    center = (1240 // 2, 1754 // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 4.0, 1.0)
    skewed = cv2.warpAffine(img, rot_mat, (1240, 1754), borderValue=(255, 255, 255))

    res = preprocessor.preprocess(skewed)
    assert res.success
    # Detected angle should be close to 4.0 degrees
    assert abs(abs(res.skew_angle) - 4.0) < 1.0


def test_thresholding_methods(preprocessor):
    """Test both Otsu and Adaptive thresholding modes."""
    img = np.full((1754, 1240, 3), 250, dtype=np.uint8)
    cv2.putText(img, "TEST", (400, 800), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (10, 10, 10), 5)

    res_otsu = preprocessor.preprocess(img, threshold_method="otsu")
    assert res_otsu.success
    assert np.count_nonzero(res_otsu.thresholded_inv) > 100

    res_adapt = preprocessor.preprocess(img, threshold_method="adaptive")
    assert res_adapt.success
    assert np.count_nonzero(res_adapt.thresholded_inv) > 100
