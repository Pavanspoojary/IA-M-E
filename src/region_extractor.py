"""Region extraction module for bluebook pages.

Extracts question-wise marks cells and student ID regions based on
template configurations with coordinate scaling and morphological border removal.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2

from src.template_manager import BluebookTemplate, QuestionRegion, StudentIdRegion


@dataclass
class ExtractedRegion:
    """Represents an extracted region crop with color, grayscale, and cleaned binary versions."""

    region_id: str
    label: str
    crop_bgr: np.ndarray
    crop_gray: np.ndarray
    crop_thresh_inv: np.ndarray
    bbox_pixel: Tuple[int, int, int, int]  # (x, y, w, h)
    is_student_id: bool = False
    max_marks: float = 0.0
    step: float = 0.5


class RegionExtractor:
    """Crops and conditions designated marks and identification fields from preprocessed sheets."""

    def __init__(self, inner_margin_ratio: float = 0.0):
        """Initialize RegionExtractor.

        Args:
            inner_margin_ratio: Inset percentage if user bbox includes outer table lines.
        """
        self.inner_margin_ratio = inner_margin_ratio

    def extract_all(
        self,
        aligned_bgr: np.ndarray,
        thresholded_inv: np.ndarray,
        template: BluebookTemplate,
    ) -> Dict[str, ExtractedRegion]:
        """Extract all question marks fields and student ID region from aligned sheet.

        Args:
            aligned_bgr: Standardized BGR image from preprocessor.
            thresholded_inv: Inverted binary image from preprocessor (strokes = 255).
            template: BluebookTemplate containing regions.

        Returns:
            Dictionary mapping region_id (e.g. "q1", "student_id") to ExtractedRegion.
        """
        img_h, img_w = aligned_bgr.shape[:2]
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        regions: Dict[str, ExtractedRegion] = {}

        # 1. Extract Student ID region if configured
        if template.student_id_region is not None:
            sid_reg = template.student_id_region
            x, y, w, h = template.get_pixel_bbox_for_resolution(sid_reg, img_w, img_h)
            crop_bgr = aligned_bgr[y : y + h, x : x + w]
            crop_gray = gray[y : y + h, x : x + w]
            crop_thresh = thresholded_inv[y : y + h, x : x + w]

            # Clean border lines
            cleaned_thresh = self._remove_table_lines(crop_thresh)

            regions["student_id"] = ExtractedRegion(
                region_id="student_id",
                label="Student ID / USN",
                crop_bgr=crop_bgr,
                crop_gray=crop_gray,
                crop_thresh_inv=cleaned_thresh,
                bbox_pixel=(x, y, w, h),
                is_student_id=True,
            )

        # 2. Extract Question Marks regions
        for q in template.questions:
            x, y, w, h = template.get_pixel_bbox_for_resolution(q, img_w, img_h)

            # Apply a slight inner inset to avoid outer bounding box printed cell borders
            inset_x = int(round(w * self.inner_margin_ratio))
            inset_y = int(round(h * self.inner_margin_ratio))

            cx = max(0, x + inset_x)
            cy = max(0, y + inset_y)
            cw = max(5, w - (2 * inset_x))
            ch = max(5, h - (2 * inset_y))

            crop_bgr = aligned_bgr[cy : cy + ch, cx : cx + cw]
            crop_gray = gray[cy : cy + ch, cx : cx + cw]
            crop_thresh = thresholded_inv[cy : cy + ch, cx : cx + cw]

            # Morphologically remove any residual table lines inside the cell
            cleaned_thresh = self._remove_table_lines(crop_thresh)

            regions[q.id] = ExtractedRegion(
                region_id=q.id,
                label=q.label,
                crop_bgr=crop_bgr,
                crop_gray=crop_gray,
                crop_thresh_inv=cleaned_thresh,
                bbox_pixel=(cx, cy, cw, ch),
                is_student_id=False,
                max_marks=q.max_marks,
                step=q.step,
            )

        return regions

    def _remove_table_lines(self, binary_inv_crop: np.ndarray) -> np.ndarray:
        """Remove horizontal and vertical table borders while preserving handwritten strokes."""
        if binary_inv_crop is None or binary_inv_crop.size == 0:
            return binary_inv_crop

        h, w = binary_inv_crop.shape[:2]
        output = binary_inv_crop.copy()

        # Horizontal line kernel (at least 60% of crop width)
        h_len = max(10, int(w * 0.65))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_lines = cv2.morphologyEx(binary_inv_crop, cv2.MORPH_OPEN, h_kernel)

        # Vertical line kernel (at least 60% of crop height)
        v_len = max(10, int(h * 0.65))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_lines = cv2.morphologyEx(binary_inv_crop, cv2.MORPH_OPEN, v_kernel)

        # Combine detected grid lines
        grid_lines = cv2.bitwise_or(h_lines, v_lines)

        # Dilate slightly to cleanly cut out intersections
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_lines = cv2.dilate(grid_lines, dilate_kernel, iterations=1)

        # Subtract grid lines from handwriting
        cleaned = cv2.subtract(output, grid_lines)

        # Remove lone single-pixel noise specs
        cleaned = cv2.medianBlur(cleaned, 3)

        return cleaned
