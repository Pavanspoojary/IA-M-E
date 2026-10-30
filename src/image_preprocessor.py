"""Image preprocessing module for college examination bluebooks.

Implements a deterministic, 10-stage classical computer vision pipeline
with orientation correction, 4-point perspective transformation,
deskewing, noise filtering, and adaptive/Otsu thresholding.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2


@dataclass
class PreprocessingResult:
    """Encapsulates the complete result of the 10-stage preprocessing pipeline."""

    success: bool
    original_bgr: np.ndarray
    aligned_bgr: np.ndarray
    grayscale: np.ndarray
    thresholded_inv: np.ndarray  # Foreground strokes are 255, background is 0
    thresholded: np.ndarray      # Standard black text on white background
    skew_angle: float
    perspective_corrected: bool
    alignment_status: str        # 'auto_aligned', 'fallback_scaled', 'manual_needed'
    error_message: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)


class ImagePreprocessor:
    """Applies classical computer vision transformations to align and clean bluebook sheets."""

    def __init__(
        self,
        default_target_width: int = 1240,
        default_target_height: int = 1754,
    ):
        """Initialize ImagePreprocessor.

        Args:
            default_target_width: Standard page width in pixels (150 DPI A4 is 1240).
            default_target_height: Standard page height in pixels (150 DPI A4 is 1754).
        """
        self.default_width = default_target_width
        self.default_height = default_target_height

    def preprocess(
        self,
        image_bgr: np.ndarray,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        auto_align: bool = True,
        denoise_method: str = "median",
        threshold_method: str = "otsu",
        manual_corners: Optional[np.ndarray] = None,
    ) -> PreprocessingResult:
        """Run the end-to-end 10-stage preprocessing pipeline.

        Args:
            image_bgr: Source OpenCV BGR image.
            target_width: Target normalized width.
            target_height: Target normalized height.
            auto_align: Whether to automatically detect page boundary and warp perspective.
            denoise_method: 'median' or 'gaussian'.
            threshold_method: 'otsu' or 'adaptive'.
            manual_corners: Optional 4-point array provided by user if auto-align fails.

        Returns:
            PreprocessingResult with aligned and thresholded outputs.
        """
        if image_bgr is None or image_bgr.size == 0:
            return PreprocessingResult(
                success=False,
                original_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
                aligned_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
                grayscale=np.zeros((100, 100), dtype=np.uint8),
                thresholded_inv=np.zeros((100, 100), dtype=np.uint8),
                thresholded=np.zeros((100, 100), dtype=np.uint8),
                skew_angle=0.0,
                perspective_corrected=False,
                alignment_status="failed",
                error_message="Input image is empty or invalid.",
            )

        orig_copy = image_bgr.copy()
        target_w = target_width or self.default_width
        target_h = target_height or self.default_height

        # Stage 1: Load and ensure orientation (portrait check)
        h, w = orig_copy.shape[:2]
        if w > h:
            # Bluebook second sheet is portrait. If landscape, rotate 90 degrees clockwise
            working_img = cv2.rotate(orig_copy, cv2.ROTATE_90_CLOCKWISE)
        else:
            working_img = orig_copy

        perspective_applied = False
        alignment_status = "fallback_scaled"
        corners_used = None

        # Stage 2-4: Page boundary detection and perspective correction
        if manual_corners is not None and len(manual_corners) == 4:
            working_img = self._four_point_transform(working_img, manual_corners, target_w, target_h)
            perspective_applied = True
            alignment_status = "manual_aligned"
            corners_used = manual_corners
        elif auto_align:
            detected_corners = self._detect_page_contour(working_img)
            if detected_corners is not None:
                working_img = self._four_point_transform(working_img, detected_corners, target_w, target_h)
                perspective_applied = True
                alignment_status = "auto_aligned"
                corners_used = detected_corners
            else:
                # If page detection fails, resize directly to reference resolution
                working_img = cv2.resize(working_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                alignment_status = "fallback_scaled"
        else:
            working_img = cv2.resize(working_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Stage 5: Ensure consistent reference resolution
        if working_img.shape[1] != target_w or working_img.shape[0] != target_h:
            working_img = cv2.resize(working_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Stage 6: Convert to grayscale
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)

        # Stage 7: Denoising
        if denoise_method == "gaussian":
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        else:
            denoised = cv2.medianBlur(gray, 3)

        # Stage 8: Minor skew detection and correction
        skew_angle = self._detect_skew_angle(denoised)
        if abs(skew_angle) > 0.3 and abs(skew_angle) <= 15.0:
            # Rotate grayscale and color image
            working_img = self._rotate_image(working_img, skew_angle)
            gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 3) if denoise_method == "median" else cv2.GaussianBlur(gray, (3, 3), 0)

        # Stage 9: Thresholding to separate handwriting/lines from background
        if threshold_method == "adaptive":
            thresh_inv = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=21,
                C=8,
            )
        else:
            # Otsu thresholding
            _, thresh_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Standard black on white representation for review
        thresh_std = cv2.bitwise_not(thresh_inv)

        return PreprocessingResult(
            success=True,
            original_bgr=orig_copy,
            aligned_bgr=working_img,
            grayscale=gray,
            thresholded_inv=thresh_inv,
            thresholded=thresh_std,
            skew_angle=round(skew_angle, 2),
            perspective_corrected=perspective_applied,
            alignment_status=alignment_status,
            error_message=None,
            debug_info={
                "target_width": target_w,
                "target_height": target_h,
                "corners_used": corners_used.tolist() if corners_used is not None else None,
            },
        )

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum of coords: top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Difference of coords: top-right has smallest diff, bottom-left has largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _four_point_transform(
        self,
        image: np.ndarray,
        pts: np.ndarray,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Apply 4-point perspective warp to un-distort sheet photograph."""
        rect = self._order_points(pts)
        dst = np.array(
            [
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_LINEAR)
        return warped

    def _detect_page_contour(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect four outer corners of the bluebook page."""
        orig_h, orig_w = image_bgr.shape[:2]
        # Downscale for robust edge finding
        scale = 800.0 / max(orig_h, orig_w)
        if scale < 1.0:
            small_w = int(orig_w * scale)
            small_h = int(orig_h * scale)
            small = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small = image_bgr.copy()
            scale = 1.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Dilate edges to close minor gaps in border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        min_area = (small.shape[0] * small.shape[1]) * 0.35  # Must occupy at least 35% of frame

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Scale coordinates back to original size
                pts = approx.reshape(4, 2) / scale
                return pts

        return None

    def _detect_skew_angle(self, gray: np.ndarray) -> float:
        """Detect dominant skew angle from printed horizontal lines and text contours."""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Look for horizontal line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=gray.shape[1] // 5,
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines:
            pts = line.ravel()
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)))
            # Keep angles that are roughly horizontal (+/- 15 degrees)
            if abs(angle) <= 15.0:
                angles.append(angle)

        if not angles:
            return 0.0

        median_angle = float(np.median(angles))
        return median_angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around its center to correct skew."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
