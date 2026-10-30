"""Handwritten digit recognition module using local ONNX Runtime CPU inference.

Implements an isolated marks-field processing pipeline with line removal,
blank cell filtering, connected-component digit segmentation,
ONNX tensor preprocessing (28x28 normalized float32), and rule-based
reconstruction for multi-digit values and decimal marks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
import onnxruntime as ort


@dataclass
class DigitRecognitionResult:
    """Detailed result of digit recognition on a single marks region."""

    value_str: Optional[str]        # Recognized string, e.g. "10", "7.5", "5"
    numeric_value: Optional[float]  # Parsed float value, or None if blank/invalid
    confidence: float               # Softmax probability or score in range [0.0, 1.0]
    is_blank: bool                  # True if region has negligible foreground ink
    is_ambiguous: bool              # True if confidence is low, margin narrow, or segmentation uncertain
    char_candidates: List[str]      # Individual recognized character tokens
    segmented_crops: List[np.ndarray] = field(default_factory=list)
    explanation: str = ""           # Diagnostic reasoning breakdown
    model_version: str = "None"     # Active model identifier


class DigitRecognizer:
    """Classifies handwritten digits from bluebook cells using local ONNX Runtime on CPU."""

    MODEL_NAME = "IA_DigitClassifier_ONNX"
    MODEL_VERSION = "1.0.0"
    TARGET_SIZE = (28, 28)
    INNER_SIZE = 20

    _cached_session: Optional[ort.InferenceSession] = None
    _cached_model_path: Optional[str] = None

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_confidence: float = 0.70,
        min_margin: float = 0.12,
    ):
        """Initialize DigitRecognizer.

        Args:
            model_path: Path to ONNX model file. Defaults to models/digit_model.onnx.
            min_confidence: Minimum softmax probability to accept without manual review.
            min_margin: Minimum margin between top-1 and top-2 candidate probabilities.
        """
        if model_path is None:
            self.model_path = Path(__file__).resolve().parent.parent / "models" / "digit_model.onnx"
        else:
            self.model_path = Path(model_path)

        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.session: Optional[ort.InferenceSession] = None
        self.model_available: bool = False
        self.load_error_message: Optional[str] = None
        self.input_name: str = "input"
        self.output_name: str = "output"

        self._initialize_onnx_session()

    def _initialize_onnx_session(self) -> None:
        """Load ONNX Runtime inference session with CPUExecutionProvider, caching across calls."""
        if not self.model_path.exists():
            self.model_available = False
            self.load_error_message = (
                f"ONNX model file not found at '{self.model_path}'. "
                "Running in manual-only verification mode. "
                "Run 'python scripts/train_digit_model.py' to generate the model."
            )
            return

        # Use cached session if available for the same path
        path_str = str(self.model_path)
        if DigitRecognizer._cached_session is not None and DigitRecognizer._cached_model_path == path_str:
            self.session = DigitRecognizer._cached_session
            self.model_available = True
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            return

        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.enable_cpu_mem_arena = True

            # Strictly configure CPU execution provider
            self.session = ort.InferenceSession(
                path_str,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Validate input and output shapes
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if not inputs or not outputs:
                raise ValueError("Model must have valid input and output tensors.")

            self.input_name = inputs[0].name
            self.output_name = outputs[0].name

            DigitRecognizer._cached_session = self.session
            DigitRecognizer._cached_model_path = path_str
            self.model_available = True
            self.load_error_message = None

        except Exception as exc:
            self.model_available = False
            self.session = None
            self.load_error_message = f"Failed to initialize ONNX Runtime CPU session: {exc}"

    def recognize_region(
        self,
        thresh_inv: np.ndarray,
        max_marks: float = 10.0,
        step: float = 0.5,
        allow_decimal: bool = True,
    ) -> DigitRecognitionResult:
        """Run complete recognition pipeline on a single marks region.

        Args:
            thresh_inv: Inverted binary image (strokes = 255, background = 0).
            max_marks: Permitted maximum marks for this question.
            step: Allowed increment (e.g. 0.5).
            allow_decimal: Whether decimal point reconstruction is permitted.

        Returns:
            DigitRecognitionResult with tokens, confidence, and audit metadata.
        """
        model_ver_str = f"{self.MODEL_NAME}-v{self.MODEL_VERSION}" if self.model_available else "Manual-Only"

        if thresh_inv is None or thresh_inv.size == 0:
            return DigitRecognitionResult(
                value_str=None,
                numeric_value=None,
                confidence=1.0,
                is_blank=True,
                is_ambiguous=False,
                char_candidates=[],
                explanation="Crop region is empty.",
                model_version=model_ver_str,
            )

        h, w = thresh_inv.shape[:2]
        total_pixels = h * w
        fg_pixels = cv2.countNonZero(thresh_inv)

        # 1. Blank cell detection (density < 0.35% or < 25 pixels)
        if fg_pixels < max(25, int(total_pixels * 0.0035)):
            return DigitRecognitionResult(
                value_str=None,
                numeric_value=None,
                confidence=0.98,
                is_blank=True,
                is_ambiguous=False,
                char_candidates=[],
                explanation=f"Blank cell: only {fg_pixels} ink pixels detected.",
                model_version=model_ver_str,
            )

        # If model is unavailable, return unclassified region for manual review
        if not self.model_available or self.session is None:
            return DigitRecognitionResult(
                value_str=None,
                numeric_value=None,
                confidence=0.0,
                is_blank=False,
                is_ambiguous=True,
                char_candidates=[],
                explanation=self.load_error_message or "Model unavailable - manual review required.",
                model_version="Manual-Only",
            )

        # 2. Segment connected components and contours
        digit_boxes, has_dot, is_segmentation_uncertain = self._segment_digits(thresh_inv, allow_decimal=allow_decimal)

        if not digit_boxes:
            return DigitRecognitionResult(
                value_str=None,
                numeric_value=None,
                confidence=0.90,
                is_blank=True,
                is_ambiguous=False,
                char_candidates=[],
                explanation="No valid digit contours found after noise filtering.",
                model_version=model_ver_str,
            )

        # 3. Classify each segmented character with ONNX Runtime
        char_tokens: List[str] = []
        confidences: List[float] = []
        crops_out: List[np.ndarray] = []
        notes: List[str] = []
        any_char_ambiguous = False

        for idx, (bx, by, bw, bh) in enumerate(digit_boxes):
            char_crop = thresh_inv[by : by + bh, bx : bx + bw]
            crops_out.append(char_crop)

            best_digit, score, margin, is_ambig, note = self._classify_char_crop(char_crop, bw, bh)
            char_tokens.append(best_digit)
            confidences.append(score)
            notes.append(f"Digit {idx+1}: {best_digit} (prob={score:.2f}, {note})")

            if is_ambig:
                any_char_ambiguous = True

        # 4. Reconstruct complete value (supporting decimal points and multi-digits)
        if not allow_decimal:
            has_dot = False

        if has_dot and len(char_tokens) == 2:
            value_str = f"{char_tokens[0]}.{char_tokens[1]}"
        elif has_dot and len(char_tokens) == 1:
            value_str = f"{char_tokens[0]}.5"
        else:
            value_str = "".join(char_tokens)

        # Parse numeric value
        numeric_val: Optional[float] = None
        try:
            numeric_val = float(value_str)
        except ValueError:
            numeric_val = None

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Over-segmented single digit fallback (only if total bounding box is narrow like a single digit: aspect < 0.95)
        if numeric_val is not None and numeric_val > max_marks and len(char_tokens) > 1 and max_marks <= 10.0:
            pts_all = cv2.findNonZero(thresh_inv)
            if pts_all is not None:
                ax, ay, aw, ah = cv2.boundingRect(pts_all)
                if aw / float(max(1, ah)) < 0.95:
                    full_crop = thresh_inv[ay : ay + ah, ax : ax + aw]
                    s_digit, s_score, s_margin, s_ambig, s_note = self._classify_char_crop(full_crop, aw, ah)
                    try:
                        s_num = float(s_digit)
                        if s_num <= max_marks:
                            value_str = s_digit
                            numeric_val = s_num
                            avg_confidence = s_score
                            char_tokens = [s_digit]
                            any_char_ambiguous = s_ambig
                            crops_out = [full_crop]
                            notes = [f"Consolidated single digit: {s_digit} (prob={s_score:.2f}, {s_note})"]
                    except ValueError:
                        pass

        # Gating rules: flag for teacher review if ambiguous, low confidence, or out of bounds
        is_overall_ambiguous = (
            any_char_ambiguous
            or is_segmentation_uncertain
            or avg_confidence < self.min_confidence
            or (numeric_val is not None and numeric_val > max_marks)
        )

        explanation = "; ".join(notes)
        if is_segmentation_uncertain:
            explanation += " [Uncertain segmentation / touching strokes]"

        return DigitRecognitionResult(
            value_str=value_str,
            numeric_value=numeric_val,
            confidence=round(avg_confidence, 3),
            is_blank=False,
            is_ambiguous=is_overall_ambiguous,
            char_candidates=char_tokens,
            segmented_crops=crops_out,
            explanation=explanation,
            model_version=model_ver_str,
        )

    def _segment_digits(
        self,
        thresh_inv: np.ndarray,
        allow_decimal: bool = True,
    ) -> Tuple[List[Tuple[int, int, int, int]], bool, bool]:
        """Segment connected components into ordered bounding boxes from left to right."""
        h, w = thresh_inv.shape[:2]
        contours, hierarchy = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 15
        min_h = max(6, int(h * 0.15))
        min_w = 3

        raw_boxes = []
        has_dot = False
        is_uncertain = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Decimal dot detection: small low-positioned component (only when allow_decimal is True)
            if allow_decimal and area >= 6 and area <= 30 and bw <= 8 and bh <= 8 and (y + bh) > (h * 0.55):
                has_dot = True
                continue

            # Noise rejection
            if area < min_area or bh < min_h or bw < min_w:
                continue

            # Cell line rejection (spans across whole cell horizontally or touches border edges)
            if (bw > (w * 0.92) and bh < (h * 0.25)) or ((y <= 2 or (y + bh) >= (h - 2)) and bh <= 5 and bw >= 20):
                continue

            # Multi-digit conjoined check (aspect ratio > 1.35)
            aspect = bw / float(bh)
            if aspect > 1.35 and bw > 25:
                split_boxes = self._attempt_vertical_split(thresh_inv, x, y, bw, bh)
                if len(split_boxes) > 1:
                    raw_boxes.extend(split_boxes)
                else:
                    raw_boxes.append((x, y, bw, bh))
                    is_uncertain = True
            else:
                raw_boxes.append((x, y, bw, bh))

        # Merge vertically overlapping components belonging to the same character
        # (e.g. top horizontal bar of 5, crossbar of 7, swoop of 2)
        merged_boxes = []
        for b in sorted(raw_boxes, key=lambda box: box[0]):
            if not merged_boxes:
                merged_boxes.append(b)
            else:
                prev = merged_boxes[-1]
                overlap_x = min(prev[0] + prev[2], b[0] + b[2]) - max(prev[0], b[0])
                min_w = min(prev[2], b[2])
                vertically_stacked = (prev[1] + prev[3] <= b[1] + 8) or (b[1] + b[3] <= prev[1] + 8)
                if vertically_stacked and overlap_x > 0.25 * min_w:
                    new_x = min(prev[0], b[0])
                    new_y = min(prev[1], b[1])
                    new_w = max(prev[0] + prev[2], b[0] + b[2]) - new_x
                    new_h = max(prev[1] + prev[3], b[1] + b[3]) - new_y
                    merged_boxes[-1] = (new_x, new_y, new_w, new_h)
                else:
                    merged_boxes.append(b)

        # Sort bounding boxes left-to-right
        sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])
        return sorted_boxes, has_dot, is_uncertain

    def _attempt_vertical_split(
        self,
        thresh: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Split touching digits using vertical projection profile valley detection."""
        crop = thresh[y : y + h, x : x + w]
        proj = np.sum(crop > 0, axis=0)

        start_col = int(w * 0.30)
        end_col = int(w * 0.70)
        if end_col <= start_col:
            return [(x, y, w, h)]

        middle_proj = proj[start_col:end_col]
        min_idx = np.argmin(middle_proj)
        min_val = middle_proj[min_idx]
        split_x = start_col + min_idx

        # If valley is sufficiently deep or component is wide (aspect >= 1.25)
        aspect = w / float(max(1, h))
        max_proj = float(np.max(proj)) if len(proj) > 0 else 1.0
        if min_val <= max(2, int(h * 0.25)) or (aspect >= 1.25 and min_val <= max_proj * 0.75):
            box1 = (x, y, split_x, h)
            box2 = (x + split_x, y, w - split_x, h)
            return [box1, box2]

        return [(x, y, w, h)]

    def _preprocess_for_onnx(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Convert digit crop to model's expected shape (1, 1, 28, 28) and float32 [0.0, 1.0]."""
        if crop is None or crop.size == 0:
            return None

        coords = cv2.findNonZero(crop)
        if coords is None:
            return None
        tx, ty, tw, th = cv2.boundingRect(coords)
        if tw <= 0 or th <= 0:
            return None
        tight = crop[ty : ty + th, tx : tx + tw]

        # Scale preserving aspect ratio so largest dimension is 20 pixels
        scale = float(self.INNER_SIZE) / max(tw, th)
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))
        resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center in 28x28 canvas
        canvas = np.zeros(self.TARGET_SIZE, dtype=np.float32)
        pad_x = (self.TARGET_SIZE[1] - new_w) // 2
        pad_y = (self.TARGET_SIZE[0] - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized / 255.0

        # Reshape to (1, 1, 28, 28) float32
        tensor = canvas[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return tensor

    def _classify_char_crop(
        self,
        char_crop: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> Tuple[str, float, float, bool, str]:
        """Run ONNX Runtime inference on a single segmented character."""
        tensor = self._preprocess_for_onnx(char_crop)
        if tensor is None:
            return "0", 0.0, 0.0, True, "Preprocessing failed"

        # Validate tensor shape
        if tensor.shape != (1, 1, 28, 28) or tensor.dtype != np.float32:
            return "0", 0.0, 0.0, True, f"Invalid tensor shape {tensor.shape}"

        try:
            # Execute inference on CPU session
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            raw_out = outputs[0][0]  # Shape: (10,)

            # Check if model outputs raw logits or normalized softmax probabilities
            if np.any(raw_out < 0.0) or abs(float(np.sum(raw_out)) - 1.0) > 0.05:
                shift = raw_out - np.max(raw_out)
                exp_vals = np.exp(shift)
                probs = exp_vals / np.sum(exp_vals)
            else:
                probs = raw_out

            ranked = np.argsort(probs)[::-1]
            best_idx = ranked[0]
            second_idx = ranked[1]

            best_digit = str(best_idx)
            second_digit = str(second_idx)
            best_score = float(probs[best_idx])
            second_score = float(probs[second_idx])
            margin = best_score - second_score

            is_ambiguous = (
                best_score < self.min_confidence
                or margin < self.min_margin
            )

            note = f"top={best_digit}:{best_score:.2f}, 2nd={second_digit}:{second_score:.2f}, margin={margin:.2f}"
            return best_digit, best_score, margin, is_ambiguous, note

        except Exception as exc:
            return "0", 0.0, 0.0, True, f"Inference error: {exc}"
