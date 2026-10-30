"""Marks validation and integrity rules module.

Implements deterministic range checking, permitted increment validation,
ambiguity gating, and audit status assignment without guessing.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from src.digit_recognizer import DigitRecognitionResult


class VerificationStatus:
    """Standardized verification status constants."""

    VALIDATED = "Validated"
    VERIFIED_AUTOMATICALLY = "Verified automatically"
    VERIFIED_MANUALLY = "Verified manually"
    NEEDS_REVIEW = "Needs review"
    CORRECTED_MANUALLY = "Corrected manually"
    BLANK = "Blank"
    INVALID = "Invalid"
    UNREADABLE = "Unreadable"


@dataclass
class ValidatedMark:
    """Represents the validated state of a question's marks."""

    question_id: str
    question_label: str
    max_marks: float
    step: float
    extracted_str: Optional[str]
    extracted_val: Optional[float]
    verified_val: Optional[float]
    confidence: float
    status: str
    validation_message: str
    crop_bgr: Optional[np.ndarray] = None
    crop_thresh_inv: Optional[np.ndarray] = None
    manual_note: Optional[str] = None


class MarksValidator:
    """Validates recognized marks against college examination rules."""

    def __init__(self, confidence_threshold: float = 0.65):
        """Initialize MarksValidator.

        Args:
            confidence_threshold: Minimum similarity confidence to mark as VALIDATED without review.
        """
        self.confidence_threshold = confidence_threshold

    def validate(
        self,
        question_id: str,
        question_label: str,
        max_marks: float,
        step: float,
        rec_result: DigitRecognitionResult,
        crop_bgr: Optional[np.ndarray] = None,
        crop_thresh_inv: Optional[np.ndarray] = None,
    ) -> ValidatedMark:
        """Validate raw digit recognition results against schema and grading rules."""
        # Case 1: Blank field
        if rec_result.is_blank:
            return ValidatedMark(
                question_id=question_id,
                question_label=question_label,
                max_marks=max_marks,
                step=step,
                extracted_str="",
                extracted_val=None,
                verified_val=None,
                confidence=rec_result.confidence,
                status=VerificationStatus.BLANK,
                validation_message="No handwriting detected in cell (Blank).",
                crop_bgr=crop_bgr,
                crop_thresh_inv=crop_thresh_inv,
            )

        # Case 2: Unparseable numeric value
        if rec_result.numeric_value is None or rec_result.value_str is None:
            return ValidatedMark(
                question_id=question_id,
                question_label=question_label,
                max_marks=max_marks,
                step=step,
                extracted_str=rec_result.value_str or "unreadable",
                extracted_val=None,
                verified_val=None,
                confidence=rec_result.confidence,
                status=VerificationStatus.INVALID,
                validation_message=f"Non-numeric or unreadable handwriting: {rec_result.explanation}",
                crop_bgr=crop_bgr,
                crop_thresh_inv=crop_thresh_inv,
            )

        val = rec_result.numeric_value

        # Case 3: Out of range (exceeds max marks or negative)
        if val < 0.0 or val > max_marks:
            return ValidatedMark(
                question_id=question_id,
                question_label=question_label,
                max_marks=max_marks,
                step=step,
                extracted_str=rec_result.value_str,
                extracted_val=val,
                verified_val=None,
                confidence=rec_result.confidence,
                status=VerificationStatus.INVALID,
                validation_message=(
                    f"Value {val} outside permitted range [0, {max_marks}]."
                ),
                crop_bgr=crop_bgr,
                crop_thresh_inv=crop_thresh_inv,
            )

        # Case 4: Step / increment check (e.g. 0.5 step only allows multiples of 0.5)
        if step > 0:
            remainder = round(val % step, 4)
            tolerance = 0.001
            if remainder > tolerance and abs(remainder - step) > tolerance:
                return ValidatedMark(
                    question_id=question_id,
                    question_label=question_label,
                    max_marks=max_marks,
                    step=step,
                    extracted_str=rec_result.value_str,
                    extracted_val=val,
                    verified_val=None,
                    confidence=rec_result.confidence,
                    status=VerificationStatus.INVALID,
                    validation_message=(
                        f"Value {val} does not match valid increment of {step}."
                    ),
                    crop_bgr=crop_bgr,
                    crop_thresh_inv=crop_thresh_inv,
                )

        # Case 5: Uncertain recognition or low confidence
        if rec_result.is_ambiguous or rec_result.confidence < self.confidence_threshold:
            return ValidatedMark(
                question_id=question_id,
                question_label=question_label,
                max_marks=max_marks,
                step=step,
                extracted_str=rec_result.value_str,
                extracted_val=val,
                verified_val=None,
                confidence=rec_result.confidence,
                status=VerificationStatus.NEEDS_REVIEW,
                validation_message=(
                    f"Candidate {val} has low confidence ({rec_result.confidence:.2f}): {rec_result.explanation}"
                ),
                crop_bgr=crop_bgr,
                crop_thresh_inv=crop_thresh_inv,
            )

        # Case 6: Fully validated mark
        return ValidatedMark(
            question_id=question_id,
            question_label=question_label,
            max_marks=max_marks,
            step=step,
            extracted_str=rec_result.value_str,
            extracted_val=val,
            verified_val=val,
            confidence=rec_result.confidence,
            status=VerificationStatus.VALIDATED,
            validation_message="Mark verified automatically within permitted range.",
            crop_bgr=crop_bgr,
            crop_thresh_inv=crop_thresh_inv,
        )

    def apply_correction(
        self,
        mark: ValidatedMark,
        corrected_input: str,
        note: Optional[str] = "Manual entry",
    ) -> ValidatedMark:
        """Apply a human correction to a mark field."""
        cleaned = corrected_input.strip()

        # Check if user marked as blank
        if cleaned == "" or cleaned.lower() in ("blank", "-", "na", "n/a"):
            return ValidatedMark(
                question_id=mark.question_id,
                question_label=mark.question_label,
                max_marks=mark.max_marks,
                step=mark.step,
                extracted_str=mark.extracted_str,
                extracted_val=mark.extracted_val,
                verified_val=None,
                confidence=1.0,
                status=VerificationStatus.CORRECTED_MANUALLY,
                validation_message="Marked as blank by evaluator.",
                crop_bgr=mark.crop_bgr,
                crop_thresh_inv=mark.crop_thresh_inv,
                manual_note=note,
            )

        try:
            val = float(cleaned)
        except ValueError:
            return ValidatedMark(
                question_id=mark.question_id,
                question_label=mark.question_label,
                max_marks=mark.max_marks,
                step=mark.step,
                extracted_str=mark.extracted_str,
                extracted_val=mark.extracted_val,
                verified_val=None,
                confidence=0.0,
                status=VerificationStatus.INVALID,
                validation_message=f"Entered value '{corrected_input}' is not numeric.",
                crop_bgr=mark.crop_bgr,
                crop_thresh_inv=mark.crop_thresh_inv,
                manual_note=note,
            )

        if val < 0.0 or val > mark.max_marks:
            return ValidatedMark(
                question_id=mark.question_id,
                question_label=mark.question_label,
                max_marks=mark.max_marks,
                step=mark.step,
                extracted_str=mark.extracted_str,
                extracted_val=mark.extracted_val,
                verified_val=None,
                confidence=0.0,
                status=VerificationStatus.INVALID,
                validation_message=f"Entered value {val} exceeds maximum marks of {mark.max_marks}.",
                crop_bgr=mark.crop_bgr,
                crop_thresh_inv=mark.crop_thresh_inv,
                manual_note=note,
            )

        return ValidatedMark(
            question_id=mark.question_id,
            question_label=mark.question_label,
            max_marks=mark.max_marks,
            step=mark.step,
            extracted_str=mark.extracted_str,
            extracted_val=mark.extracted_val,
            verified_val=val,
            confidence=1.0,
            status=VerificationStatus.CORRECTED_MANUALLY,
            validation_message="Manually verified and confirmed by evaluator.",
            crop_bgr=mark.crop_bgr,
            crop_thresh_inv=mark.crop_thresh_inv,
            manual_note=note,
        )

    def calculate_total(
        self,
        marks: List[ValidatedMark],
    ) -> Tuple[Optional[float], bool, str]:
        """Calculate total marks from verified entries only.

        Returns:
            (total_marks, is_all_verified, status_message)
        """
        unresolved = [
            m for m in marks
            if m.status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.INVALID)
        ]

        # Calculate sum of all non-null verified values
        valid_values = [m.verified_val for m in marks if m.verified_val is not None]
        total = round(sum(valid_values), 2) if valid_values or not marks else 0.0

        if unresolved:
            names = ", ".join(m.question_label for m in unresolved)
            return (
                total,
                False,
                f"Total is provisional: {len(unresolved)} field(s) require review ({names}).",
            )

        return total, True, "All marks verified successfully."
