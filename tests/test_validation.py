"""Unit tests for marks validation, ambiguity gating, and duplicate detection."""

import pytest
from src.marks_validator import MarksValidator, VerificationStatus, ValidatedMark
from src.digit_recognizer import DigitRecognitionResult
from src.review_manager import ReviewManager, StudentAssessmentRecord


@pytest.fixture
def validator():
    return MarksValidator(confidence_threshold=0.65)


def test_valid_numeric_mark(validator):
    """Test valid mark within permitted range."""
    rec = DigitRecognitionResult(
        value_str="8",
        numeric_value=8.0,
        confidence=0.88,
        is_blank=False,
        is_ambiguous=False,
        char_candidates=["8"],
        explanation="Clear match",
    )
    res = validator.validate("q1", "Q1", max_marks=10.0, step=0.5, rec_result=rec)
    assert res.status == VerificationStatus.VALIDATED
    assert res.verified_val == 8.0
    assert res.extracted_val == 8.0


def test_mark_exceeding_max(validator):
    """Test mark exceeding question maximum is flagged invalid."""
    rec = DigitRecognitionResult(
        value_str="12",
        numeric_value=12.0,
        confidence=0.85,
        is_blank=False,
        is_ambiguous=False,
        char_candidates=["1", "2"],
        explanation="Candidate 12",
    )
    res = validator.validate("q1", "Q1", max_marks=10.0, step=0.5, rec_result=rec)
    assert res.status == VerificationStatus.INVALID
    assert res.verified_val is None
    assert "outside permitted range" in res.validation_message


def test_negative_mark(validator):
    """Test negative values are invalid."""
    rec = DigitRecognitionResult(
        value_str="-2",
        numeric_value=-2.0,
        confidence=0.70,
        is_blank=False,
        is_ambiguous=False,
        char_candidates=[],
        explanation="Negative value",
    )
    res = validator.validate("q1", "Q1", max_marks=10.0, step=0.5, rec_result=rec)
    assert res.status == VerificationStatus.INVALID


def test_decimal_step_increment(validator):
    """Test fractional / decimal increments."""
    # 7.5 on 0.5 step is valid
    rec_valid = DigitRecognitionResult(
        value_str="7.5",
        numeric_value=7.5,
        confidence=0.80,
        is_blank=False,
        is_ambiguous=False,
        char_candidates=["7", "5"],
    )
    res1 = validator.validate("q1", "Q1", max_marks=10.0, step=0.5, rec_result=rec_valid)
    assert res1.status == VerificationStatus.VALIDATED
    assert res1.verified_val == 7.5

    # 7.3 on 0.5 step is invalid increment
    rec_invalid = DigitRecognitionResult(
        value_str="7.3",
        numeric_value=7.3,
        confidence=0.80,
        is_blank=False,
        is_ambiguous=False,
        char_candidates=["7", "3"],
    )
    res2 = validator.validate("q1", "Q1", max_marks=10.0, step=0.5, rec_result=rec_invalid)
    assert res2.status == VerificationStatus.INVALID


def test_blank_field_detection(validator):
    """Test blank field receives BLANK status without defaulting to zero."""
    rec_blank = DigitRecognitionResult(
        value_str=None,
        numeric_value=None,
        confidence=0.98,
        is_blank=True,
        is_ambiguous=False,
        char_candidates=[],
        explanation="Blank cell",
    )
    res = validator.validate("q4", "Q4", max_marks=5.0, step=0.5, rec_result=rec_blank)
    assert res.status == VerificationStatus.BLANK
    assert res.verified_val is None  # Must never replace unreadable/blank with 0


def test_uncertain_recognition_needs_review(validator):
    """Test low confidence or ambiguous classification triggers NEEDS_REVIEW."""
    rec_low = DigitRecognitionResult(
        value_str="5",
        numeric_value=5.0,
        confidence=0.55,  # below 0.65 threshold
        is_blank=False,
        is_ambiguous=True,
        char_candidates=["5"],
        explanation="Low margin vs 3",
    )
    res = validator.validate("q2", "Q2", max_marks=5.0, step=0.5, rec_result=rec_low)
    assert res.status == VerificationStatus.NEEDS_REVIEW
    assert res.verified_val is None


def test_manual_correction(validator):
    """Test manual human correction overrides status and updates verified value."""
    rec_low = DigitRecognitionResult(
        value_str="5",
        numeric_value=5.0,
        confidence=0.50,
        is_blank=False,
        is_ambiguous=True,
        char_candidates=["5"],
    )
    initial_mark = validator.validate("q2", "Q2", max_marks=5.0, step=0.5, rec_result=rec_low)
    assert initial_mark.status == VerificationStatus.NEEDS_REVIEW

    # Correct manually to 4.5
    corrected = validator.apply_correction(initial_mark, "4.5", note="Teacher verified")
    assert corrected.status == VerificationStatus.CORRECTED_MANUALLY
    assert corrected.verified_val == 4.5
    assert corrected.manual_note == "Teacher verified"


def test_total_calculation_and_duplicates(validator):
    """Test total marks calculation and duplicate record warning."""
    rm = ReviewManager(validator=validator)

    # Record 1
    m1 = ValidatedMark("q1", "Q1", 5.0, 0.5, "4", 4.0, 4.0, 0.9, VerificationStatus.VALIDATED, "")
    m2 = ValidatedMark("q2", "Q2", 5.0, 0.5, "5", 5.0, 5.0, 0.9, VerificationStatus.VALIDATED, "")
    rec1 = StudentAssessmentRecord(
        student_id="1MS21CS001",
        subject="OS",
        assessment="IA-1",
        marks={"q1": m1, "q2": m2},
    )
    rm.add_record(rec1)
    assert rec1.total_marks == 9.0
    assert rec1.is_fully_verified is True

    # Record 2 with identical Student ID and Assessment (Duplicate)
    rec2 = StudentAssessmentRecord(
        student_id="1MS21CS001",
        subject="OS",
        assessment="IA-1",
        marks={"q1": m1},
    )
    rm.add_record(rec2)

    dups = rm.find_duplicates()
    assert len(dups) == 1
    dup_id, dup_ass, dup_sub, ids = dups[0]
    assert dup_id == "1MS21CS001"
    assert len(ids) == 2
