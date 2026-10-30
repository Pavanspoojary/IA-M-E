"""Review manager module for student assessment records and batch lifecycle.

Tracks document verification statuses, audit modifications, student metadata,
duplicate detection, and export readiness gates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import uuid

from src.marks_validator import ValidatedMark, VerificationStatus, MarksValidator


@dataclass
class StudentAssessmentRecord:
    """Represents a student's assessed bluebook page with marks and metadata."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_filename: str = ""
    page_number: int = 2
    student_id: str = "PENDING_ID"
    student_name: str = ""
    subject: str = "Subject"
    assessment: str = "IA-1"
    marks: Dict[str, ValidatedMark] = field(default_factory=dict)
    total_marks: Optional[float] = None
    is_fully_verified: bool = False
    processing_timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    log_message: str = "Processed"
    needs_review_count: int = 0
    invalid_count: int = 0
    model_version: str = "IA_DigitClassifier_ONNX-v1.0.0"


class ReviewManager:
    """Orchestrates batch verification, duplicate detection, and editing workflows."""

    def __init__(self, validator: Optional[MarksValidator] = None):
        """Initialize ReviewManager.

        Args:
            validator: MarksValidator instance.
        """
        self.validator = validator or MarksValidator()
        self.records: List[StudentAssessmentRecord] = []

    def clear(self) -> None:
        """Clear all stored records."""
        self.records.clear()

    def add_record(self, record: StudentAssessmentRecord) -> None:
        """Add a processed record and compute verification totals."""
        self.recompute_record_total(record)
        self.records.append(record)

    def get_record(self, record_id: str) -> Optional[StudentAssessmentRecord]:
        """Find record by ID."""
        for r in self.records:
            if r.record_id == record_id:
                return r
        return None

    def update_mark(
        self,
        record_id: str,
        question_id: str,
        new_val_str: str,
        user_note: str = "Manual evaluator review",
    ) -> bool:
        """Update a specific question mark with manual correction."""
        rec = self.get_record(record_id)
        if not rec or question_id not in rec.marks:
            return False

        current_mark = rec.marks[question_id]
        updated = self.validator.apply_correction(current_mark, new_val_str, note=user_note)
        rec.marks[question_id] = updated

        self.recompute_record_total(rec)
        return True

    def update_student_info(
        self,
        record_id: str,
        student_id: str,
        student_name: Optional[str] = None,
        subject: Optional[str] = None,
        assessment: Optional[str] = None,
    ) -> bool:
        """Update student metadata for a record."""
        rec = self.get_record(record_id)
        if not rec:
            return False

        if student_id:
            rec.student_id = student_id.strip()
        if student_name is not None:
            rec.student_name = student_name.strip()
        if subject is not None:
            rec.subject = subject.strip()
        if assessment is not None:
            rec.assessment = assessment.strip()

        return True

    def recompute_record_total(self, record: StudentAssessmentRecord) -> None:
        """Recompute total marks and verification completion for a record."""
        marks_list = list(record.marks.values())
        tot, is_comp, msg = self.validator.calculate_total(marks_list)

        record.total_marks = tot
        record.is_fully_verified = is_comp
        record.needs_review_count = sum(
            1 for m in marks_list if m.status == VerificationStatus.NEEDS_REVIEW
        )
        record.invalid_count = sum(
            1 for m in marks_list if m.status == VerificationStatus.INVALID
        )

        if not is_comp:
            record.log_message = msg
        else:
            record.log_message = "All marks verified and consistent."

    def find_duplicates(self) -> List[Tuple[str, str, str, List[str]]]:
        """Find duplicate records with matching (student_id, assessment, subject).

        Returns:
            List of tuples: (student_id, assessment, subject, list_of_matching_record_ids).
        """
        seen: Dict[Tuple[str, str, str], List[str]] = {}

        for r in self.records:
            s_id = r.student_id.strip().upper()
            ass = r.assessment.strip().upper()
            sub = r.subject.strip().upper()
            # Ignore placeholder IDs
            if not s_id or s_id in ("PENDING_ID", "UNKNOWN", "UNREADABLE"):
                continue

            key = (s_id, ass, sub)
            seen.setdefault(key, []).append(r.record_id)

        duplicates = [
            (key[0], key[1], key[2], ids)
            for key, ids in seen.items()
            if len(ids) > 1
        ]
        return duplicates

    def get_summary_statistics(self) -> Dict[str, int]:
        """Calculate aggregate batch statistics across all records."""
        total_records = len(self.records)
        total_fields = sum(len(r.marks) for r in self.records)

        counts = {
            "total_records": total_records,
            "total_fields": total_fields,
            "validated": 0,
            "needs_review": 0,
            "corrected": 0,
            "blank": 0,
            "invalid": 0,
            "duplicates": len(self.find_duplicates()),
            "fully_verified_records": sum(1 for r in self.records if r.is_fully_verified),
        }

        for r in self.records:
            for m in r.marks.values():
                if m.status == VerificationStatus.VALIDATED:
                    counts["validated"] += 1
                elif m.status == VerificationStatus.NEEDS_REVIEW:
                    counts["needs_review"] += 1
                elif m.status == VerificationStatus.CORRECTED_MANUALLY:
                    counts["corrected"] += 1
                elif m.status == VerificationStatus.BLANK:
                    counts["blank"] += 1
                elif m.status == VerificationStatus.INVALID:
                    counts["invalid"] += 1

        return counts
