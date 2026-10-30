"""Template manager for bluebook fixed-coordinate configurations.

Stores and manipulates reference coordinates, question metadata,
maximum marks, and student identifier regions with resolution normalization.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json


@dataclass
class QuestionRegion:
    """Configuration for an individual question's marks cell."""

    id: str
    label: str
    max_marks: float
    step: float = 0.5
    # Pixel coordinates on reference resolution
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 50
    # Normalized coordinates [x, y, w, h] as floats between 0.0 and 1.0
    normalized_bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def update_normalized(self, ref_w: int, ref_h: int) -> None:
        """Update normalized coordinates from current pixel coordinates."""
        if ref_w > 0 and ref_h > 0:
            self.normalized_bbox = [
                round(self.x / ref_w, 5),
                round(self.y / ref_h, 5),
                round(self.w / ref_w, 5),
                round(self.h / ref_h, 5),
            ]

    def update_pixels(self, ref_w: int, ref_h: int) -> None:
        """Update pixel coordinates from normalized coordinates."""
        if self.normalized_bbox and len(self.normalized_bbox) == 4:
            nx, ny, nw, nh = self.normalized_bbox
            self.x = int(round(nx * ref_w))
            self.y = int(round(ny * ref_h))
            self.w = int(round(nw * ref_w))
            self.h = int(round(nh * ref_h))


@dataclass
class StudentIdRegion:
    """Configuration for student identifier / register number region."""

    x: int = 0
    y: int = 0
    w: int = 300
    h: int = 60
    normalized_bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def update_normalized(self, ref_w: int, ref_h: int) -> None:
        """Update normalized coordinates from reference pixels."""
        if ref_w > 0 and ref_h > 0:
            self.normalized_bbox = [
                round(self.x / ref_w, 5),
                round(self.y / ref_h, 5),
                round(self.w / ref_w, 5),
                round(self.h / ref_h, 5),
            ]

    def update_pixels(self, ref_w: int, ref_h: int) -> None:
        """Update pixel coordinates from normalized coordinates."""
        if self.normalized_bbox and len(self.normalized_bbox) == 4:
            nx, ny, nw, nh = self.normalized_bbox
            self.x = int(round(nx * ref_w))
            self.y = int(round(ny * ref_h))
            self.w = int(round(nw * ref_w))
            self.h = int(round(nh * ref_h))


@dataclass
class BluebookTemplate:
    """Master template specification for an IA Bluebook second sheet layout."""

    template_name: str = "Standard IA Bluebook Sheet 2"
    description: str = "Standard internal assessment question-wise marks grid"
    college_name: str = "Engineering College"
    reference_width: int = 1240
    reference_height: int = 1754
    student_id_region: Optional[StudentIdRegion] = None
    questions: List[QuestionRegion] = field(default_factory=list)

    @property
    def total_max_marks(self) -> float:
        """Sum of all configured maximum marks."""
        return sum(q.max_marks for q in self.questions)

    def get_pixel_bbox_for_resolution(
        self,
        region: QuestionRegion | StudentIdRegion,
        actual_width: int,
        actual_height: int,
    ) -> Tuple[int, int, int, int]:
        """Convert normalized region coordinates to pixel bounds for any resolution."""
        if region.normalized_bbox and len(region.normalized_bbox) == 4 and any(region.normalized_bbox):
            nx, ny, nw, nh = region.normalized_bbox
            x = int(round(nx * actual_width))
            y = int(round(ny * actual_height))
            w = int(round(nw * actual_width))
            h = int(round(nh * actual_height))
        else:
            # Scale from reference resolution
            scale_x = actual_width / self.reference_width if self.reference_width > 0 else 1.0
            scale_y = actual_height / self.reference_height if self.reference_height > 0 else 1.0
            x = int(round(region.x * scale_x))
            y = int(round(region.y * scale_y))
            w = int(round(region.w * scale_x))
            h = int(round(region.h * scale_y))

        # Clamp inside boundaries
        x = max(0, min(x, actual_width - 1))
        y = max(0, min(y, actual_height - 1))
        w = max(1, min(w, actual_width - x))
        h = max(1, min(h, actual_height - y))
        return x, y, w, h


class TemplateManager:
    """Handles loading, saving, creating, and modifying bluebook templates."""

    def __init__(self, default_config_path: Optional[str | Path] = None):
        """Initialize TemplateManager.

        Args:
            default_config_path: Path to default JSON template.
        """
        self.config_path = Path(default_config_path) if default_config_path else None

    def create_default_template(self) -> BluebookTemplate:
        """Create a standard 8-question IA marks table template."""
        template = BluebookTemplate(
            template_name="Standard IA Bluebook Sheet 2",
            description="8-Question Internal Assessment Marks Grid",
            college_name="Engineering College Assessment",
            reference_width=1240,
            reference_height=1754,
            student_id_region=StudentIdRegion(
                x=750, y=280, w=350, h=70
            ),
        )
        template.student_id_region.update_normalized(1240, 1754)

        # Standard grid location for 8 questions in Part A and Part B
        # Column X positions: Questions Q1-Q4 at left side, Q5-Q8 at right side
        left_col_x = 420
        right_col_x = 940
        start_y = 520
        row_gap = 110

        questions_data = [
            ("q1", "Q1", 5.0, 0.5, left_col_x, start_y + 0 * row_gap, 130, 65),
            ("q2", "Q2", 5.0, 0.5, left_col_x, start_y + 1 * row_gap, 130, 65),
            ("q3", "Q3", 5.0, 0.5, left_col_x, start_y + 2 * row_gap, 130, 65),
            ("q4", "Q4", 5.0, 0.5, left_col_x, start_y + 3 * row_gap, 130, 65),
            ("q5", "Q5", 10.0, 0.5, right_col_x, start_y + 0 * row_gap, 130, 65),
            ("q6", "Q6", 10.0, 0.5, right_col_x, start_y + 1 * row_gap, 130, 65),
            ("q7", "Q7", 10.0, 0.5, right_col_x, start_y + 2 * row_gap, 130, 65),
            ("q8", "Q8", 10.0, 0.5, right_col_x, start_y + 3 * row_gap, 130, 65),
        ]

        for q_id, label, max_m, step, x, y, w, h in questions_data:
            q = QuestionRegion(
                id=q_id,
                label=label,
                max_marks=max_m,
                step=step,
                x=x,
                y=y,
                w=w,
                h=h,
            )
            q.update_normalized(1240, 1754)
            template.questions.append(q)

        return template

    def load_from_json(self, file_path_or_str: str | Path) -> BluebookTemplate:
        """Load template from a JSON file."""
        path = Path(file_path_or_str)
        if not path.exists():
            raise FileNotFoundError(f"Template configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.from_dict(data)

    def save_to_json(self, template: BluebookTemplate, file_path: str | Path) -> None:
        """Save a BluebookTemplate instance to a formatted JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(template)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def to_dict(self, template: BluebookTemplate) -> Dict[str, Any]:
        """Convert BluebookTemplate to dictionary representation."""
        # Ensure all normalized coords are up-to-date
        ref_w = template.reference_width
        ref_h = template.reference_height

        if template.student_id_region:
            template.student_id_region.update_normalized(ref_w, ref_h)

        for q in template.questions:
            q.update_normalized(ref_w, ref_h)

        return asdict(template)

    def from_dict(self, data: Dict[str, Any]) -> BluebookTemplate:
        """Construct BluebookTemplate from dictionary data."""
        ref_w = data.get("reference_width", 1240)
        ref_h = data.get("reference_height", 1754)

        student_id_reg = None
        if data.get("student_id_region"):
            s_data = data["student_id_region"]
            student_id_reg = StudentIdRegion(**s_data)
            if not student_id_reg.normalized_bbox or not any(student_id_reg.normalized_bbox):
                student_id_reg.update_normalized(ref_w, ref_h)

        questions: List[QuestionRegion] = []
        for q_data in data.get("questions", []):
            q = QuestionRegion(**q_data)
            if not q.normalized_bbox or not any(q.normalized_bbox):
                q.update_normalized(ref_w, ref_h)
            questions.append(q)

        return BluebookTemplate(
            template_name=data.get("template_name", "IA Template"),
            description=data.get("description", ""),
            college_name=data.get("college_name", ""),
            reference_width=ref_w,
            reference_height=ref_h,
            student_id_region=student_id_reg,
            questions=questions,
        )
