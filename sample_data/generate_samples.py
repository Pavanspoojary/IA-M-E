"""Synthetic IA Bluebook generator for testing and validation.

Generates realistic blank and filled IA bluebook second sheets with tables,
header metadata, and simulated handwritten marks for comprehensive offline testing.
"""

from pathlib import Path
import cv2
import numpy as np
import pymupdf


def draw_bluebook_sheet(
    student_id: str = "",
    student_name: str = "",
    subject: str = "Operating Systems (21CS42)",
    assessment: str = "IA Test - 1",
    marks_dict: dict = None,
    width: int = 1240,
    height: int = 1754,
    handwriting_style: str = "normal",
) -> np.ndarray:
    """Draw a realistic college bluebook second sheet."""
    # Create white page canvas with subtle paper texture
    img = np.full((height, width, 3), 252, dtype=np.uint8)

    # Outer border line
    cv2.rectangle(img, (50, 50), (width - 50, height - 50), (120, 120, 120), 2)

    # Header section
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    font_reg = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", (180, 110), font_bold, 0.9, (20, 20, 20), 2)
    cv2.putText(img, "CONTINUOUS INTERNAL ASSESSMENT (IA) EVALUATION SHEET", (210, 155), font_bold, 0.75, (40, 40, 40), 2)

    # Sub-header rules
    cv2.line(img, (80, 180), (width - 80, 180), (100, 100, 100), 2)

    # Metadata fields
    cv2.putText(img, f"Subject: {subject}", (90, 230), font_reg, 0.7, (30, 30, 30), 2)
    cv2.putText(img, f"Assessment: {assessment}", (90, 280), font_reg, 0.7, (30, 30, 30), 2)
    cv2.putText(img, f"Student Name: {student_name}", (90, 330), font_reg, 0.7, (30, 30, 30), 2)

    # Student ID / USN Box
    cv2.putText(img, "USN / Reg No:", (750, 250), font_bold, 0.7, (20, 20, 20), 2)
    # Box matching template: x=750, y=280, w=350, h=70
    cv2.rectangle(img, (750, 280), (1100, 350), (60, 60, 60), 2)
    if student_id:
        cv2.putText(img, student_id, (770, 330), font_bold, 1.0, (15, 15, 120), 2)

    cv2.line(img, (80, 380), (width - 80, 380), (100, 100, 100), 2)

    # Marks Section Title
    cv2.putText(img, "SCHEME OF VALUATION & MARKS AWARDED", (360, 430), font_bold, 0.8, (10, 10, 10), 2)

    # Table Headers
    # Part A Table (Left)
    cv2.rectangle(img, (120, 460), (560, 940), (40, 40, 40), 2)
    cv2.putText(img, "PART A (Max: 5 per Q)", (180, 495), font_bold, 0.65, (20, 20, 20), 2)
    cv2.line(img, (120, 510), (560, 510), (40, 40, 40), 2)
    cv2.line(img, (380, 510), (380, 940), (40, 40, 40), 2)  # Question vs Marks split

    # Part B Table (Right)
    cv2.rectangle(img, (640, 460), (1080, 940), (40, 40, 40), 2)
    cv2.putText(img, "PART B (Max: 10 per Q)", (700, 495), font_bold, 0.65, (20, 20, 20), 2)
    cv2.line(img, (640, 510), (1080, 510), (40, 40, 40), 2)
    cv2.line(img, (900, 510), (900, 940), (40, 40, 40), 2)

    left_labels = [("Q1 (a)", "q1", 520), ("Q2 (b)", "q2", 630), ("Q3 (a)", "q3", 740), ("Q4 (b)", "q4", 850)]
    right_labels = [("Q5", "q5", 520), ("Q6", "q6", 630), ("Q7", "q7", 740), ("Q8", "q8", 850)]

    # Draw horizontal grid lines and labels
    for lbl, q_id, y in left_labels:
        cv2.putText(img, lbl, (160, y + 42), font_bold, 0.75, (30, 30, 30), 2)
        cv2.line(img, (120, y + 65), (560, y + 65), (100, 100, 100), 1)

    for lbl, q_id, y in right_labels:
        cv2.putText(img, lbl, (720, y + 42), font_bold, 0.75, (30, 30, 30), 2)
        cv2.line(img, (640, y + 65), (1080, y + 65), (100, 100, 100), 1)

    # Evaluator Signature & Notes at bottom
    cv2.putText(img, "Total Marks Awarded (Verified): ____________", (120, 1050), font_reg, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "Evaluator Signature: __________________", (650, 1050), font_reg, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "Note: Half-marks (0.5) allowed. Cross-verify totals before submission.", (120, 1150), font_reg, 0.6, (80, 80, 80), 1)

    # Draw Marks if provided
    # Left questions: x=420, y=520, 630, 740, 850, w=130, h=65
    # Right questions: x=940, y=520, 630, 740, 850, w=130, h=65
    q_coords = {
        "q1": (435, 565),
        "q2": (435, 675),
        "q3": (435, 785),
        "q4": (435, 895),
        "q5": (955, 565),
        "q6": (955, 675),
        "q7": (955, 785),
        "q8": (955, 895),
    }

    if marks_dict:
        # Handwriting style selection
        ink_color = (130, 20, 20)  # Red / dark examiner ballpoint ink
        hw_font = cv2.FONT_HERSHEY_SIMPLEX if handwriting_style == "normal" else cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        thickness = 2

        for q_id, val_str in marks_dict.items():
            if val_str is None or val_str == "":
                continue
            if q_id in q_coords:
                cell_x = 420 if q_id in ("q1", "q2", "q3", "q4") else 940
                base_y = q_coords[q_id][1]
                (tw, th), _ = cv2.getTextSize(str(val_str), hw_font, 1.15, thickness)
                text_x = cell_x + max(10, (130 - tw) // 2)
                cv2.putText(img, str(val_str), (text_x, base_y), hw_font, 1.15, ink_color, thickness, cv2.LINE_AA)

    return img


def generate_all_samples(output_dir: Path) -> None:
    """Generate comprehensive set of test bluebook files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Blank Template Reference Sheet
    blank = draw_bluebook_sheet()
    cv2.imwrite(str(output_dir / "clean_bluebook_template.png"), blank)

    # 2. Student Sample 1 - Standard full evaluation
    s1_marks = {
        "q1": "4",
        "q2": "5",
        "q3": "3",
        "q4": "4",
        "q5": "9",
        "q6": "8",
        "q7": "10",
        "q8": "7",
    }
    s1 = draw_bluebook_sheet(
        student_id="1MS21CS042",
        student_name="Aarav Sharma",
        marks_dict=s1_marks,
    )
    cv2.imwrite(str(output_dir / "student_sample_1.png"), s1)

    # 3. Student Sample 2 - Skewed (+3.8 degrees) and different resolution
    s2_marks = {
        "q1": "5",
        "q2": "4",
        "q3": "2",
        "q4": "5",
        "q5": "7",
        "q6": "6",
        "q7": "8",
        "q8": "9",
    }
    s2 = draw_bluebook_sheet(
        student_id="1MS21CS058",
        student_name="Bhavna Patel",
        marks_dict=s2_marks,
    )
    # Apply minor rotation
    center = (s2.shape[1] // 2, s2.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 3.8, 1.0)
    s2_skewed = cv2.warpAffine(s2, rot_mat, (s2.shape[1], s2.shape[0]), borderValue=(240, 240, 240))
    # Resize slightly
    s2_resized = cv2.resize(s2_skewed, (1000, 1414), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_dir / "student_sample_2_skewed.jpg"), s2_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 4. Student Sample 3 - Perspective distortion (photographed from phone)
    s3_marks = {
        "q1": "3",
        "q2": "4",
        "q3": "5",
        "q4": "2",
        "q5": "8",
        "q6": "7",
        "q7": "6",
        "q8": "8",
    }
    s3 = draw_bluebook_sheet(
        student_id="1MS21CS089",
        student_name="Chetan Kumar",
        marks_dict=s3_marks,
    )
    # Perspective distortion
    h, w = s3.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_pts = np.float32([[40, 30], [w - 30, 60], [w - 60, h - 40], [30, h - 20]])
    persp_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    s3_persp = cv2.warpPerspective(s3, persp_mat, (w, h), borderValue=(220, 220, 220))
    cv2.imwrite(str(output_dir / "student_sample_3_perspective.png"), s3_persp)

    # 5. Student Sample 4 - Multi-page PDF (Page 1: Cover, Page 2: IA Sheet)
    doc = pymupdf.open()
    # Page 1: Front Cover
    cover = np.full((1754, 1240, 3), 245, dtype=np.uint8)
    cv2.rectangle(cover, (80, 80), (1160, 1674), (60, 60, 60), 3)
    cv2.putText(cover, "ENGINEERING COLLEGE EXAMINATION BOOKLET", (160, 300), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(cover, "FIRST SHEET: CANDIDATE COVER PAGE", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(cover, "USN: 1MS21CS104", (200, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(cover, "(Turn to sheet 2 for marks grid)", (350, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # Convert cover to PDF page
    is_success, cover_buf = cv2.imencode(".png", cover)
    pdf_page1 = doc.new_page(width=1240 * 72 / 150, height=1754 * 72 / 150)
    pdf_page1.insert_image(pdf_page1.rect, stream=cover_buf.tobytes())

    # Page 2: Marks Sheet 2
    s4_marks = {
        "q1": "4",
        "q2": "5",
        "q3": "4",
        "q4": "3",
        "q5": "10",
        "q6": "9",
        "q7": "8",
        "q8": "10",
    }
    s4 = draw_bluebook_sheet(
        student_id="1MS21CS104",
        student_name="Divya Rao",
        marks_dict=s4_marks,
    )
    is_success, s4_buf = cv2.imencode(".png", s4)
    pdf_page2 = doc.new_page(width=1240 * 72 / 150, height=1754 * 72 / 150)
    pdf_page2.insert_image(pdf_page2.rect, stream=s4_buf.tobytes())

    doc.save(str(output_dir / "student_sample_4_multipage.pdf"))
    doc.close()

    # 6. Student Sample 5 - Edge cases: invalid mark (>10 for Q1), blank Q4, and unreadable smudge
    s5_marks = {
        "q1": "15",  # Exceeds max_marks of 5.0 -> Should trigger INVALID status
        "q2": "4",
        "q3": "3",
        "q4": "",    # Blank question -> Should trigger BLANK status
        "q5": "8",
        "q6": "7",
        "q7": "9",
        "q8": "6",
    }
    s5 = draw_bluebook_sheet(
        student_id="1MS21CS115",
        student_name="Esha Verma",
        marks_dict=s5_marks,
    )
    cv2.imwrite(str(output_dir / "student_sample_5_review_needed.png"), s5)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    generate_all_samples(out_dir)
    print("Generated all test samples successfully in:", out_dir)
