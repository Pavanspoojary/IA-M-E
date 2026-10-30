"""End-to-end integration tests on synthetic examination bluebooks."""

from pathlib import Path
import pytest
import numpy as np

from src.file_loader import FileLoader
from src.pdf_processor import PDFProcessor
from src.image_preprocessor import ImagePreprocessor
from src.template_manager import TemplateManager
from src.region_extractor import RegionExtractor
from src.digit_recognizer import DigitRecognizer
from src.marks_validator import MarksValidator, VerificationStatus
from src.review_manager import ReviewManager, StudentAssessmentRecord
from src.excel_exporter import ExcelExporter


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture
def template():
    config_path = Path(__file__).resolve().parent.parent / "config" / "template.json"
    tm = TemplateManager()
    return tm.load_from_json(config_path)


def test_pdf_conversion_and_rendering(sample_dir):
    """Test PDF processor correctly renders page 2 from multi-page document."""
    pdf_path = sample_dir / "student_sample_4_multipage.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not found.")

    processor = PDFProcessor()
    page_count = processor.get_page_count(str(pdf_path))
    assert page_count == 2

    # Render sheet 2
    sheet2_img = processor.render_page(str(pdf_path), page_number=2)
    assert sheet2_img is not None
    assert sheet2_img.shape[0] > 1000
    assert sheet2_img.shape[1] > 800


def test_end_to_end_student_sample_1(sample_dir, template):
    """Process student_sample_1.png end to end through all stages."""
    img_path = sample_dir / "student_sample_1.png"
    if not img_path.exists():
        pytest.skip("Sample 1 not found.")

    # 1. Load File
    loader = FileLoader()
    docs = loader.load_from_path(img_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.status == "ready"

    # 2. Preprocess
    preprocessor = ImagePreprocessor()
    prep_res = preprocessor.preprocess(doc.image_bgr)
    assert prep_res.success
    assert prep_res.aligned_bgr.shape == (1754, 1240, 3)

    # 3. Extract Regions
    extractor = RegionExtractor()
    extracted = extractor.extract_all(prep_res.aligned_bgr, prep_res.thresholded_inv, template)
    assert "q1" in extracted
    assert "q8" in extracted

    # 4. Recognize Digits & Validate
    recognizer = DigitRecognizer()
    validator = MarksValidator()
    marks_dict = {}

    for q in template.questions:
        reg = extracted[q.id]
        rec = recognizer.recognize_region(reg.crop_thresh_inv, max_marks=q.max_marks, step=q.step)
        v_mark = validator.validate(
            question_id=q.id,
            question_label=q.label,
            max_marks=q.max_marks,
            step=q.step,
            rec_result=rec,
            crop_bgr=reg.crop_bgr,
            crop_thresh_inv=reg.crop_thresh_inv,
        )
        marks_dict[q.id] = v_mark

    # 5. Review Manager
    rm = ReviewManager(validator=validator)
    rec_obj = StudentAssessmentRecord(
        source_filename=img_path.name,
        page_number=1,
        student_id="1MS21CS042",
        student_name="Aarav Sharma",
        subject="OS",
        assessment="IA-1",
        marks=marks_dict,
    )
    rm.add_record(rec_obj)

    # All marks should be recognized as valid numbers
    assert len(rec_obj.marks) == 8
    assert rec_obj.total_marks is not None
    assert rec_obj.total_marks > 20.0

    # 6. Excel Export
    exporter = ExcelExporter()
    wb = exporter.generate_workbook(rm.records, template=template)
    assert "Marks Summary" in wb.sheetnames
    assert wb["Marks Summary"].max_row == 2


def test_edge_case_sample_5(sample_dir, template):
    """Verify that sample 5 detects invalid mark >10 and blank field."""
    img_path = sample_dir / "student_sample_5_review_needed.png"
    if not img_path.exists():
        pytest.skip("Sample 5 not found.")

    loader = FileLoader()
    doc = loader.load_from_path(img_path)[0]

    preprocessor = ImagePreprocessor()
    prep_res = preprocessor.preprocess(doc.image_bgr)

    extractor = RegionExtractor()
    extracted = extractor.extract_all(prep_res.aligned_bgr, prep_res.thresholded_inv, template)

    recognizer = DigitRecognizer()
    validator = MarksValidator()

    # Q1 was drawn with 15 (exceeds max 5.0) -> must be INVALID
    q1_reg = extracted["q1"]
    q1_rec = recognizer.recognize_region(q1_reg.crop_thresh_inv, max_marks=5.0, step=0.5)
    q1_val = validator.validate("q1", "Q1", max_marks=5.0, step=0.5, rec_result=q1_rec)
    assert q1_val.status == VerificationStatus.INVALID

    # Q4 was left blank -> must be BLANK
    q4_reg = extracted["q4"]
    q4_rec = recognizer.recognize_region(q4_reg.crop_thresh_inv, max_marks=5.0, step=0.5)
    q4_val = validator.validate("q4", "Q4", max_marks=5.0, step=0.5, rec_result=q4_rec)
    assert q4_val.status == VerificationStatus.BLANK
