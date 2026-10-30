"""FastAPI web server for IA Bluebook Marks Extraction System.

A minimal, high-performance web backend serving REST APIs,
processing uploaded bluebook sheets locally, and generating Excel workbooks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import base64
import io
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np

from src.file_loader import FileLoader, LoadedDocument
from src.pdf_processor import PDFProcessor
from src.image_preprocessor import ImagePreprocessor
from src.template_manager import TemplateManager, BluebookTemplate
from src.region_extractor import RegionExtractor
from src.digit_recognizer import DigitRecognizer
from src.marks_validator import MarksValidator, VerificationStatus
from src.review_manager import ReviewManager, StudentAssessmentRecord
from src.excel_exporter import ExcelExporter
from src.processing_pipeline import ProcessingPipeline
from src.eval_sheet_extractor import EvalSheetExtractor


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "template.json"
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"
WEB_DIR = BASE_DIR / "public" if (BASE_DIR / "public").exists() else BASE_DIR / "web"
SAMPLE_DIR = BASE_DIR / "sample_data"

app = FastAPI(title="IA Bluebook Marks Extraction System")

# Initialize Backend Singletons
template_manager = TemplateManager(CONFIG_PATH)
if CONFIG_PATH.exists():
    current_template = template_manager.load_from_json(CONFIG_PATH)
else:
    current_template = template_manager.create_default_template()

digit_recognizer = DigitRecognizer(model_path=MODEL_PATH)
validator = MarksValidator(confidence_threshold=0.65)
pipeline = ProcessingPipeline(recognizer=digit_recognizer, validator=validator)
review_manager = ReviewManager(validator=validator)
exporter = ExcelExporter()
eval_sheet_extractor = EvalSheetExtractor(recognizer=digit_recognizer)
last_eval_sheet_cache: Optional[Dict[str, Any]] = None


def encode_image_base64(img_bgr: Optional[np.ndarray]) -> str:
    """Encode an OpenCV BGR image as a base64 PNG data URL."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def serialize_record(record: StudentAssessmentRecord) -> Dict[str, Any]:
    """Serialize a StudentAssessmentRecord to a JSON-compatible dict."""
    marks_data = {}
    for q_id, mark in record.marks.items():
        marks_data[q_id] = {
            "question_id": mark.question_id,
            "question_label": mark.question_label,
            "max_marks": mark.max_marks,
            "step": mark.step,
            "extracted_str": mark.extracted_str or "",
            "extracted_val": mark.extracted_val,
            "verified_val": mark.verified_val,
            "confidence": mark.confidence,
            "status": mark.status,
            "validation_message": mark.validation_message,
            "crop_base64": encode_image_base64(mark.crop_bgr),
        }

    return {
        "record_id": record.record_id,
        "source_filename": record.source_filename,
        "page_number": record.page_number,
        "student_id": record.student_id,
        "student_name": record.student_name,
        "subject": record.subject,
        "assessment": record.assessment,
        "total_marks": record.total_marks,
        "is_fully_verified": record.is_fully_verified,
        "processing_timestamp": record.processing_timestamp,
        "log_message": record.log_message,
        "needs_review_count": record.needs_review_count,
        "invalid_count": record.invalid_count,
        "model_version": record.model_version,
        "marks": marks_data,
    }


# Mount Static Files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

if (BASE_DIR / "models").exists():
    app.mount("/models", StaticFiles(directory=str(BASE_DIR / "models")), name="models")


@app.get("/health")
async def health_check():
    """Health check endpoint for system monitoring."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the minimal single-page web app."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web application interface not found.")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_system_status():
    """Return local model and CPU inference execution provider status."""
    return {
        "model_name": digit_recognizer.MODEL_NAME,
        "model_version": digit_recognizer.MODEL_VERSION,
        "model_available": digit_recognizer.model_available,
        "model_path": str(digit_recognizer.model_path),
        "execution_provider": "CPUExecutionProvider",
        "load_error": digit_recognizer.load_error_message,
        "total_configured_questions": len(current_template.questions),
        "total_max_marks": current_template.total_max_marks,
    }


@app.get("/api/template")
async def get_template():
    """Return active template specification."""
    return template_manager.to_dict(current_template)


@app.post("/api/template")
async def update_template(payload: Dict[str, Any]):
    """Update active template configuration."""
    global current_template
    try:
        new_template = template_manager.from_dict(payload)
        template_manager.save_to_json(new_template, CONFIG_PATH)
        current_template = new_template
        return {"status": "success", "message": "Template updated successfully."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to update template: {exc}")


def format_duplicates_for_api(raw_duplicates) -> List[Dict[str, Any]]:
    """Format review manager duplicate tuples into structured API dicts."""
    res = []
    for s_id, ass, sub, rec_ids in raw_duplicates:
        filenames = []
        for rid in rec_ids:
            rec = review_manager.get_record(rid)
            if rec:
                filenames.append(rec.source_filename)
        res.append({
            "student_id": s_id,
            "assessment": ass,
            "subject": sub,
            "record_ids": rec_ids,
            "filenames": filenames,
        })
    return res


@app.post("/api/process")
async def process_bluebooks(
    files: List[UploadFile] = File(...),
    assessment: str = Form("IA Test - 1"),
    subject: str = Form("Operating Systems (21CS42)"),
    target_sheet: int = Form(2),
    auto_align: bool = Form(True),
):
    """Process uploaded bluebook scans and extract handwritten marks."""
    review_manager.clear()
    loader = FileLoader(pdf_processor=PDFProcessor(default_dpi=150))
    staged_docs: List[LoadedDocument] = []

    for f in files:
        contents = await f.read()
        docs = loader.load_from_bytes(f.filename, contents, target_page=target_sheet)
        staged_docs.extend(docs)

    pipeline.process_batch(
        docs=staged_docs,
        template=current_template,
        review_manager=review_manager,
        assessment=assessment,
        subject=subject,
        auto_align=auto_align,
    )

    records_data = [serialize_record(r) for r in review_manager.records]
    stats = review_manager.get_summary_statistics()
    duplicates = format_duplicates_for_api(review_manager.find_duplicates())

    return {
        "records": records_data,
        "summary": stats,
        "duplicates": duplicates,
    }


@app.post("/api/load-samples")
async def load_sample_dataset(
    assessment: str = Form("IA Test - 1"),
    subject: str = Form("Operating Systems (21CS42)"),
    auto_align: bool = Form(True),
):
    """Load pre-generated test bluebooks from sample_data/ for one-click testing."""
    review_manager.clear()
    loader = FileLoader(pdf_processor=PDFProcessor(default_dpi=150))
    staged_docs: List[LoadedDocument] = []

    sample_files = [
        SAMPLE_DIR / "student_sample_1.png",
        SAMPLE_DIR / "student_sample_2_skewed.jpg",
        SAMPLE_DIR / "student_sample_4_multipage.pdf",
        SAMPLE_DIR / "student_sample_5_review_needed.png",
    ]

    for p in sample_files:
        if p.exists():
            docs = loader.load_from_path(p, target_page=2)
            staged_docs.extend(docs)

    pipeline.process_batch(
        docs=staged_docs,
        template=current_template,
        review_manager=review_manager,
        assessment=assessment,
        subject=subject,
        auto_align=auto_align,
    )

    records_data = [serialize_record(r) for r in review_manager.records]
    stats = review_manager.get_summary_statistics()
    duplicates = format_duplicates_for_api(review_manager.find_duplicates())

    return {
        "records": records_data,
        "summary": stats,
        "duplicates": duplicates,
    }


class MarkUpdateRequest(BaseModel):
    record_id: str
    question_id: str
    value: str


@app.post("/api/update-mark")
async def update_mark(req: MarkUpdateRequest):
    """Update a specific question mark manually."""
    # Support case-insensitive question id lookup
    rec = review_manager.get_record(req.record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found.")

    target_qid = req.question_id
    if target_qid not in rec.marks:
        for qk in rec.marks.keys():
            if qk.lower() == req.question_id.lower():
                target_qid = qk
                break

    success = review_manager.update_mark(req.record_id, target_qid, req.value, user_note="Teacher review")
    if not success:
        raise HTTPException(status_code=404, detail="Question not found in record.")

    updated_rec = review_manager.get_record(req.record_id)
    return {
        "record": serialize_record(updated_rec),
        "summary": review_manager.get_summary_statistics(),
    }


class StudentUpdateRequest(BaseModel):
    record_id: str
    student_id: str
    student_name: str


@app.post("/api/update-student")
async def update_student(req: StudentUpdateRequest):
    """Update student ID and name for a record."""
    success = review_manager.update_student_info(
        req.record_id,
        student_id=req.student_id,
        student_name=req.student_name,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Record not found.")

    rec = review_manager.get_record(req.record_id)
    return {
        "record": serialize_record(rec),
        "summary": review_manager.get_summary_statistics(),
    }


@app.get("/api/export")
async def export_excel():
    """Stream formatted Excel workbook for download."""
    if not review_manager.records:
        raise HTTPException(status_code=400, detail="No records available to export. Process documents first.")

    xlsx_bytes = exporter.export_to_bytes(review_manager.records, template=current_template)
    filename = "IA_Bluebook_Marks_Ledger.xlsx"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return Response(content=xlsx_bytes, media_type=headers["Content-Type"], headers=headers)


@app.post("/api/extract-sheet")
async def extract_sheet(
    file: Optional[UploadFile] = File(None),
    target_ia: str = Form("IA 1"),
    use_demo: bool = Form(False),
):
    """Extract marks from the 3-column Marks Evaluation Sheet with column cropping.
    
    Supports single images and multi-page PDFs containing multiple evaluation sheets.
    """
    global last_eval_sheet_cache
    demo_path = SAMPLE_DIR / "marks_eval_sheet_filled.jpg"
    pages_to_process: List[Tuple[int, np.ndarray]] = []

    if use_demo or file is None:
        if not demo_path.exists():
            raise HTTPException(status_code=404, detail="Demo evaluation sheet not found on server.")
        img_bgr = cv2.imread(str(demo_path))
        if img_bgr is None or img_bgr.size == 0:
            raise HTTPException(status_code=500, detail="Failed to load demo evaluation sheet image.")
        pages_to_process.append((1, img_bgr))
    else:
        contents = await file.read()
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=contents, filetype="pdf")
            if len(doc) == 0:
                raise HTTPException(status_code=400, detail="Empty PDF file uploaded.")
            for page_idx, page in enumerate(doc, start=1):
                img = None
                imgs = page.get_images()
                if imgs:
                    try:
                        base_img = doc.extract_image(imgs[0][0])
                        img = cv2.imdecode(np.frombuffer(base_img["image"], np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        img = None
                if img is None or img.size == 0:
                    pix = page.get_pixmap(dpi=150)
                    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)

                if img is not None and img.size > 0:
                    pages_to_process.append((page_idx, img))
            doc.close()
            if not pages_to_process:
                raise HTTPException(status_code=400, detail="Could not render any pages from uploaded PDF.")
        else:
            arr = np.frombuffer(contents, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None or img_bgr.size == 0:
                raise HTTPException(status_code=400, detail="Could not decode uploaded evaluation sheet image.")
            pages_to_process.append((1, img_bgr))

    try:
        res = eval_sheet_extractor.extract_multipage(pages_to_process, target_ia=target_ia)
        last_eval_sheet_cache = res
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


class SheetExportRequest(BaseModel):
    format: str = "xlsx"  # "xlsx" or "csv"
    data: Dict[str, Any]


@app.post("/api/export-sheet")
async def export_sheet(req: SheetExportRequest):
    """Export verified marks as .xlsx or .csv."""
    export_fmt = req.format.strip().lower()
    data = req.data

    if not data or ("results_by_ia" not in data and "sheets" not in data):
        if last_eval_sheet_cache:
            data = last_eval_sheet_cache
        else:
            raise HTTPException(status_code=400, detail="No evaluation sheet data available to export.")

    if export_fmt in ("csv", "text/csv"):
        csv_str = eval_sheet_extractor.generate_csv_string(data)
        # Prepend UTF-8 BOM so Microsoft Excel and Apple Numbers open with correct columns & UTF-8 encoding
        csv_bytes = b"\xef\xbb\xbf" + csv_str.encode("utf-8")
        filename = "IA_Marks_Evaluation.csv"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/csv; charset=utf-8",
        }
        return Response(content=csv_bytes, media_type="text/csv; charset=utf-8", headers=headers)
    else:
        xlsx_bytes = eval_sheet_extractor.generate_excel_bytes(data)
        filename = "IA_Marks_Evaluation.xlsx"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return Response(content=xlsx_bytes, media_type=headers["Content-Type"], headers=headers)


@app.get("/{file_name:path}")
async def serve_static_root(file_name: str):
    """Serve relative static files directly from web/."""
    from fastapi.responses import FileResponse
    target = WEB_DIR / file_name
    if target.exists() and target.is_file():
        return FileResponse(target)
    raise HTTPException(status_code=404, detail="File not found")
