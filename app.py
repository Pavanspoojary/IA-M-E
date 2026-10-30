"""Streamlit Web User Interface for IA Bluebook Marks Extraction System.

A professional, local computer vision and ONNX Runtime CPU application
for extracting, verifying, and exporting college examination marks
with deterministic audit gates and multi-sheet Excel generation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import io
import json
import time

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.file_loader import FileLoader, LoadedDocument
from src.pdf_processor import PDFProcessor
from src.image_preprocessor import ImagePreprocessor, PreprocessingResult
from src.template_manager import TemplateManager, BluebookTemplate, QuestionRegion, StudentIdRegion
from src.region_extractor import RegionExtractor, ExtractedRegion
from src.digit_recognizer import DigitRecognizer, DigitRecognitionResult
from src.marks_validator import MarksValidator, VerificationStatus, ValidatedMark
from src.review_manager import ReviewManager, StudentAssessmentRecord
from src.excel_exporter import ExcelExporter
from src.processing_pipeline import ProcessingPipeline


# App setup and page config
st.set_page_config(
    page_title="IA Bluebook Marks Extraction System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS (Zero Emojis Policy, Crisp Typography, Badges, and Cards)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-validated {
        background-color: #E2EFDA;
        color: #276A3C;
        border: 1px solid #C5E0B4;
    }
    .badge-needs-review {
        background-color: #FFF2CC;
        color: #8C6B00;
        border: 1px solid #FFE699;
    }
    .badge-corrected {
        background-color: #DDEBF7;
        color: #1B4F72;
        border: 1px solid #BDD7EE;
    }
    .badge-blank {
        background-color: #F2F2F2;
        color: #595959;
        border: 1px solid #D9D9D9;
    }
    .badge-invalid {
        background-color: #FCE4D6;
        color: #C00000;
        border: 1px solid #F8CBAD;
    }
    .card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths and Session State
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "template.json"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"

if "template_manager" not in st.session_state:
    st.session_state.template_manager = TemplateManager(DEFAULT_CONFIG_PATH)

if "current_template" not in st.session_state:
    if DEFAULT_CONFIG_PATH.exists():
        st.session_state.current_template = st.session_state.template_manager.load_from_json(DEFAULT_CONFIG_PATH)
    else:
        st.session_state.current_template = st.session_state.template_manager.create_default_template()

if "review_manager" not in st.session_state:
    st.session_state.review_manager = ReviewManager()

if "digit_recognizer" not in st.session_state:
    st.session_state.digit_recognizer = DigitRecognizer(model_path=DEFAULT_MODEL_PATH)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = ProcessingPipeline(recognizer=st.session_state.digit_recognizer)

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "batch_run_done" not in st.session_state:
    st.session_state.batch_run_done = False

if "selected_review_index" not in st.session_state:
    st.session_state.selected_review_index = 0


# Utility functions
def get_status_badge_html(status: str) -> str:
    """Return semantic HTML badge for verification status."""
    class_map = {
        VerificationStatus.VALIDATED: "badge-validated",
        VerificationStatus.VERIFIED_AUTOMATICALLY: "badge-validated",
        VerificationStatus.VERIFIED_MANUALLY: "badge-corrected",
        VerificationStatus.CORRECTED_MANUALLY: "badge-corrected",
        VerificationStatus.NEEDS_REVIEW: "badge-needs-review",
        VerificationStatus.BLANK: "badge-blank",
        VerificationStatus.INVALID: "badge-invalid",
        VerificationStatus.UNREADABLE: "badge-invalid",
    }
    badge_class = class_map.get(status, "badge-blank")
    return f'<span class="badge {badge_class}">{status}</span>'


def bgr_to_pil(bgr_array: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR or single channel image to PIL for display."""
    if bgr_array is None or bgr_array.size == 0:
        return Image.new("RGB", (100, 100), color=(240, 240, 240))
    if len(bgr_array.shape) == 2:
        return Image.fromarray(bgr_array)
    rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# Sidebar Navigation with 8 Exact Sections
st.sidebar.title("Navigation")
sections = [
    "1. Dashboard",
    "2. Upload Bluebooks",
    "3. Template Configuration",
    "4. Image Preprocessing & Crop Preview",
    "5. Batch Recognition",
    "6. Marks Verification",
    "7. Excel Export",
    "8. Settings & Local Model Status",
]
nav_choice = st.sidebar.radio("Select Workflow Step:", sections, index=0)

st.sidebar.markdown("---")
recognizer_inst = st.session_state.digit_recognizer
if recognizer_inst.model_available:
    st.sidebar.caption("ONNX Engine: Active (CPU)")
else:
    st.sidebar.caption("ONNX Engine: Offline (Manual Mode)")

st.sidebar.caption("IA Bluebook Marks Extraction System")
st.sidebar.caption("Local Computer Vision [Zero External APIs]")


# SECTION 1: DASHBOARD
if nav_choice == "1. Dashboard":
    st.markdown('<div class="main-title">Continuous Internal Assessment Marks Extraction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "An enterprise-grade, deterministic computer vision system for digitizing handwritten question-wise marks "
        "from examination bluebooks using local OpenCV image processing and ONNX Runtime CPU inference."
        "</div>",
        unsafe_allow_html=True,
    )

    # Top Status Banner
    rec = st.session_state.digit_recognizer
    if rec.model_available:
        st.success(f"Local ONNX Model Active: {rec.MODEL_NAME} (v{rec.MODEL_VERSION}) running on CPUExecutionProvider.")
    else:
        st.warning(f"ONNX Model Unavailable: {rec.load_error_message}. Application is operating in Manual-Only Verification Mode.")

    # High-Level Metrics
    st.markdown("### Operational Metrics")
    c1, c2, c3, c4 = st.columns(4)
    rm: ReviewManager = st.session_state.review_manager
    t = st.session_state.current_template
    docs = st.session_state.uploaded_documents

    with c1:
        st.markdown(f'<div class="card"><div class="metric-value">{len(docs)}</div><div class="metric-label">Staged Documents</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="metric-value">{len(rm.records)}</div><div class="metric-label">Processed Records</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="metric-value">{len(t.questions)}</div><div class="metric-label">Configured Questions</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="metric-value">{t.total_max_marks}</div><div class="metric-label">Total Max Marks</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Architecture & Operational Guardrails")
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown(
            """
            **Core Engine Features:**
            - **100% Local Execution**: All image preprocessing, tensor inference, and spreadsheet formatting execute on the local CPU without external network dependencies.
            - **PyMuPDF Engine**: Seamlessly extracts and renders high-resolution marks sheets from multi-page scanned examination PDF booklets.
            - **10-Stage CV Pipeline**: Corrects document boundary skew (+/- 15 deg), photograph perspective distortion, and paper noise.
            - **Isolated Cell Segmentation**: Morphologically strips printed table grid lines to isolate handwritten strokes without clipping.
            """
        )
    with col_arch2:
        st.markdown(
            """
            **Auditing & Validation Gates:**
            - **Deterministic Increment Verification**: Enforces step rules (e.g. 0.5 or 1.0 increments) and flags values exceeding question maximums.
            - **Ambiguity Gating**: Classifications with low probability or narrow candidate margins are held as `Needs review` rather than guessed.
            - **Duplicate Detection**: Flags duplicate `(Student ID, Assessment, Subject)` records automatically.
            - **Non-Destructive Blank Handling**: Unattempted questions remain blank and are never converted to zero automatically.
            """
        )


# SECTION 2: UPLOAD BLUEBOOKS
elif nav_choice == "2. Upload Bluebooks":
    st.markdown('<div class="main-title">Upload Examination Bluebooks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Stage examination booklets in JPG, JPEG, PNG, or multi-page PDF format for batch processing."
        "</div>",
        unsafe_allow_html=True,
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        assessment_name = st.selectbox(
            "Assessment Identifier",
            ["IA Test - 1", "IA Test - 2", "IA Test - 3", "Semester End Examination", "Makeup Assessment"],
            index=0,
        )
    with col_meta2:
        subject_name = st.text_input("Course / Subject Name", value="Operating Systems (21CS42)")

    st.markdown("#### Select Files")
    uploaded_files = st.file_uploader(
        "Upload Bluebook Scans (Images or Scanned PDFs)",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        help="Upload single or multiple image files or scanned PDFs.",
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        target_sheet_num = st.number_input(
            "Target Bluebook Sheet Page Number (for PDFs)",
            min_value=1,
            max_value=20,
            value=2,
            help="Colleges print question-wise marks on the second sheet of the booklet.",
        )
    with col_opt2:
        dpi_setting = st.selectbox("PDF Render Resolution (DPI)", [150, 200, 300], index=0)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Prepare Batch", type="primary", disabled=(not uploaded_files)):
            loader = FileLoader(pdf_processor=PDFProcessor(default_dpi=dpi_setting))
            loaded_list: List[LoadedDocument] = []

            prog_bar = st.progress(0.0)
            status_text = st.empty()

            for i, uf in enumerate(uploaded_files):
                status_text.text(f"Staging {uf.name}...")
                bytes_data = uf.read()
                docs = loader.load_from_bytes(uf.name, bytes_data, target_page=target_sheet_num)
                loaded_list.extend(docs)
                prog_bar.progress((i + 1) / len(uploaded_files))

            prog_bar.empty()
            status_text.empty()

            st.session_state.uploaded_documents = loaded_list
            st.session_state.batch_assessment = assessment_name
            st.session_state.batch_subject = subject_name
            st.success(f"Successfully staged {len(loaded_list)} document page(s).")
            st.rerun()

    with col_btn2:
        if st.session_state.uploaded_documents and st.button("Clear Staged Documents"):
            st.session_state.uploaded_documents = []
            st.session_state.batch_run_done = False
            st.rerun()

    if st.session_state.uploaded_documents:
        st.markdown("#### Staged Document Inventory")
        table_rows = []
        for i, d in enumerate(st.session_state.uploaded_documents, start=1):
            dim = f"{d.image_bgr.shape[1]}x{d.image_bgr.shape[0]}" if d.image_bgr is not None else "Failed"
            table_rows.append({
                "Index": i,
                "Filename": d.filename,
                "Type": d.file_type.upper(),
                "Target Page": d.page_number,
                "Dimensions": dim,
                "Status": d.status.upper(),
            })
        st.dataframe(table_rows, use_container_width=True)


# SECTION 3: TEMPLATE CONFIGURATION
elif nav_choice == "3. Template Configuration":
    st.markdown('<div class="main-title">Template Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Define and calibrate fixed rectangular coordinates for question marks and student ID regions "
        "on the reference bluebook layout."
        "</div>",
        unsafe_allow_html=True,
    )

    t: BluebookTemplate = st.session_state.current_template
    tm: TemplateManager = st.session_state.template_manager

    tab1, tab2, tab3 = st.tabs(["Layout Calibration & Preview", "Questions List", "Save & Load Template"])

    with tab1:
        st.subheader("Reference Page & Visual Bounding Box Calibration")
        template_img_file = st.file_uploader(
            "Upload Blank Reference Bluebook Sheet (Optional, or uses default)",
            type=["png", "jpg", "jpeg"],
            key="template_uploader",
        )

        ref_image = None
        if template_img_file:
            arr = np.frombuffer(template_img_file.read(), dtype=np.uint8)
            ref_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            sample_ref_path = BASE_DIR / "sample_data" / "clean_bluebook_template.png"
            if sample_ref_path.exists():
                ref_image = cv2.imread(str(sample_ref_path))

        if ref_image is not None:
            preview_canvas = cv2.resize(ref_image.copy(), (t.reference_width, t.reference_height))

            # Draw Student ID box
            if t.student_id_region:
                sx, sy, sw, sh = t.student_id_region.x, t.student_id_region.y, t.student_id_region.w, t.student_id_region.h
                cv2.rectangle(preview_canvas, (sx, sy), (sx + sw, sy + sh), (200, 50, 50), 2)
                cv2.putText(preview_canvas, "Student ID", (sx, max(15, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 50), 2)

            # Draw Question boxes
            for q in t.questions:
                cv2.rectangle(preview_canvas, (q.x, q.y), (q.x + q.w, q.y + q.h), (30, 140, 30), 2)
                cv2.putText(preview_canvas, f"{q.label} (Max:{q.max_marks})", (q.x, max(15, q.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 140, 30), 2)

            st.image(bgr_to_pil(preview_canvas), caption="Configured Regions on Reference Sheet", use_container_width=True)

    with tab2:
        st.subheader("Configured Question Marks Fields")
        st.caption("Adjust pixel coordinates (Reference: 1240 x 1754). Normalized coordinates update automatically.")

        for idx, q in enumerate(t.questions):
            with st.expander(f"Question {idx+1}: {q.label} (Max: {q.max_marks})", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    new_label = st.text_input("Label", value=q.label, key=f"q_lbl_{idx}")
                    new_max = st.number_input("Max Marks", min_value=1.0, max_value=100.0, value=float(q.max_marks), step=0.5, key=f"q_max_{idx}")
                with c2:
                    new_step = st.selectbox("Increment Step", [0.5, 1.0, 0.25], index=0 if q.step==0.5 else 1, key=f"q_step_{idx}")
                    new_x = st.number_input("X Coordinate", min_value=0, max_value=t.reference_width, value=q.x, step=5, key=f"q_x_{idx}")
                with c3:
                    new_y = st.number_input("Y Coordinate", min_value=0, max_value=t.reference_height, value=q.y, step=5, key=f"q_y_{idx}")
                    new_w = st.number_input("Width", min_value=10, max_value=400, value=q.w, step=5, key=f"q_w_{idx}")
                with c4:
                    new_h = st.number_input("Height", min_value=10, max_value=200, value=q.h, step=5, key=f"q_h_{idx}")

                q.label = new_label
                q.max_marks = new_max
                q.step = new_step
                q.x = new_x
                q.y = new_y
                q.w = new_w
                q.h = new_h
                q.update_normalized(t.reference_width, t.reference_height)

        if st.button("Add New Question Field"):
            next_idx = len(t.questions) + 1
            new_q = QuestionRegion(
                id=f"q{next_idx}",
                label=f"Q{next_idx}",
                max_marks=10.0,
                step=0.5,
                x=420,
                y=500 + (next_idx * 60),
                w=130,
                h=65,
            )
            new_q.update_normalized(t.reference_width, t.reference_height)
            t.questions.append(new_q)
            st.rerun()

    with tab3:
        st.subheader("Save / Load Template Specification")
        c_save1, c_save2 = st.columns(2)

        with c_save1:
            st.markdown("#### Save Current Template")
            save_name = st.text_input("Template Name", value=t.template_name)
            t.template_name = save_name

            template_json_str = json.dumps(tm.to_dict(t), indent=4)
            st.download_button(
                "Download Template JSON",
                data=template_json_str,
                file_name="bluebook_template.json",
                mime="application/json",
            )
            if st.button("Save as Default Config (template.json)"):
                tm.save_to_json(t, DEFAULT_CONFIG_PATH)
                st.success("Saved successfully as default configuration.")

        with c_save2:
            st.markdown("#### Load Template from JSON")
            uploaded_json = st.file_uploader("Upload JSON Template", type=["json"])
            if uploaded_json:
                data = json.load(uploaded_json)
                st.session_state.current_template = tm.from_dict(data)
                st.success(f"Loaded template: {st.session_state.current_template.template_name}")
                st.rerun()


# SECTION 4: IMAGE PREPROCESSING & CROP PREVIEW
elif nav_choice == "4. Image Preprocessing & Crop Preview":
    st.markdown('<div class="main-title">Image Preprocessing & Crop Preview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Inspect the 10-stage computer vision preprocessing pipeline, perspective alignment, "
        "and isolated cell crops for any uploaded bluebook."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_docs: List[LoadedDocument] = st.session_state.uploaded_documents
    if not uploaded_docs:
        st.warning("No documents uploaded. Please upload bluebooks in 'Upload Bluebooks' first.")
    else:
        doc_names = [f"{d.filename} (Page {d.page_number})" for d in uploaded_docs]
        selected_doc_idx = st.selectbox(
            "Select Document to Preview:",
            range(len(doc_names)),
            format_func=lambda i: doc_names[i],
        )
        target_doc = uploaded_docs[selected_doc_idx]

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            denoise_sel = st.selectbox("Denoise Filter", ["median", "gaussian"], index=0)
            thresh_sel = st.selectbox("Binarization Mode", ["otsu", "adaptive"], index=0)
        with c_opt2:
            auto_align_sel = st.checkbox("Perspective Warp & Page Boundary Detection", value=True)
            deskew_sel = st.checkbox("Rotational Deskewing (+/- 15 deg)", value=True)

        if target_doc.image_bgr is not None:
            preprocessor = ImagePreprocessor()
            t = st.session_state.current_template
            prep_res = preprocessor.preprocess(
                target_doc.image_bgr,
                target_width=t.reference_width,
                target_height=t.reference_height,
                auto_align=auto_align_sel,
                denoise_method=denoise_sel,
                threshold_method=thresh_sel,
            )

            col_view1, col_view2 = st.columns(2)
            with col_view1:
                st.caption(f"Original Input ({target_doc.image_bgr.shape[1]}x{target_doc.image_bgr.shape[0]})")
                st.image(bgr_to_pil(target_doc.image_bgr), use_container_width=True)
            with col_view2:
                st.caption(f"Aligned & Deskewed ({prep_res.aligned_bgr.shape[1]}x{prep_res.aligned_bgr.shape[0]} | Skew: {prep_res.skew_angle} deg | Status: {prep_res.alignment_status})")
                st.image(bgr_to_pil(prep_res.thresholded), use_container_width=True)

            st.markdown("---")
            st.subheader("Question Cell Crops (With Grid Lines Removed)")
            extractor = RegionExtractor(inner_margin_ratio=0.0)
            extracted_regions = extractor.extract_all(prep_res.aligned_bgr, prep_res.thresholded_inv, t)

            crop_cols = st.columns(min(8, len(extracted_regions)))
            for c_i, (q_id, reg) in enumerate(extracted_regions.items()):
                if not reg.is_student_id:
                    with crop_cols[c_i % len(crop_cols)]:
                        st.caption(reg.label)
                        st.image(bgr_to_pil(reg.crop_bgr), use_container_width=True)


# SECTION 5: BATCH RECOGNITION
elif nav_choice == "5. Batch Recognition":
    st.markdown('<div class="main-title">Batch Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Execute automated question marks extraction across all staged bluebook documents using local ONNX Runtime inference."
        "</div>",
        unsafe_allow_html=True,
    )

    docs: List[LoadedDocument] = st.session_state.uploaded_documents
    if not docs:
        st.warning("No documents staged. Please upload files in 'Upload Bluebooks' first.")
    else:
        st.write(f"Staged Documents: **{len(docs)} sheet(s)**")

        rec = st.session_state.digit_recognizer
        if rec.model_available:
            st.info(f"Model Active: {rec.MODEL_NAME} (v{rec.MODEL_VERSION}) on CPUExecutionProvider.")
        else:
            st.warning(f"Model Offline: {rec.load_error_message}. Processing will stage sheets for manual entry.")

        col_run1, col_run2 = st.columns([2, 1])
        with col_run1:
            denoise_mode = st.selectbox("Denoise Mode", ["median", "gaussian"], index=0)
            thresh_mode = st.selectbox("Threshold Mode", ["otsu", "adaptive"], index=0)
        with col_run2:
            auto_align = st.checkbox("Auto Perspective Alignment", value=True)
            step_batch = st.session_state.get("batch_assessment", "IA-1")
            sub_batch = st.session_state.get("batch_subject", "Subject")

        if st.button("Execute Batch Recognition", type="primary"):
            pipeline: ProcessingPipeline = st.session_state.pipeline
            rm: ReviewManager = st.session_state.review_manager
            rm.clear()
            template: BluebookTemplate = st.session_state.current_template

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_callback(cur, tot):
                progress_bar.progress(cur / tot)
                status_text.text(f"Processing sheet {cur} of {tot}...")

            results = pipeline.process_batch(
                docs=docs,
                template=template,
                review_manager=rm,
                assessment=step_batch,
                subject=sub_batch,
                auto_align=auto_align,
                denoise_method=denoise_mode,
                threshold_method=thresh_mode,
                progress_callback=progress_callback,
            )

            progress_bar.empty()
            status_text.empty()
            st.session_state.batch_run_done = True
            st.success(f"Batch recognition completed for {len(results)} sheet(s). Proceed to 'Marks Verification'.")

        if st.session_state.batch_run_done and rm.records:
            st.markdown("---")
            st.subheader("Batch Recognition Summary")
            stats = rm.get_summary_statistics()
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                st.metric("Total Records", stats["total_records"])
            with cm2:
                st.metric("Validated Fields", stats["validated"])
            with cm3:
                st.metric("Needs Review", stats["needs_review"])
            with cm4:
                st.metric("Fully Verified Sheets", stats["fully_verified_records"])


# SECTION 6: MARKS VERIFICATION
elif nav_choice == "6. Marks Verification":
    st.markdown('<div class="main-title">Marks Verification & Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Inspect each candidate mark alongside original handwriting crops. Correct uncertain, invalid, or low-margin marks."
        "</div>",
        unsafe_allow_html=True,
    )

    rm: ReviewManager = st.session_state.review_manager
    if not rm.records:
        st.warning("No records to verify. Run 'Batch Recognition' first.")
    else:
        # Duplicate Alert
        dups = rm.find_duplicates()
        if dups:
            for s_id, ass, sub, r_ids in dups:
                st.warning(f"Duplicate Record Warning: Student ID '{s_id}' has {len(r_ids)} sheets for '{ass}' in '{sub}'.")

        # Stats Cards
        stats = rm.get_summary_statistics()
        c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
        with c_m1:
            st.markdown(f'<div class="card"><div class="metric-value">{stats["total_records"]}</div><div class="metric-label">Total Sheets</div></div>', unsafe_allow_html=True)
        with c_m2:
            st.markdown(f'<div class="card"><div class="metric-value">{stats["validated"]}</div><div class="metric-label">Validated Marks</div></div>', unsafe_allow_html=True)
        with c_m3:
            st.markdown(f'<div class="card"><div class="metric-value">{stats["needs_review"]}</div><div class="metric-label">Needs Review</div></div>', unsafe_allow_html=True)
        with c_m4:
            st.markdown(f'<div class="card"><div class="metric-value">{stats["corrected"]}</div><div class="metric-label">Corrected Manually</div></div>', unsafe_allow_html=True)
        with c_m5:
            st.markdown(f'<div class="card"><div class="metric-value">{stats["invalid"] + stats["blank"]}</div><div class="metric-label">Invalid / Blank</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Record Selector and Navigation
        c_nav1, c_nav2, c_nav3 = st.columns([1, 4, 1])
        cur_idx = st.session_state.selected_review_index
        cur_idx = max(0, min(cur_idx, len(rm.records) - 1))

        with c_nav1:
            if st.button("< Previous Record", disabled=(cur_idx == 0)):
                st.session_state.selected_review_index = cur_idx - 1
                st.rerun()

        with c_nav2:
            record_labels = [
                f"[{i+1}/{len(rm.records)}] {r.student_id} - {r.source_filename} (Total: {r.total_marks or 0.0} | {'VERIFIED' if r.is_fully_verified else 'REQUIRES REVIEW'})"
                for i, r in enumerate(rm.records)
            ]
            sel = st.selectbox(
                "Active Student Record:",
                range(len(rm.records)),
                index=cur_idx,
                format_func=lambda i: record_labels[i],
            )
            if sel != cur_idx:
                st.session_state.selected_review_index = sel
                st.rerun()

        with c_nav3:
            if st.button("Next Record >", disabled=(cur_idx >= len(rm.records) - 1)):
                st.session_state.selected_review_index = cur_idx + 1
                st.rerun()

        curr_rec = rm.records[st.session_state.selected_review_index]

        # Metadata Row
        c_id1, c_id2, c_id3 = st.columns(3)
        with c_id1:
            new_sid = st.text_input("Student USN / Register Number", value=curr_rec.student_id, key=f"rec_sid_{curr_rec.record_id}")
            if new_sid != curr_rec.student_id:
                rm.update_student_info(curr_rec.record_id, student_id=new_sid)
                st.rerun()
        with c_id2:
            new_sname = st.text_input("Student Name (Optional)", value=curr_rec.student_name, key=f"rec_sname_{curr_rec.record_id}")
            if new_sname != curr_rec.student_name:
                rm.update_student_info(curr_rec.record_id, student_id=curr_rec.student_id, student_name=new_sname)
        with c_id3:
            st.write("**Total Marks (Verified):**")
            st.markdown(f"### {curr_rec.total_marks or 0.0}")

        st.markdown("#### Question-wise Marks Review")
        col_hdr1, col_hdr2, col_hdr3, col_hdr4, col_hdr5, col_hdr6 = st.columns([1.5, 2.5, 1.0, 1.5, 1.8, 2.5])
        col_hdr1.write("**Question**")
        col_hdr2.write("**Handwriting Crop**")
        col_hdr3.write("**Max**")
        col_hdr4.write("**Detected**")
        col_hdr5.write("**Status**")
        col_hdr6.write("**Verified Mark (Editable)**")

        for q_id, mark in curr_rec.marks.items():
            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2.5, 1.0, 1.5, 1.8, 2.5])
            with col1:
                st.write(f"**{mark.question_label}**")
            with col2:
                if mark.crop_bgr is not None and mark.crop_bgr.size > 0:
                    st.image(bgr_to_pil(mark.crop_bgr), width=130)
                else:
                    st.caption("[No crop]")
            with col3:
                st.write(f"{mark.max_marks}")
            with col4:
                disp_val = mark.extracted_str if mark.extracted_str else "-"
                st.write(f"`{disp_val}`")
            with col5:
                st.markdown(get_status_badge_html(mark.status), unsafe_allow_html=True)
            with col6:
                cur_val_str = str(mark.verified_val) if mark.verified_val is not None else ""
                new_input = st.text_input(
                    f"Mark for {mark.question_label}",
                    value=cur_val_str,
                    key=f"edit_{curr_rec.record_id}_{q_id}",
                    label_visibility="collapsed",
                )
                if new_input != cur_val_str:
                    rm.update_mark(curr_rec.record_id, q_id, new_input, user_note="Manual teacher edit")
                    st.rerun()


# SECTION 7: EXCEL EXPORT
elif nav_choice == "7. Excel Export":
    st.markdown('<div class="main-title">Excel Export & Ledger Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Review final verified marks ledger and export formatted multi-worksheet Excel workbook."
        "</div>",
        unsafe_allow_html=True,
    )

    rm: ReviewManager = st.session_state.review_manager
    template: BluebookTemplate = st.session_state.current_template

    if not rm.records:
        st.warning("No records available to export. Run recognition and verify marks first.")
    else:
        stats = rm.get_summary_statistics()
        has_unresolved = (stats["needs_review"] > 0) or (stats["invalid"] > 0)
        has_duplicates = stats["duplicates"] > 0

        if has_unresolved:
            st.error(
                f"Export Gate Warning: {stats['needs_review']} field(s) require review and {stats['invalid']} field(s) are invalid. "
                "Resolve all fields in 'Marks Verification' before finalizing."
            )
        if has_duplicates:
            st.warning(f"Duplicate Warning: {stats['duplicates']} student record(s) have duplicate entries.")

        st.markdown("### Marks Summary Ledger Preview")
        summary_rows = []
        for sl_no, r in enumerate(rm.records, start=1):
            row_dict = {
                "Sl. No.": sl_no,
                "Student ID": r.student_id,
                "Student Name": r.student_name,
                "Subject": r.subject,
                "Assessment": r.assessment,
            }
            for q in template.questions:
                m = r.marks.get(q.id)
                row_dict[q.label] = m.verified_val if (m and m.verified_val is not None) else ""
            row_dict["Total Marks"] = r.total_marks or 0.0
            row_dict["Status"] = "Fully Verified" if r.is_fully_verified else "Requires Review"
            summary_rows.append(row_dict)

        st.dataframe(summary_rows, use_container_width=True)

        exporter = ExcelExporter()
        xlsx_bytes = exporter.export_to_bytes(rm.records, template=template)

        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            st.download_button(
                label="Download Formatted Excel Workbook (.xlsx)",
                data=xlsx_bytes,
                file_name="IA_Bluebook_Marks_Ledger.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
        with col_dl2:
            st.caption("Worksheets: Marks Summary (with Excel SUM formulas), Detailed Extraction, and Processing Log.")


# SECTION 8: SETTINGS & LOCAL MODEL STATUS
elif nav_choice == "8. Settings & Local Model Status":
    st.markdown('<div class="main-title">Settings & Local Model Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Inspect ONNX Runtime CPU inference environment, test predictions, and view system diagnostic information."
        "</div>",
        unsafe_allow_html=True,
    )

    rec: DigitRecognizer = st.session_state.digit_recognizer

    st.markdown("### Local Model Specification")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.write(f"**Model Name:** `{rec.MODEL_NAME}`")
        st.write(f"**Model Version:** `v{rec.MODEL_VERSION}`")
        st.write(f"**Model Path:** `{rec.model_path}`")
        st.write(f"**File Exists:** `{'YES' if rec.model_path.exists() else 'NO'}`")
        if rec.model_path.exists():
            kb_size = rec.model_path.stat().st_size / 1024.0
            st.write(f"**File Size:** `{kb_size:.1f} KB`")

    with c_s2:
        st.write(f"**Execution Provider:** `['CPUExecutionProvider']`")
        st.write(f"**Input Tensor:** `[batch, 1, 28, 28]` float32")
        st.write(f"**Output Tensor:** `[batch, 10]` float32")
        st.write(f"**Status:** `{'ONLINE (CPU Ready)' if rec.model_available else 'OFFLINE (Manual Mode)'}`")

    st.markdown("---")
    st.markdown("### Interactive Model Inference Test")
    st.caption("Run CPU inference on synthetic digits to measure real-time latency and softmax probabilities.")

    test_col1, test_col2 = st.columns([1, 2])
    with test_col1:
        test_digit_choice = st.selectbox("Select Test Digit:", [str(d) for d in range(10)], index=5)
        run_test_btn = st.button("Run ONNX CPU Inference", type="primary")

    with test_col2:
        if run_test_btn:
            if not rec.model_available:
                st.error("Model is offline. Run 'python scripts/train_digit_model.py' to generate the model.")
            else:
                test_canvas = np.zeros((60, 60), dtype=np.uint8)
                cv2.putText(test_canvas, test_digit_choice, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
                t0 = time.perf_counter()
                res = rec.recognize_region(test_canvas, max_marks=10.0)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                st.success(f"Recognized Digit: **{res.value_str}** | Probability: **{res.confidence:.4f}** | Latency: **{elapsed_ms:.3f} ms**")
                st.text(f"Diagnostic Breakdown: {res.explanation}")
                st.image(bgr_to_pil(test_canvas), caption=f"Test Digit '{test_digit_choice}'", width=120)

    st.markdown("---")
    st.markdown("### Model Training & Maintenance Commands")
    st.code(
        """# To retrain the compact ONNX digit model locally:
python scripts/train_digit_model.py

# To verify model graph structure and benchmark CPU latency:
python scripts/verify_model.py
""",
        language="bash",
    )
