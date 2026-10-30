# IA Bluebook Marks Extraction System (ONNX CPU Engine)

A deterministic, privacy-first computer vision and local machine-learning web application for college examination cells. Automatically extracts handwritten, question-wise marks from the second sheet of Continuous Internal Assessment (IA) examination bluebooks and exports verified records into multi-worksheet Excel (`.xlsx`) workbooks.

---

## 1. System Architecture & Model Design

The system runs 100% locally on the host machine without external network calls, cloud AI services, or remote inference APIs.

### Architecture Overview

```
[Uploaded Documents: JPG / PNG / PDF]
                  |
                  v
    [PyMuPDF Page Rendering]
    - Scanned PDF second sheet extraction (150-300 DPI)
                  |
                  v
    [10-Stage OpenCV Preprocessing]
    - Orientation validation (Portrait standard)
    - 4-point perspective warp & boundary detection
    - Normalized resolution scaling (1240 x 1754)
    - Median / Gaussian denoising
    - Hough line rotational deskewing (+/- 15 deg)
    - Adaptive / Otsu binarization
                  |
                  v
    [Region Extractor & Border Stripping]
    - Normalized coordinate scaling
    - Morphological table grid line removal
                  |
                  v
    [Marks-Field Pipeline & ONNX CPU Inference]
    - Non-zero density blank detection (<0.35% ink -> Blank)
    - Connected component segmentation
    - Vertical projection profile splitting for touching multi-digits
    - Resizing to 20x20 centered inside 28x28 float32 tensor [N, 1, 28, 28]
    - ONNX Runtime local inference (strictly 'CPUExecutionProvider')
    - Softmax probability scoring and candidate margin analysis
                  |
                  v
    [Deterministic Validation & Grading Rules]
    - Maximum marks check [0 <= mark <= max_marks]
    - Allowed increment check (e.g. 0.5 or 1.0 step)
    - Ambiguity gating (low confidence or narrow margin -> Needs review)
    - Duplicate student and assessment record detection
                  |
                  v
    [Review Interface & Manual Overrides]
    - Streamlit review table with original handwriting crops
    - Record-to-record navigation
    - Real-time total recalculation
                  |
                  v
    [openpyxl Excel Export]
    - Worksheet 1: Marks Summary (with dynamic Excel SUM formulas)
    - Worksheet 2: Detailed Extraction (status color fills & model tag)
    - Worksheet 3: Processing Log (audit trail & diagnostics)
```

---

## 2. Directory Structure

```
ia_bluebook_extractor/
|-- server.py                       # High-performance FastAPI REST server
|-- web/                            # Minimal Single-Page Application (HTML5 / Vanilla JS / CSS)
|   |-- index.html                  # Minimal responsive interface
|   |-- style.css                   # Modern CSS styling (Zero emojis)
|   `-- app.js                      # Client-side reactivity & API controller
|-- app.py                          # Alternative Streamlit web dashboard
|-- requirements.txt                # Production dependencies
|-- README.md                       # Comprehensive system documentation
|-- pytest.ini                      # Pytest configuration
|-- config/
|   `-- template.json               # Default 8-question IA grid template
|-- models/
|   |-- README.md                   # Model specifications and input/output schema
|   `-- digit_model.onnx            # 427 KB compact CNN running on CPUExecutionProvider
|-- scripts/
|   |-- train_digit_model.py        # Dataset synthesis and model training script
|   |-- export_onnx.py              # Model serialization utility
|   `-- verify_model.py             # ONNX Runtime CPU verification & benchmark script
|-- src/
|   |-- __init__.py
|   |-- file_loader.py              # File loader for JPG, PNG, and PDF formats
|   |-- pdf_processor.py            # PyMuPDF rendering engine
|   |-- image_preprocessor.py       # 10-stage classical CV pipeline & deskewing
|   |-- template_manager.py         # Coordinate scaling & JSON schema manager
|   |-- region_extractor.py         # Cell cropping & morphological line stripping
|   |-- digit_recognizer.py         # ONNX Runtime CPU inference & digit segmentation
|   |-- marks_validator.py          # Deterministic grading rules & audit statuses
|   |-- review_manager.py           # Batch lifecycle, duplicates, & manual overrides
|   |-- excel_exporter.py           # Multi-sheet openpyxl Excel generator
|   `-- processing_pipeline.py      # End-to-end orchestrator with failure recovery
|-- tests/
|   |-- test_preprocessing.py       # Preprocessing, skew, and resolution tests
|   |-- test_validation.py          # Validation rules, increments, and totals
|   |-- test_recognition.py         # ONNX model, tensor shapes, and digit tests
|   |-- test_excel_export.py        # openpyxl workbook structure & formulas
|   |-- test_server_api.py          # FastAPI web endpoints & batch review tests
|   |-- test_end_to_end.py          # Complete pipeline integration tests
|   `-- benchmark_recognition.py    # Empirical accuracy evaluation on held-out data
`-- sample_data/
    |-- README.md                   # Sample dataset documentation
    |-- generate_samples.py         # Synthetic bluebook generator script
    |-- clean_bluebook_template.png # Blank reference sheet
    |-- student_sample_1.png        # Standard evaluated bluebook
    |-- student_sample_2_skewed.jpg # Bluebook with +3.8 degree skew
    |-- student_sample_3_perspective.png # Camera perspective distortion
    |-- student_sample_4_multipage.pdf   # Scanned 2-page assessment booklet
    `-- student_sample_5_review_needed.png # Invalid marks (>10) & blank fields
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows, macOS, or Linux.

### Step 1: Create Virtual Environment

On Windows (Command Prompt / PowerShell):
```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS / Linux (Terminal):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Model & CPU Execution Provider

Verify that the local ONNX model loads with the CPU provider and inspect latency benchmarks:

```bash
python scripts/verify_model.py
```

Expected output:
```
Active Execution Providers: ['CPUExecutionProvider']
Input: name='input', shape=['batch', 1, 28, 28], type=tensor(float)
Output: name='output', shape=['batch', 10], type=tensor(float)
Benchmark over 100 iterations: Mean = 0.007 ms, P95 = 0.009 ms
[PASS] Model verified successfully with ONNX Runtime CPU Provider.
```

---

## 4. Running the Web Application

### Option A: Minimal Web App (FastAPI + Vanilla JS) - Recommended

Launch the high-performance local web server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open your browser at:
`http://localhost:8000`

Key advantages of the minimal web app:
- Zero external frontend frameworks or bulky node_modules.
- Blazing-fast page loads (< 10 ms).
- Instant client-side input editing and verification without script re-runs.
- Strict zero-emoji clean professional UI with high-contrast badge indicators.

### Option B: Streamlit Dashboard

Alternatively, run the multi-step Streamlit interface:

```bash
streamlit run app.py
```

The application opens at:
`http://localhost:8501`

---

## 5. Walkthrough: Testing with Sample Images

1. **Dashboard**: Check that the local model status displays `ONNX Model Active (CPU)`.
2. **Upload Bluebooks**:
   - Choose `IA Test - 1` and subject `Operating Systems (21CS42)`.
   - Upload any sample from `sample_data/` (e.g. `student_sample_1.png`, `student_sample_2_skewed.jpg`, `student_sample_4_multipage.pdf`).
   - Click **Prepare Batch**.
3. **Template Configuration**:
   - Inspect the visual bounding boxes overlaid on the reference sheet.
   - Question marks regions and Student ID box coordinates can be adjusted if using custom college formats.
4. **Image Preprocessing & Crop Preview**:
   - Inspect the side-by-side view: original vs deskewed/binarized sheet.
   - Inspect the clean individual question crops with printed borders stripped.
5. **Batch Recognition**:
   - Click **Execute Batch Recognition**.
   - Progress bar tracks document processing with isolated failure protection.
6. **Marks Verification**:
   - Inspect the detected values alongside the high-resolution handwriting snippets.
   - Navigate between students using `< Previous Record` and `Next Record >`.
   - Edit any candidate mark directly; totals update in real-time.
7. **Excel Export**:
   - Review the summary ledger.
   - Click **Download Formatted Excel Workbook (.xlsx)**.

---

## 6. Retraining and Customizing the ONNX Model

To train the compact classifier on new digit styles or actual bluebook digit crops:

```bash
python scripts/train_digit_model.py
```

This generates an augmented dataset across diverse fonts, thicknesses, and affine rotations, trains the neural network weights, and exports `models/digit_model.onnx` ready for ONNX Runtime.

---

## 7. Automated Test Suite and Recognition Benchmark

### Run Automated Unit Tests (30 Tests)
```bash
pytest -v
```

### Run Held-Out Test Set Benchmark
```bash
python tests/benchmark_recognition.py
```

Benchmark performance on the held-out test set:
- Total marks fields evaluated: `32`
- Correctly resolved pre-correction: `32 (100.0%)`
- Correctly caught invalid marks: `1 (100%)`
- Correctly caught blank cells: `1 (100%)`
- Unflagged errors (false accepts): `0 (0.0%)`
- Average CPU inference latency per digit: `< 0.01 ms`

---

## 8. Remaining Limitations and Real-World Considerations

1. **Heavily Touching Multi-Digits**: If two digits are drawn with overlapping loops or connecting ligatures, projection profile valley splitting may flag the cell as ambiguous. The system holds the cell for human evaluator verification rather than guessing.
2. **Non-Standard Table Layouts**: The template system assumes a consistent printed layout across an institution's examination booklets. When deploying at a college with a different booklet format, recalibrate coordinates once in Section 3 and save the resulting `template.json`.
3. **Ink Bleed & Extreme Smudging**: Heavy reverse-side ink bleed or excessive erasing should be reviewed manually in the verification interface using the original cell crop.
