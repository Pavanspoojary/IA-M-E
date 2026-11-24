# IA Bluebook Marks Extraction System

A web application for reading handwritten marks from Continuous Internal Assessment (IA) evaluation sheets. Upload a photo or a PDF, choose which IA columns to read, review and correct the detected marks, then download the result as an Excel workbook or CSV file.

The main interface is a static HTML, CSS, and JavaScript app in [`public/`](public/). Its primary extraction engine runs in the browser using OpenCV.js and an ONNX digit-recognition model. When run locally with FastAPI, the app can also fall back to the Python extraction endpoint if browser extraction fails.

## Contents

- [What you need](#what-you-need)
- [Run locally](#run-locally)
- [Using the app](#using-the-app)
- [How extraction works](#how-extraction-works)
- [Excel and CSV output](#excel-and-csv-output)
- [Technology](#technology)
- [Project layout](#project-layout)
- [Known limits](#known-limits)

## What you need

- Python 3.10 or later for the local FastAPI server.
- A modern browser with JavaScript and WebAssembly enabled.
- Internet access when opening the web app so the browser can load its JavaScript libraries from their CDNs. The ONNX model is served from this project.

## Run locally

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

Start the FastAPI app:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). The root page and assets are served from `public/`. The local server also exposes extraction and export fallback endpoints.

The static web app is also configured for Vercel: static assets are served from `public/`, with the lightweight API entry point in [`api/index.py`](api/index.py).

## Using the app

### 1. Upload or capture an evaluation sheet

Choose **Take Photo**, **Upload File**, or drag a file onto the upload area. Supported documents are JPG/JPEG, PNG, and PDF. The app processes one uploaded document at a time. A PDF may contain multiple evaluation sheets, one per page.

### 2. Set PDF sequence details, when applicable

When the selected file is a PDF, the **PDF sequence settings** appear automatically beside the IA selector. They do not appear for image uploads.

- The `4MW` prefix is fixed and added by the app.
- **Starting USN** contains the remaining suffix and defaults to `24AD001`. For example, the first assigned USN is `4MW24AD001`.
- **Skip USN suffixes** is optional. Enter comma-separated suffixes such as `24AD004, 24AD009`. The app skips those full USNs while assigning IDs to pages.
- USNs are assigned in PDF page order. Each page gets the next available USN.

### 3. Choose which assessment to fetch

Select **IA 1**, **IA 2**, **IA 3**, or **All IA**, then click **Fetch Marks & Crop Column**. The app processes each page and creates a separate result sheet for it.

### 4. Review detections and correct marks

The results view shows the selected column crop and the extracted marks for the active page. For each question it shows:

- The question label and handwriting crop. Click a handwriting crop to inspect it larger.
- The detected mark and recognition confidence, when available.
- A status such as **Valid**, **Needs Review**, or **NA**.
- An editable **Verified Mark** field.

Use the page selector or Previous/Next controls to review other PDF pages. Edit any incorrect value in **Verified Mark**; the calculated total updates as you edit. The app compares the calculated question sum with the handwritten total when it can read one. A mismatch is shown for review; the written total is not used to silently change question marks.

### 5. Download the results

Choose **Download Excel (.xlsx)** or **Download CSV (.csv)**. The export includes the verified values currently shown in the results table.

## How extraction works

The browser engine is implemented in [`public/browser_extractor.js`](public/browser_extractor.js) and controlled by [`public/app.js`](public/app.js).

1. **Load the document.** An image is decoded into a browser canvas. PDF.js renders every PDF page in order at 2× scale.
2. **Correct and locate the table.** OpenCV.js rectifies perspective and detects the table grid and IA columns. A pure JavaScript rectification fallback is available if OpenCV.js is unavailable.
3. **Crop the selected IA.** The engine locates question rows and crops the relevant marks column. For **All IA**, it processes IA 1, IA 2, and IA 3 separately.
4. **Isolate handwriting.** Image processing separates likely pen ink from printed grid lines, detects blank or crossed-out cells, and segments digit strokes.
5. **Recognize digits.** The ONNX model classifies normalized 28×28 digit images in ONNX Runtime Web using WebAssembly. Small shifted versions of each crop are also classified to reduce sensitivity to alignment.
6. **Validate and display.** Marks are checked against the configured cell range. Low-confidence or out-of-range results are marked for review. Blank or crossed-out cells appear as `NA`.
7. **Assign PDF USNs.** After page extraction, the app assigns the configured USN suffixes in page order, applying the fixed `4MW` prefix and skipping IDs supplied in the PDF settings.

The browser performs the normal extraction and export without sending the document to an inference API. The frontend libraries (OpenCV.js, PDF.js, ONNX Runtime Web, and SheetJS) are loaded from CDNs. If browser extraction fails while the app is being served by the local FastAPI server, the frontend attempts the local `/api/extract-sheet` endpoint as a fallback.

## Excel and CSV output

Both formats contain one row per selected IA per page, with question marks, total, and status. PDF exports use a `USN` column; image exports use a `Sheet` column. Column names follow the detected sheet layout.

### Excel (`.xlsx`)

- Uses Arial text and a formatted header row.
- For a PDF page where all three IAs were fetched, the USN cell is merged vertically across its IA 1, IA 2, and IA 3 rows and centered vertically. This merge is only applied in Excel exports.
- Single-IA exports and image uploads keep one identifier cell per row.

### CSV (`.csv`)

- Uses the same fields as the Excel summary, in plain comma-separated rows.
- Does not merge cells. With **All IA**, the USN is repeated on each IA row so every CSV row remains self-contained.
- Text fields are quoted when needed to preserve commas, quotes, or line breaks.

The filename includes the selected assessment, for example `IA_Marks_Evaluation_IA_1.xlsx` or `IA_Marks_Evaluation_All_IA.csv`.

## Technology

| Area | Implementation |
| --- | --- |
| User interface | HTML, CSS, and vanilla JavaScript in `public/` |
| Image processing and table detection | OpenCV.js, with JavaScript fallback routines |
| PDF page rendering | PDF.js |
| Handwritten digit inference | ONNX Runtime Web (WebAssembly) and `public/models/digit_model.onnx` |
| Excel generation in the browser | `xlsx-js-style` / SheetJS |
| Local API and extraction fallback | FastAPI, OpenCV-Python, PyMuPDF, and ONNX Runtime in `server.py` |
| Python Excel/CSV fallback | `openpyxl` and Python CSV utilities in `src/eval_sheet_extractor.py` |

The project also contains a separate Streamlit dashboard in [`app.py`](app.py), along with Python processing modules for preprocessing, recognition, validation, review, and workbook generation. The step-by-step workflow above describes the main `public/` web app.

## Project layout

```text
public/
  index.html             Web app interface
  app.js                 Upload, review, IA selection, and download flow
  browser_extractor.js   Browser image/PDF processing and export code
  models/digit_model.onnx Handwritten digit model used in the browser
src/
  eval_sheet_extractor.py Python extraction and export fallback
  pdf_processor.py       Python PDF rendering utilities
  image_preprocessor.py  Image cleanup and normalization
  digit_recognizer.py    Python ONNX digit recognizer
  marks_validator.py     Python-side mark validation
server.py                Local FastAPI app and fallback endpoints
api/index.py             Vercel API entry point
app.py                   Separate Streamlit dashboard
config/template.json     Python pipeline sheet template
sample_data/             Example evaluation sheets and PDFs
tests/                   Python tests and recognition benchmark
```

## Known limits

- Recognition quality depends on the scan or photo: keep the full table in frame, use good lighting, and avoid blur, glare, heavy skew, and cut-off marks.
- The browser grid detector is designed for the evaluation sheet layouts represented by this project. Different printed layouts may need changes to the grid detection and row/column definitions in `public/browser_extractor.js`.
- Handwriting that touches printed lines, has overlapping digits, or is faint can be misread. Review the handwriting crop and correct the verified mark before exporting.
- The browser classifier recognizes digits; it does not read student names or USNs from the handwritten sheet. In PDF sequence mode, USNs are assigned from page order using the starting and skipped suffixes you provide.
- The browser needs access to the configured CDN URLs to load its processing libraries. When deployed without those libraries, the static Vercel API endpoints do not provide server-side inference; the local FastAPI server provides the Python fallback.

## Development checks

Install `requirements-dev.txt`, then run the existing Python test suite with:

```bash
pytest -v
```

Run the recognition benchmark with:

```bash
python tests/benchmark_recognition.py
```
