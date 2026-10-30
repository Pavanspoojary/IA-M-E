"""FastAPI entrypoint for Vercel Serverless deployment.

Lightweight, robust entrypoint serving the in-browser neural extraction web application.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
INDEX_FILE = WEB_DIR / "index.html"

app = FastAPI(title="IA Bluebook Marks Extraction System")


@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "in-browser-wasm"}


@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "engine": "In-Browser Neural Extraction (ONNX Runtime Web)",
        "deployment": "Vercel Production",
    }


@app.post("/api/extract-sheet")
async def extract_sheet():
    return JSONResponse(
        content={
            "info": "Document extraction is executed client-side via in-browser ONNX Runtime Web.",
            "mode": "client-side",
        }
    )


@app.post("/api/export-sheet")
async def export_sheet():
    return JSONResponse(
        content={
            "info": "Export generation is executed client-side via in-browser SheetJS / Blob.",
            "mode": "client-side",
        }
    )


@app.get("/")
async def root():
    if INDEX_FILE.exists():
        return HTMLResponse(content=INDEX_FILE.read_text(encoding="utf-8"))
    return {"message": "IA Bluebook Marks Extraction System"}


# Mount static assets
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    """Serve relative files from web/ directory."""
    target = WEB_DIR / file_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    if INDEX_FILE.exists():
        return HTMLResponse(content=INDEX_FILE.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="File not found")
