"""Root FastAPI entrypoint for Vercel deployment and local execution.

Serves health and status endpoints for the In-Browser Neural Extraction web application.
Static assets and ONNX models are served automatically from public/ via Vercel Edge CDN.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
INDEX_FILE = PUBLIC_DIR / "index.html"

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
    return HTMLResponse(content="<h1>IA Bluebook Marks Extraction System</h1>")


if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public_root")


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
