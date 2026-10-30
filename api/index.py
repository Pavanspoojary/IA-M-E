"""FastAPI entrypoint for Vercel Serverless deployment.

Lightweight, robust entrypoint serving health, status, and fallback endpoints.
Static assets and in-browser neural extraction are served from public/ via Vercel Edge CDN.
Strict Zero Emojis Policy Enforced.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
