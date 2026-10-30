"""Root entrypoint for local execution and Vercel serverless."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# If heavy backend dependencies are available (local/Docker), load full server.py
try:
    from server import app
except Exception:
    from api.index import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
