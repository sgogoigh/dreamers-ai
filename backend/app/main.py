"""Dreamers-AI trailer backend — FastAPI app.

Self-contained service implementing the full trailer pipeline (LoRA adapter
draft -> Gemini trailer script -> Gemini Veo prompts -> Veo chained video; see
app/pipeline/) behind a review-friendly REST + SSE API. Run:

    uvicorn app.main:app --reload        # from the backend/ directory
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import BACKEND_ROOT, settings
from .routers import meta, pipeline, projects

FRONTEND_DIR = BACKEND_ROOT / "frontend"

app = FastAPI(
    title="Dreamers-AI Trailer API",
    version="1.0.0",
    description=(
        "Generate movie trailers end to end: optional fine-tuned LoRA draft -> "
        "Gemini trailer script (with AI revision) -> chained Veo 3.1 prompts -> "
        "Veo video, stitched to one file. Each stage is reviewable and editable."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"
app.include_router(meta.router, prefix=API_PREFIX)
app.include_router(projects.router, prefix=API_PREFIX)
app.include_router(pipeline.router, prefix=API_PREFIX)


@app.get("/api/info")
def info() -> dict:
    return {
        "name": "Dreamers-AI Trailer API",
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
        "config": f"{API_PREFIX}/config",
        "mock": settings.MOCK,
    }


@app.get("/", include_in_schema=False)
def index():
    """Serve the single-page trailer studio UI (falls back to API info)."""
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html, media_type="text/html")
    return JSONResponse(info())
