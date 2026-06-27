"""Health + capability discovery for the frontend."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..models import ConfigResponse

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "mock": settings.MOCK}


@router.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    """Everything the frontend needs to render capabilities and defaults."""
    return ConfigResponse(
        mock=settings.MOCK,
        gemini_available=settings.gemini_available,
        adapter_available=settings.adapter_available,
        gemini_model=settings.GEMINI_MODEL,
        veo_model=settings.VEO_MODEL,
        adapter_repo=settings.ADAPTER_REPO,
        base_model=settings.BASE_MODEL,
        default_segments=settings.N_SEGMENTS,
        default_seg_seconds=settings.SEG_SECONDS,
        resolution=settings.RESOLUTION,
        aspect_ratio=settings.ASPECT_RATIO,
    )
