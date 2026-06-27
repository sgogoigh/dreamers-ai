"""Backend configuration — fully self-contained.

This package owns its entire pipeline (see app/pipeline/), so there is NO import
of, or sys.path dependency on, the surrounding `dreamers-ai` project. The only
outside touch is an *optional* convenience: if no backend-local .env supplies the
Gemini key, we fall back to reading the parent project's .env so an existing key
keeps working. Set everything in backend/.env to be 100% independent.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# Windows consoles default to cp1252 and crash on emoji / em-dashes / model-
# generated Unicode. Force UTF-8 on stdout/stderr for every entry point.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

# app/config.py -> app -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Own .env first; then an optional fallback to the parent project's .env so a
# pre-existing GEMINI_API_KEY / HF_TOKEN keeps working without being copied.
load_dotenv(BACKEND_ROOT / ".env")
_PARENT_ENV = BACKEND_ROOT.parent / ".env"
if _PARENT_ENV.exists():
    load_dotenv(_PARENT_ENV, override=False)

from .pipeline.constants import (
    GEMINI_MODEL,
    VEO_MODEL,
    ADAPTER_REPO,
    BASE_MODEL,
    N_SEGMENTS,
    SEG_SECONDS,
    RESOLUTION,
    ASPECT_RATIO,
)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Process-wide settings, resolved from env at import time."""

    # --- mode -------------------------------------------------------------
    # In MOCK mode no paid API is called: Gemini/Veo are stubbed with valid,
    # deterministic fixtures (and real ffmpeg-generated placeholder clips), so
    # the entire flow is exercisable for free in tests and local dev.
    MOCK: bool = _env_bool("DREAMERS_MOCK", False)

    # --- data / artifacts -------------------------------------------------
    DATA_DIR: Path = Path(os.getenv("DREAMERS_DATA_DIR", str(BACKEND_ROOT / "data")))

    # --- keys / capabilities ---------------------------------------------
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")

    # --- model ids (from app/pipeline/constants.py) -----------------------
    GEMINI_MODEL: str = GEMINI_MODEL
    VEO_MODEL: str = VEO_MODEL
    ADAPTER_REPO: str = ADAPTER_REPO
    BASE_MODEL: str = BASE_MODEL

    # --- trailer defaults -------------------------------------------------
    N_SEGMENTS: int = N_SEGMENTS
    SEG_SECONDS: int = SEG_SECONDS
    RESOLUTION: str = RESOLUTION
    ASPECT_RATIO: str = ASPECT_RATIO

    # --- http -------------------------------------------------------------
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("DREAMERS_CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    @property
    def projects_dir(self) -> Path:
        return self.DATA_DIR / "projects"

    @property
    def gemini_available(self) -> bool:
        return self.MOCK or bool(self.GEMINI_API_KEY)

    @property
    def adapter_available(self) -> bool:
        """True only if the GPU stack (torch/peft) is importable. The local box
        has no GPU, so step 1 normally degrades to manual draft input."""
        if self.MOCK:
            return True
        try:
            import torch  # noqa: F401
            import peft  # noqa: F401
        except Exception:
            return False
        return True

    def ensure_dirs(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()


def require_gemini_key() -> str:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to backend/.env (or the parent "
            "dreamers-ai/.env), or run with DREAMERS_MOCK=1."
        )
    return settings.GEMINI_API_KEY
