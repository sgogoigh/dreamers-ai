"""Env loading + model ids for the graph pipeline (self-contained).

Reads graph/.env (copy of the project key). Windows consoles default to cp1252
and crash on model-generated Unicode, so we force UTF-8 on stdout/stderr.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

GRAPH_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(GRAPH_ROOT / ".env")
# fall back to the parent project's .env too (same convenience as the backend)
_PARENT_ENV = GRAPH_ROOT.parent / ".env"
if _PARENT_ENV.exists():
    load_dotenv(_PARENT_ENV, override=False)

GEMINI_MODEL = "gemini-3.5-flash"
VEO_MODEL = "veo-3.1-generate-preview"
THINKING_LEVEL = "medium"

RESOLUTION = "720p"
ASPECT_RATIO = "16:9"
VALID_DURATIONS = (4, 6, 8)


def snap_duration(seconds: int) -> int:
    """Snap any requested clip length to the nearest Veo-valid duration (4/6/8);
    ties round up so a '5s' beat becomes 6s."""
    if seconds <= VALID_DURATIONS[0]:
        return VALID_DURATIONS[0]
    if seconds >= VALID_DURATIONS[-1]:
        return VALID_DURATIONS[-1]
    return min(VALID_DURATIONS, key=lambda d: (abs(d - seconds), -d))


def gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def require_gemini_key() -> str:
    key = gemini_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy the project .env into graph/.env "
            "(cp ../.env ./.env) or export GEMINI_API_KEY."
        )
    return key
