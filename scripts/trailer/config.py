"""Shared config, env loading, and the data models passed between pipeline steps."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- env -------------------------------------------------------------------
# .env lives at the repo root (two levels up from scripts/trailer/).
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- model ids -------------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash"
VEO_MODEL = "veo-3.1-generate-preview"        # or veo-3.1-fast-generate-preview
ADAPTER_REPO = "sgogoi/Llama-fine-tune-movies"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
STOP_STR = "<|end_of_scene|>"

# --- trailer defaults ------------------------------------------------------
N_SEGMENTS = 8            # 8 clips
SEG_SECONDS = 8           # x 8s  =>  ~64s trailer
RESOLUTION = "720p"       # keep 720p while chaining (Veo extension constraint)
ASPECT_RATIO = "16:9"
THINKING_LEVEL = "medium"  # gemini-3.5-flash thinking level


# --- data models (step 2 output) ------------------------------------------
class TrailerBeat(BaseModel):
    """One beat of the trailer (becomes one ~8s video segment)."""
    beat_no: int = Field(description="1-based ordinal of this beat")
    section: Literal["hook", "setup", "escalation", "climax", "title_card", "stinger"] = Field(
        description="Trailer act this beat belongs to"
    )
    setting: str = Field(description="Where/when this beat takes place")
    visual: str = Field(description="What we SEE — subject, action, staging (no camera jargon)")
    voiceover: str = Field(default="", description="Narrator/VO line, if any (keep short)")
    dialogue: str = Field(default="", description="Spoken character line, if any (keep short)")
    on_screen_text: str = Field(default="", description="Title-card / caption text, if any")
    mood: str = Field(description="Emotional tone of the beat")
    continues_previous: bool = Field(
        default=False,
        description="True if this beat is a direct continuation of the previous shot "
        "(same location/action, no hard cut); False if it is a fresh cut.",
    )


class TrailerScript(BaseModel):
    """Structured trailer script produced by gemini-3.5-flash (step 2)."""
    title: str
    logline: str = Field(description="One-sentence hook for the film")
    genre: str
    tone: str
    beats: List[TrailerBeat]


# --- data models (step 3 output) ------------------------------------------
class SegmentPrompt(BaseModel):
    """A single Veo 3.1 segment: a fully-formed cinematic prompt + chaining info."""
    beat_no: int
    prompt: str = Field(description="Complete Veo 3.1 prompt (subject, action, camera, lighting, style, audio)")
    negative_prompt: str = Field(default="", description="Things to avoid")
    continues_previous: bool = Field(default=False)


class TrailerPrompts(BaseModel):
    segments: List[SegmentPrompt]


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY not set in .env")
    return GEMINI_API_KEY
