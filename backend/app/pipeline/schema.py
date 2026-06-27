"""Pydantic content models passed between pipeline steps.

Vendored from scripts/trailer/config.py so the backend owns its own schema.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


# --- step 2 output ---------------------------------------------------------
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


# --- step 3 output ---------------------------------------------------------
class SegmentPrompt(BaseModel):
    """A single Veo 3.1 segment: a fully-formed cinematic prompt + chaining info."""
    beat_no: int
    prompt: str = Field(description="Complete Veo 3.1 prompt (subject, action, camera, lighting, style, audio)")
    negative_prompt: str = Field(default="", description="Things to avoid")
    continues_previous: bool = Field(default=False)


class TrailerPrompts(BaseModel):
    segments: List[SegmentPrompt]
