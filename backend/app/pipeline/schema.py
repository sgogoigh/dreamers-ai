"""Pydantic content models passed between pipeline steps.

Vendored from scripts/trailer/config.py so the backend owns its own schema.

The **TrailerBlueprint** is the structural contract for a trailer: a small typed
graph of slots (nodes) whose `continues_previous` flags are the CONTINUES-vs-CUT
edges. It declares each shot's role, pace, target duration, and fade transition
UP FRONT, so the creative model (step 2) only fills content — it cannot change
the structure. The blueprint drives step 2, is an editable project artifact, and
its structural fields are stamped deterministically onto beats/segments (no LLM
drift). See `default_blueprint()` for the canonical 8-shot trailer.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

# Fade transition applied at STITCH time (ffmpeg), not by Veo.
Transition = Literal["none", "fade_in", "fade_out"]
Pace = Literal["slow", "medium", "fast"]
Section = Literal["hook", "setup", "escalation", "climax", "title_card", "stinger"]


# --- blueprint (the structural graph) --------------------------------------
class Slot(BaseModel):
    """One node in the trailer blueprint — a shot's structural spec.

    Purely structural: what ROLE the shot plays and HOW it is rendered/cut. The
    creative content (setting/visual/dialogue) is filled by step 2 to match.
    """
    beat_no: int = Field(description="1-based ordinal of this slot")
    section: Section
    pace: Pace = "medium"
    target_seconds: int = Field(default=8, description="Desired clip length (snapped to Veo's 4/6/8)")
    continues_previous: bool = Field(
        default=False,
        description="CONTINUES edge: same continuous shot as the previous slot (no hard cut)",
    )
    transition_in: Transition = "none"
    transition_out: Transition = "none"
    is_title_card: bool = Field(default=False, description="This slot is the movie-title card")
    intent: str = Field(default="", description="Human note on what this shot must accomplish")


class TrailerBlueprint(BaseModel):
    """The ordered set of slots that structures a trailer (a linear DAG)."""
    slots: List[Slot]


def default_blueprint() -> TrailerBlueprint:
    """The canonical 8-shot trailer structure.

    1  fade-in cinematic setting + character shadow/silhouette buildup (no reveal)
    2  CONTINUES 1 — introduce character(s) + first dialogue, continue the buildup
    3-6 fast-paced cut shots with dialogue and scene resets; slot 4 is a brief
       slow-down "normal scene" (6s — Veo has no 5s)
    7  fast climax tease
    8  movie-title card, aesthetic fade-out (4s)
    """
    return TrailerBlueprint(slots=[
        Slot(beat_no=1, section="hook", pace="slow", target_seconds=8,
             continues_previous=False, transition_in="fade_in",
             intent="Cinematic establishing setting; build atmosphere; character in "
                    "shadow/silhouette only — do not fully reveal them yet."),
        Slot(beat_no=2, section="setup", pace="medium", target_seconds=8,
             continues_previous=True,
             intent="Continue directly from shot 1; introduce the character(s) and "
                    "their first short line; keep building tension."),
        Slot(beat_no=3, section="escalation", pace="fast", target_seconds=8,
             continues_previous=False,
             intent="Hard cut. Fast, energetic beat with a short line; new location/reset."),
        Slot(beat_no=4, section="escalation", pace="slow", target_seconds=6,
             continues_previous=False,
             intent="A brief slower 'normal' beat to let the trailer breathe between "
                    "fast cuts (with or without dialogue)."),
        Slot(beat_no=5, section="escalation", pace="fast", target_seconds=8,
             continues_previous=False,
             intent="Hard cut back to fast pacing; short punchy line; rising stakes."),
        Slot(beat_no=6, section="escalation", pace="fast", target_seconds=8,
             continues_previous=False,
             intent="Fast cut; peak of the escalation; short line."),
        Slot(beat_no=7, section="climax", pace="fast", target_seconds=8,
             continues_previous=False,
             intent="Fast climax tease — hint at the film's biggest moment without "
                    "resolving it."),
        Slot(beat_no=8, section="title_card", pace="slow", target_seconds=4,
             continues_previous=False, transition_out="fade_out", is_title_card=True,
             intent="The movie title, held on screen with an aesthetic fade to black."),
    ])


# --- shared entities (consistency) -----------------------------------------
class CastMember(BaseModel):
    """A recurring character. Re-described identically across beats so Veo keeps
    the look consistent across cuts (the anchor reference image reinforces it)."""
    name: str
    description: str = Field(description="Fixed physical description: age, build, hair, wardrobe")


# --- step 2 output ---------------------------------------------------------
class TrailerBeat(BaseModel):
    """One beat of the trailer (becomes one Veo video segment).

    Creative fields are written by the model; structural fields (section, pace,
    target_seconds, continues_previous, transitions, is_title_card) are stamped
    from the blueprint so they never drift.
    """
    beat_no: int = Field(description="1-based ordinal of this beat")
    section: Section = Field(description="Trailer act this beat belongs to")
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
    # --- structural fields (stamped from the blueprint) --------------------
    pace: Pace = Field(default="medium", description="Editorial pace of the beat")
    target_seconds: int = Field(default=8, description="Clip length (snapped to Veo's 4/6/8)")
    transition_in: Transition = Field(default="none")
    transition_out: Transition = Field(default="none")
    is_title_card: bool = Field(default=False)


class TrailerScript(BaseModel):
    """Structured trailer script produced by gemini-3.5-flash (step 2)."""
    title: str
    logline: str = Field(description="One-sentence hook for the film")
    genre: str
    tone: str
    cast: List[CastMember] = Field(
        default_factory=list,
        description="Small, consistent cast (2-4) reused verbatim across beats",
    )
    beats: List[TrailerBeat]


# --- step 3 output ---------------------------------------------------------
class SegmentPrompt(BaseModel):
    """A single Veo 3.1 segment: a fully-formed cinematic prompt + chaining info."""
    beat_no: int
    prompt: str = Field(description="Complete Veo 3.1 prompt (subject, action, camera, lighting, style, audio)")
    negative_prompt: str = Field(default="", description="Things to avoid")
    continues_previous: bool = Field(default=False)
    # --- structural fields (carried from the beat/blueprint) ---------------
    pace: Pace = Field(default="medium")
    duration_seconds: int = Field(default=8, description="Clip length for this segment (Veo 4/6/8)")
    transition_in: Transition = Field(default="none")
    transition_out: Transition = Field(default="none")
    is_title_card: bool = Field(default=False)
    title_text: str = Field(default="", description="Movie title to render on the title-card segment")


class TrailerPrompts(BaseModel):
    segments: List[SegmentPrompt]
