"""Stage 2 — flatten the graph into Veo 3.1 prompts.

This is where the graph's *connections* pay off: for each beat we resolve its
APPEARS_IN entities (so the exact same character description is re-injected every
cut), its LOCATED_IN location, and the global Style — then Gemini writes one
cinematic, self-contained prompt per beat. Prose only; the graph/JSON never
reaches Veo. Prompts are written back onto the Beat nodes.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .gemini import structured
from .scene_graph import SceneGraph


class _SegPrompt(BaseModel):
    beat_no: int
    prompt: str = Field(description="Complete Veo 3.1 prompt: subject, action, camera, "
                                    "lighting, style, ambiance, then an explicit audio line")
    negative_prompt: str = Field(default="no watermark, no subtitles, no distorted faces")


class _Serialized(BaseModel):
    segments: List[_SegPrompt]


_SYSTEM = (
    "You are a Veo 3.1 prompt engineer. You turn a structured beat (with its "
    "resolved cast, location and global style) into ONE flowing, cinematic, "
    "single-shot prompt: subject + setting + action + camera move + lighting + "
    "style + ambiance, then an explicit audio line (Veo renders native audio). "
    "Output ONLY JSON matching the schema."
)


def _beat_context(g: SceneGraph, beat) -> str:
    ents = g.entities_in(beat.id)
    loc = g.location_of(beat.id)
    style = g.style()
    lines = [
        f"beat_no: {beat.beat_no}",
        f"section/pace: {beat.section} / {beat.pace}",
        f"duration_seconds: {beat.target_seconds}",
        f"continues_previous: {beat.continues_previous}",
        f"transition_in/out: {beat.transition_in} / {beat.transition_out}",
        f"is_title_card: {beat.is_title_card}",
        f"visual: {beat.visual}",
        f"voiceover: {beat.voiceover or '(none)'}",
        f"dialogue: {beat.dialogue or '(none)'}",
        f"on_screen_text: {beat.on_screen_text or '(none)'}",
        f"mood: {beat.mood}",
    ]
    if ents:
        lines.append("cast in shot (re-describe EXACTLY, identically every beat):")
        for e in ents:
            lines.append(f"  - {e.name} [{e.role}]: {e.description}")
    if loc:
        lines.append(f"location: {loc.name} — {loc.description}")
    if style:
        lines.append(f"global style: palette={style.palette}; stock={style.film_stock}; "
                     f"lens={style.lens}; lighting={style.lighting}; grade={style.grade}; "
                     f"mood={style.mood}")
    return "\n".join(lines)


_PROMPT = """Write {n} Veo 3.1 prompts, one per beat, in order. Rules:
- Match each beat's pace ("fast" = punchy, quick camera; "slow" = lingering).
- Scale spoken content to duration (an 8s beat = one short line; 4s = a few words
  or none). Include any voiceover/dialogue verbatim as spoken audio + SFX/music.
- If continues_previous is true, open by matching the previous shot's framing for a
  seamless continuation; otherwise it is a fresh cut.
- If transition_in is fade_in, open from darkness (the fade is added in post).
- For the title-card beat, describe an elegant hold; the title text is rendered in
  post, so do NOT rely on Veo to draw letters.
- Keep the cast description identical to what is given, every time.

BEATS:
{beats}
"""


def serialize_prompts(g: SceneGraph) -> SceneGraph:
    """Write a Veo prompt onto each Beat node (in place) and return the graph."""
    beats = g.ordered_beats()
    blocks = "\n\n".join(_beat_context(g, b) for b in beats)
    out = structured(_PROMPT.format(n=len(beats), beats=blocks), _Serialized, system=_SYSTEM)

    by_no = {s.beat_no: s for s in out.segments}
    for b in beats:
        seg = by_no.get(b.beat_no)
        if seg:
            b.prompt = seg.prompt
            b.negative_prompt = seg.negative_prompt
        else:  # defensive fallback — never leave a beat promptless
            b.prompt = b.visual or f"Cinematic {b.pace} shot for beat {b.beat_no}."
    return g
