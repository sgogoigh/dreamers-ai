"""Gemini-backed services: trailer script (step 2), AI revision, Veo prompts (step 3).

Steps 2 and 3 delegate to the existing pipeline functions so the API and CLI
share identical generation logic. The *revision* capability is new (the CLI has
no equivalent): it takes the current structured script plus a free-text note and
asks Gemini for a revised script that keeps the same shape.

In MOCK mode every function returns deterministic, schema-valid fixtures — no
network, no key, no cost.
"""
from __future__ import annotations

from ..config import settings
from ..models import TrailerScript, TrailerPrompts, TrailerBeat, SegmentPrompt

# Vendored pipeline functions (steps 2 & 3).
from ..pipeline.script_writer import refine_to_trailer_script
from ..pipeline.prompt_builder import build_segment_prompts


# ---------------------------------------------------------------------------
# MOCK fixtures
# ---------------------------------------------------------------------------
_SECTIONS = ["hook", "setup", "escalation", "climax", "title_card", "stinger"]


def _mock_script(concept: str, genre: str, tone: str, n: int) -> TrailerScript:
    beats = []
    for i in range(1, n + 1):
        section = _SECTIONS[min(i - 1, len(_SECTIONS) - 1)]
        beats.append(
            TrailerBeat(
                beat_no=i,
                section=section,
                setting=f"Mock setting {i} for '{concept[:40]}'",
                visual=f"Mock visual beat {i}: a character reacts as the central "
                f"conflict of '{concept[:60]}' escalates.",
                voiceover="In a world..." if i == 1 else "",
                dialogue="We have to move. Now." if section == "escalation" else "",
                on_screen_text=(concept[:24].upper() if section == "title_card" else ""),
                mood=tone or "tense",
                continues_previous=(i > 1 and i % 3 == 0),
            )
        )
    beats[0].continues_previous = False
    return TrailerScript(
        title=f"MOCK: {concept[:30].strip().upper() or 'UNTITLED'}",
        logline=f"A mock logline derived from: {concept}",
        genre=genre or "Drama",
        tone=tone or "tense",
        beats=beats,
    )


def _mock_prompts(script: TrailerScript) -> TrailerPrompts:
    segs = []
    for b in script.beats:
        segs.append(
            SegmentPrompt(
                beat_no=b.beat_no,
                prompt=(
                    f"[MOCK PROMPT beat {b.beat_no}] Cinematic shot. {b.visual} "
                    f"Setting: {b.setting}. Mood: {b.mood}. Audio: trailer score swell."
                ),
                negative_prompt="no watermark, no subtitles, no distorted faces",
                continues_previous=b.continues_previous,
            )
        )
    if segs:
        segs[0].continues_previous = False
    return TrailerPrompts(segments=segs)


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------
def generate_script(
    *, concept: str, draft: str, genre: str, tone: str, n_segments: int
) -> TrailerScript:
    """Step 2 — structured trailer script from concept (+ optional draft)."""
    if settings.MOCK:
        return _mock_script(concept, genre, tone, n_segments)
    return refine_to_trailer_script(
        concept=concept, draft=draft, genre=genre, tone=tone, n_segments=n_segments
    )


def build_prompts(*, script: TrailerScript, seg_seconds: int) -> TrailerPrompts:
    """Step 3 — expand the script into chained Veo 3.1 segment prompts."""
    if settings.MOCK:
        return _mock_prompts(script)
    return build_segment_prompts(script, seg_seconds=seg_seconds)


# --- AI revision (new) -----------------------------------------------------
_REVISE_SYSTEM = (
    "You are a film trailer script editor. You revise an existing structured "
    "trailer script according to the user's feedback while preserving its JSON "
    "schema. Keep what works; change only what the feedback asks for. Keep the "
    "cast small and consistent. Output ONLY valid JSON matching the schema."
)

_REVISE_TMPL = """Revise the trailer script below according to the feedback.

FEEDBACK:
\"\"\"{feedback}\"\"\"

RULES:
- Keep exactly {n} beats forming a coherent trailer arc
  (hook -> setup -> escalation -> climax -> title_card -> optional stinger).
- Preserve character names/descriptions unless the feedback asks to change them.
- Keep voiceover/dialogue lines short (one beat ~= one ~8s shot).
- `continues_previous` = true ONLY for a same-shot continuation; beat 1 is false.
- Return the FULL revised script, not a diff.

CURRENT SCRIPT (JSON):
{script_json}
"""


def revise_script(*, script: TrailerScript, feedback: str) -> TrailerScript:
    """Apply natural-language feedback to a script and return a revised one."""
    if settings.MOCK:
        revised = script.model_copy(deep=True)
        revised.title = f"{revised.title} (rev)"
        revised.logline = f"{revised.logline} [revised: {feedback[:60]}]"
        if revised.beats:
            revised.beats[0].visual = (
                f"[revised per feedback: {feedback[:50]}] {revised.beats[0].visual}"
            )
        return revised

    from google import genai
    from google.genai import types

    from ..config import require_gemini_key
    from ..pipeline.constants import GEMINI_MODEL, THINKING_LEVEL

    client = genai.Client(api_key=require_gemini_key())
    prompt = _REVISE_TMPL.format(
        feedback=feedback.strip(),
        n=len(script.beats),
        script_json=script.model_dump_json(indent=2),
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_REVISE_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
            response_mime_type="application/json",
            response_schema=TrailerScript,
        ),
    )
    revised: TrailerScript = resp.parsed
    for i, beat in enumerate(revised.beats, start=1):
        beat.beat_no = i
    if revised.beats:
        revised.beats[0].continues_previous = False
    return revised
