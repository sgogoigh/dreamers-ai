"""Step 2 — refine a noisy adapter draft into a structured TRAILER SCRIPT.

Uses gemini-3.5-flash with structured (JSON-schema) output. Vendored from
scripts/trailer/step2_refine_script.py.
"""
from __future__ import annotations

from .constants import GEMINI_MODEL, THINKING_LEVEL, N_SEGMENTS
from .schema import TrailerScript

SYSTEM = (
    "You are a film trailer writer for a studio marketing team. You turn rough, "
    "noisy screenplay drafts into tight, structured movie-trailer scripts. "
    "Be direct and cinematic. Output ONLY valid JSON matching the schema."
)

PROMPT_TMPL = """Build a movie TRAILER SCRIPT from the material below.

CONCEPT: {concept}
GENRE: {genre}
TONE: {tone}

DRAFT SCENE(S) FROM A FINE-TUNED MODEL (rough, may contain OCR artifacts, stray
page numbers, inconsistent character names — CLEAN these, do not copy them):
\"\"\"
{draft}
\"\"\"

Requirements:
- Produce exactly {n} beats. They must form a trailer arc:
  hook -> setup -> escalation -> climax -> title_card -> (optional) stinger.
- Lock a SMALL, CONSISTENT cast (2-4 named characters). Reuse the same names and
  descriptions across beats. Discard any hallucinated/throwaway names from the draft.
- Each beat = one ~8 second shot. Keep voiceover/dialogue lines SHORT (a trailer
  beat has time for ~1 short line).
- Strip all OCR noise, form-feeds, and stray numbers. Never include them.
- `continues_previous` = true ONLY when a beat is the same continuous shot as the
  one before it (no hard cut). Most trailer beats are fresh cuts (false).
- Give the film a title and a one-sentence logline.
"""


def refine_to_trailer_script(
    concept: str,
    draft: str,
    genre: str = "",
    tone: str = "",
    n_segments: int = N_SEGMENTS,
) -> TrailerScript:
    from google import genai
    from google.genai import types

    from ..config import require_gemini_key

    client = genai.Client(api_key=require_gemini_key())
    prompt = PROMPT_TMPL.format(
        concept=concept, genre=genre or "(infer)", tone=tone or "(infer)",
        draft=draft.strip()[:8000], n=n_segments,
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
            response_mime_type="application/json",
            response_schema=TrailerScript,
        ),
    )
    script: TrailerScript = resp.parsed
    # Defensive: renumber beats so downstream chaining is reliable.
    for i, beat in enumerate(script.beats, start=1):
        beat.beat_no = i
    if script.beats:
        script.beats[0].continues_previous = False  # first beat can't continue anything
    return script
