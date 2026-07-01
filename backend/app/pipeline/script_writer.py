"""Step 2 — refine a noisy adapter draft into a structured TRAILER SCRIPT.

Uses gemini-3.5-flash with structured (JSON-schema) output. Vendored from
scripts/trailer/step2_refine_script.py.

The script is generated AGAINST A BLUEPRINT (see schema.default_blueprint): the
blueprint fixes each shot's structure (role, pace, duration, continuity, fade),
and the model only fills the creative content. Structural fields are then stamped
back from the blueprint so they can never drift.
"""
from __future__ import annotations

from .constants import GEMINI_MODEL, THINKING_LEVEL, snap_duration
from .schema import TrailerScript, TrailerBlueprint, default_blueprint

SYSTEM = (
    "You are a film trailer writer for a studio marketing team. You turn rough, "
    "noisy screenplay drafts into tight, structured movie-trailer scripts that "
    "follow a fixed shot-by-shot blueprint. Be direct and cinematic. Output ONLY "
    "valid JSON matching the schema."
)

PROMPT_TMPL = """Build a movie TRAILER SCRIPT from the material below, following the
SHOT BLUEPRINT exactly.

CONCEPT: {concept}
GENRE: {genre}
TONE: {tone}

DRAFT SCENE(S) FROM A FINE-TUNED MODEL (rough, may contain OCR artifacts, stray
page numbers, inconsistent character names — CLEAN these, do not copy them):
\"\"\"
{draft}
\"\"\"

SHOT BLUEPRINT — produce EXACTLY {n} beats, one per slot below, IN THIS ORDER.
Fill each beat's creative content (setting, visual, dialogue, voiceover, mood) so
it fulfils that slot's role. Do NOT change the structure: the pace, duration,
continuation, and title-card nature of each slot are fixed and will be enforced.
{blueprint_lines}

GLOBAL RULES:
- First define a SMALL, CONSISTENT `cast` (2-4 named characters) with fixed
  physical descriptions (age, build, hair, wardrobe). Reuse the SAME names and
  descriptions across every beat so the film looks consistent across cuts.
  Discard any hallucinated/throwaway names from the draft.
- Respect each slot's pace: a "fast" beat is punchy with a quick action/short
  line; a "slow" beat lingers and lets the trailer breathe.
- A beat marked `continues_previous` must read as the SAME continuous shot as the
  beat before it (same location/framing, no hard cut). All other beats are fresh cuts.
- Keep voiceover/dialogue SHORT — one beat is a few seconds of screen time. Very
  short beats (4s) should have at most a handful of spoken words, or none.
- The title-card slot's `on_screen_text` is the film's TITLE only (no other text).
- Strip all OCR noise, form-feeds, and stray numbers. Never include them.
- Give the film a title and a one-sentence logline.
"""


def _blueprint_lines(bp: TrailerBlueprint) -> str:
    """Render the blueprint as a readable, model-facing slot list."""
    out = []
    for s in bp.slots:
        chain = "CONTINUES the previous shot" if s.continues_previous else "fresh cut"
        title = " [MOVIE TITLE CARD]" if s.is_title_card else ""
        out.append(
            f"  Slot {s.beat_no} — {s.section}, pace={s.pace}, ~{s.target_seconds}s, "
            f"{chain}{title}: {s.intent}"
        )
    return "\n".join(out)


def _stamp_structure(script: TrailerScript, bp: TrailerBlueprint) -> TrailerScript:
    """Overwrite each beat's structural fields from the blueprint (no drift).

    Aligns beat count/order to the blueprint, then copies section, pace, duration
    (snapped), continues_previous, transitions and is_title_card from each slot.
    Creative fields written by the model are preserved.
    """
    by_slot = {s.beat_no: s for s in bp.slots}
    # Renumber + order beats defensively.
    for i, beat in enumerate(script.beats, start=1):
        beat.beat_no = i
    for beat in script.beats:
        slot = by_slot.get(beat.beat_no)
        if slot is None:
            continue
        beat.section = slot.section
        beat.pace = slot.pace
        beat.target_seconds = snap_duration(slot.target_seconds)
        beat.continues_previous = slot.continues_previous
        beat.transition_in = slot.transition_in
        beat.transition_out = slot.transition_out
        beat.is_title_card = slot.is_title_card
        if slot.is_title_card:
            beat.on_screen_text = script.title
    if script.beats:
        script.beats[0].continues_previous = False  # first beat can't continue anything
    return script


def refine_to_trailer_script(
    concept: str,
    draft: str,
    genre: str = "",
    tone: str = "",
    blueprint: TrailerBlueprint | None = None,
) -> TrailerScript:
    from google import genai
    from google.genai import types

    from ..config import require_gemini_key

    bp = blueprint or default_blueprint()
    client = genai.Client(api_key=require_gemini_key())
    prompt = PROMPT_TMPL.format(
        concept=concept, genre=genre or "(infer)", tone=tone or "(infer)",
        draft=draft.strip()[:8000], n=len(bp.slots),
        blueprint_lines=_blueprint_lines(bp),
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
    return _stamp_structure(script, bp)
