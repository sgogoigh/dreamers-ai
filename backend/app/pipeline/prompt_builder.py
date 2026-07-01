"""Step 3 — expand each trailer beat into a strong, cinematic Veo 3.1 prompt.

Vendored from scripts/trailer/step3_build_prompts.py.
"""
from __future__ import annotations

from .constants import GEMINI_MODEL, THINKING_LEVEL, snap_duration
from .schema import TrailerScript, TrailerPrompts

SYSTEM = (
    "You are a Veo 3.1 prompt engineer. You convert trailer beats into complete, "
    "cinematic, single-shot video prompts. Each prompt is self-contained and "
    "specifies subject, setting, action, camera (shot size + motion), lighting, "
    "film style, ambiance, and audio (Veo renders native audio: describe voiceover, "
    "dialogue, SFX, and music). Output ONLY JSON matching the schema."
)

PROMPT_TMPL = """Convert this trailer script into {n} Veo 3.1 segment prompts — one per beat,
in order. Each beat lists its own duration and pace — respect them.

GLOBAL CONSISTENCY (apply to EVERY prompt so the trailer looks like one film):
- Title: {title}
- Genre/Tone: {genre} / {tone}
- Cast & look must stay identical across beats. Re-describe recurring characters
  the SAME way every time (same age, build, hair, wardrobe) so Veo keeps them
  consistent across cuts. The cast is defined in the script's `cast` list — use it.
- Maintain one coherent color grade / film style across all segments.

PER-PROMPT RULES:
- Write one flowing paragraph: subject + setting + action + camera move + lighting
  + style + ambiance, then an explicit audio line.
- Match the beat's `pace`: a "fast" beat is a punchy, high-energy quick shot with
  brisk camera movement; a "slow" beat lingers with slower, deliberate motion.
- Scale spoken content to `target_seconds`: an 8s beat has time for one short
  line; a 4s beat has a few words or none. Keep all spoken lines short.
- Audio: include the beat's voiceover/dialogue verbatim if present (as spoken
  audio), plus fitting SFX and trailer music cues.
- If `continues_previous` is true, begin the prompt by matching the previous
  shot's framing/lighting for a seamless continuation; otherwise it's a fresh cut.
- If `transition_in` is "fade_in", the shot should emerge from darkness; the fade
  itself is added in post, so just describe opening from black/low light.
- For the title-card beat (`is_title_card`), describe an elegant hold suitable for
  the movie title; the title text itself is rendered in post (do NOT rely on Veo
  to draw legible letters).
- Provide a short `negative_prompt` to suppress artifacts (e.g. "no on-screen
  text glitches, no watermark, no distorted faces, no subtitles").

TRAILER SCRIPT (JSON):
{script_json}
"""


def build_segment_prompts(script: TrailerScript, seg_seconds: int | None = None) -> TrailerPrompts:
    """Expand each beat into a Veo prompt. Per-beat duration comes from the beat's
    own `target_seconds` (blueprint-driven); `seg_seconds` is an ignored legacy
    arg kept for call-site compatibility."""
    from google import genai
    from google.genai import types

    from ..config import require_gemini_key

    client = genai.Client(api_key=require_gemini_key())
    prompt = PROMPT_TMPL.format(
        n=len(script.beats), title=script.title,
        genre=script.genre, tone=script.tone,
        script_json=script.model_dump_json(indent=2),
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
            response_mime_type="application/json",
            response_schema=TrailerPrompts,
        ),
    )
    prompts: TrailerPrompts = resp.parsed
    # Stamp structural fields from the source beats (the LLM only writes prose).
    by_beat = {b.beat_no: b for b in script.beats}
    prompts.segments.sort(key=lambda s: s.beat_no)
    for seg in prompts.segments:
        beat = by_beat.get(seg.beat_no)
        if beat is None:
            continue
        seg.continues_previous = beat.continues_previous
        seg.pace = beat.pace
        seg.duration_seconds = snap_duration(beat.target_seconds)
        seg.transition_in = beat.transition_in
        seg.transition_out = beat.transition_out
        seg.is_title_card = beat.is_title_card
        if beat.is_title_card:
            seg.title_text = script.title
    if prompts.segments:
        prompts.segments[0].continues_previous = False
    return prompts
