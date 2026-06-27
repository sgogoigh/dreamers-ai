"""Step 3 — expand each trailer beat into a strong, cinematic Veo 3.1 prompt.

Vendored from scripts/trailer/step3_build_prompts.py.
"""
from __future__ import annotations

from .constants import GEMINI_MODEL, THINKING_LEVEL, SEG_SECONDS
from .schema import TrailerScript, TrailerPrompts

SYSTEM = (
    "You are a Veo 3.1 prompt engineer. You convert trailer beats into complete, "
    "cinematic, single-shot video prompts. Each prompt is self-contained and "
    "specifies subject, setting, action, camera (shot size + motion), lighting, "
    "film style, ambiance, and audio (Veo renders native audio: describe voiceover, "
    "dialogue, SFX, and music). Output ONLY JSON matching the schema."
)

PROMPT_TMPL = """Convert this trailer script into {n} Veo 3.1 segment prompts — one per beat,
in order. Each clip is {secs} seconds.

GLOBAL CONSISTENCY (apply to EVERY prompt so the trailer looks like one film):
- Title: {title}
- Genre/Tone: {genre} / {tone}
- Cast & look must stay identical across beats. Re-describe recurring characters
  the SAME way every time (same age, build, hair, wardrobe) so Veo keeps them
  consistent across cuts.
- Maintain one coherent color grade / film style across all segments.

PER-PROMPT RULES:
- Write one flowing paragraph: subject + setting + action + camera move + lighting
  + style + ambiance, then an explicit audio line.
- Audio: include the beat's voiceover/dialogue verbatim if present (as spoken
  audio), plus fitting SFX and trailer music cues. Keep spoken lines short.
- If `continues_previous` is true, begin the prompt by matching the previous
  shot's framing/lighting for a seamless continuation; otherwise it's a fresh cut.
- Put text-on-screen (title cards) in `on_screen_text` beats as bold centered
  typography described in the prompt.
- Provide a short `negative_prompt` to suppress artifacts (e.g. "no on-screen
  text glitches, no watermark, no distorted faces, no subtitles").
- Preserve each beat's `continues_previous` flag in the output.

TRAILER SCRIPT (JSON):
{script_json}
"""


def build_segment_prompts(script: TrailerScript, seg_seconds: int = SEG_SECONDS) -> TrailerPrompts:
    from google import genai
    from google.genai import types

    from ..config import require_gemini_key

    client = genai.Client(api_key=require_gemini_key())
    prompt = PROMPT_TMPL.format(
        n=len(script.beats), secs=seg_seconds, title=script.title,
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
    # Align flags/order with the source script (the LLM occasionally drifts).
    by_beat = {b.beat_no: b for b in script.beats}
    prompts.segments.sort(key=lambda s: s.beat_no)
    for seg in prompts.segments:
        beat = by_beat.get(seg.beat_no)
        if beat is not None:
            seg.continues_previous = beat.continues_previous
    if prompts.segments:
        prompts.segments[0].continues_previous = False
    return prompts
