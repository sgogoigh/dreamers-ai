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
from ..models import TrailerScript, TrailerPrompts, TrailerBeat, SegmentPrompt, CastMember
from ..pipeline.schema import TrailerBlueprint, default_blueprint

# Vendored pipeline functions (steps 2 & 3).
from ..pipeline.script_writer import refine_to_trailer_script, _stamp_structure
from ..pipeline.prompt_builder import build_segment_prompts
from ..pipeline.constants import snap_duration


# ---------------------------------------------------------------------------
# MOCK fixtures
# ---------------------------------------------------------------------------
def _mock_script(concept: str, genre: str, tone: str, blueprint: TrailerBlueprint) -> TrailerScript:
    """A schema-valid script shaped exactly by the blueprint (structure stamped)."""
    beats = []
    for slot in blueprint.slots:
        i = slot.beat_no
        beats.append(
            TrailerBeat(
                beat_no=i,
                section=slot.section,
                setting=f"Mock setting {i} for '{concept[:40]}'",
                visual=f"Mock visual beat {i} ({slot.pace}): {slot.intent}",
                voiceover="In a world..." if i == 1 else "",
                dialogue="We have to move. Now." if slot.section == "escalation" else "",
                mood=tone or "tense",
            )
        )
    script = TrailerScript(
        title=f"MOCK: {concept[:30].strip().upper() or 'UNTITLED'}",
        logline=f"A mock logline derived from: {concept}",
        genre=genre or "Drama",
        tone=tone or "tense",
        cast=[
            CastMember(name="Alex", description="early-30s, lean, dark cropped hair, grey field jacket"),
            CastMember(name="Mara", description="late-20s, tall, auburn braid, navy overcoat"),
        ],
        beats=beats,
    )
    return _stamp_structure(script, blueprint)  # same deterministic stamping as the real path


def _mock_prompts(script: TrailerScript) -> TrailerPrompts:
    segs = []
    for b in script.beats:
        segs.append(
            SegmentPrompt(
                beat_no=b.beat_no,
                prompt=(
                    f"[MOCK PROMPT beat {b.beat_no}, {b.pace}] Cinematic shot. {b.visual} "
                    f"Setting: {b.setting}. Mood: {b.mood}. Audio: trailer score swell."
                ),
                negative_prompt="no watermark, no subtitles, no distorted faces",
                continues_previous=b.continues_previous,
                pace=b.pace,
                duration_seconds=snap_duration(b.target_seconds),
                transition_in=b.transition_in,
                transition_out=b.transition_out,
                is_title_card=b.is_title_card,
                title_text=(script.title if b.is_title_card else ""),
            )
        )
    if segs:
        segs[0].continues_previous = False
    return TrailerPrompts(segments=segs)


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------
def generate_script(
    *, concept: str, draft: str, genre: str, tone: str,
    blueprint: TrailerBlueprint | None = None,
) -> TrailerScript:
    """Step 2 — structured trailer script from concept (+ optional draft), shaped
    by the trailer blueprint."""
    bp = blueprint or default_blueprint()
    if settings.MOCK:
        return _mock_script(concept, genre, tone, bp)
    return refine_to_trailer_script(
        concept=concept, draft=draft, genre=genre, tone=tone, blueprint=bp
    )


def build_prompts(*, script: TrailerScript, seg_seconds: int | None = None) -> TrailerPrompts:
    """Step 3 — expand the script into chained Veo 3.1 segment prompts.

    Per-segment duration comes from each beat's blueprint-stamped `target_seconds`;
    `seg_seconds` is retained only for call-site compatibility."""
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
