"""Step 2 — refine the noisy adapter draft into a structured TRAILER SCRIPT.

Uses gemini-3.5-flash with structured (JSON-schema) output. This is where we
fix the adapter's limitations (LIMITATIONS.md L1-L5): we strip OCR junk, lock a
consistent cast, and re-shape fragments into a proper trailer arc.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from .config import (GEMINI_MODEL, THINKING_LEVEL, N_SEGMENTS, TrailerScript,
                     require_gemini_key)

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
    script.beats[0].continues_previous = False  # first beat can't continue anything
    return script


if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument("--draft-file", help="path to raw adapter scene; '-' for stdin")
    p.add_argument("--genre", default="")
    p.add_argument("--tone", default="")
    p.add_argument("--out", default="trailer_script.json")
    a = p.parse_args()

    if a.draft_file == "-":
        draft = sys.stdin.read()
    elif a.draft_file:
        draft = open(a.draft_file, encoding="utf-8").read()
    else:
        draft = ""

    script = refine_to_trailer_script(a.concept, draft, a.genre, a.tone)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(script.model_dump_json(indent=2))
    print(f"Wrote {a.out}: {script.title} — {len(script.beats)} beats")
