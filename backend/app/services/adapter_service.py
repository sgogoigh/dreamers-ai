"""Step 1 — optional draft scene from the fine-tuned LoRA adapter.

The adapter (`sgogoi/Llama-fine-tune-movies` on Llama-3.2-3B-Instruct) is a
QLoRA draft generator. Its output is intentionally a *rough* seed — noisy,
fragmentary, character-inconsistent (see dreamers-ai/LIMITATIONS.md) — that
step 2 (Gemini) cleans up. It requires a GPU + torch/peft, which the typical
deploy box lacks, so this service degrades gracefully: callers can always
supply a manual draft instead, and the adapter path raises a clear, typed error
when the GPU stack is absent.
"""
from __future__ import annotations

from ..config import settings


class AdapterUnavailable(RuntimeError):
    """Raised when an adapter draft is requested but the GPU stack isn't present."""


# Lazily cache the loaded model so repeated requests don't reload weights.
_MODEL = None
_TOK = None


def _mock_draft(concept: str, genre: str, theme: str, tone: str) -> str:
    return (
        "INT. MOCK LOCATION - NIGHT\n\n"
        f"A rough, fragmentary draft scene seeded from the concept: {concept}.\n"
        f"(genre={genre or 'n/a'}, theme={theme or concept}, tone={tone or 'n/a'})\n\n"
        "CHARACTER\n  This is deliberately noisy adapter-style output that step 2 "
        "will clean and restructure into a trailer arc.\n"
    )


def generate_draft(
    *,
    concept: str,
    genre: str = "",
    tone: str = "",
    max_new_tokens: int = 400,
) -> str:
    """Generate a rough draft scene with the LoRA adapter (GPU required)."""
    if settings.MOCK:
        return _mock_draft(concept, genre, concept, tone)

    if not settings.adapter_available:
        raise AdapterUnavailable(
            "The fine-tuned adapter requires a GPU with torch/peft/bitsandbytes "
            "and an HF token licensed for the gated base model. This server has "
            "no GPU stack. Generate the draft on a GPU (Colab / an inference "
            "endpoint) and submit it via the manual draft option, or skip step 1 "
            "and let Gemini write the script from the concept."
        )

    global _MODEL, _TOK
    from ..pipeline.adapter import generate_raw_scene, load_model

    if _MODEL is None:
        _MODEL, _TOK = load_model()
    return generate_raw_scene(
        concept=concept,
        genre=genre,
        theme=concept,
        tone=tone,
        previous_scene=concept,
        model=_MODEL,
        tok=_TOK,
        max_new_tokens=max_new_tokens,
    )
