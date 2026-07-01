"""Thin shared wrapper over google-genai for structured JSON generation."""
from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from .config import GEMINI_MODEL, THINKING_LEVEL, require_gemini_key

T = TypeVar("T", bound=BaseModel)

_client = None


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=require_gemini_key())
    return _client


def structured(prompt: str, schema: Type[T], *, system: str = "",
               model: str = GEMINI_MODEL) -> T:
    """One structured-output call: returns a validated instance of `schema`."""
    from google.genai import types

    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = client().models.generate_content(model=model, contents=prompt, config=cfg)
    return resp.parsed  # type: ignore[return-value]
