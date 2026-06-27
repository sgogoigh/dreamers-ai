"""Self-contained trailer-generation pipeline.

A standalone copy of the logic in ../../scripts/trailer, vendored here so the
backend has no imports outside its own folder. Layout:

  constants.py       model ids + trailer defaults
  schema.py          the Pydantic content models (script / prompts / beats)
  adapter.py         step 1 — fine-tuned LoRA draft scene (GPU)
  script_writer.py   step 2 — Gemini structured trailer script
  prompt_builder.py  step 3 — Gemini chained Veo 3.1 prompts
  video.py           step 4 — Veo clip generation + chaining + ffmpeg stitch
"""
