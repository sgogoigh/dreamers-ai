"""Model ids + trailer defaults (vendored from scripts/trailer/config.py)."""
from __future__ import annotations

# --- model ids -------------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash"
VEO_MODEL = "veo-3.1-generate-preview"        # or veo-3.1-fast-generate-preview
ADAPTER_REPO = "sgogoi/Llama-fine-tune-movies"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
STOP_STR = "<|end_of_scene|>"

# --- trailer defaults ------------------------------------------------------
N_SEGMENTS = 8            # 8 clips
SEG_SECONDS = 8           # x 8s  =>  ~64s trailer
RESOLUTION = "720p"       # keep 720p while chaining (Veo extension constraint)
ASPECT_RATIO = "16:9"
THINKING_LEVEL = "medium"  # gemini-3.5-flash thinking level
