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

# Veo 3.1 accepts ONLY these per-clip durations (seconds). There is no 5s, so a
# "5 second" slow-down beat is rendered at the nearest valid length.
VALID_DURATIONS = (4, 6, 8)


def snap_duration(seconds: int) -> int:
    """Snap any requested clip length to the nearest Veo-valid duration (4/6/8).

    Ties round DOWN (e.g. 5 -> 4? no) — we bias toward the LONGER neighbour on a
    tie so a "5s" beat becomes 6s (keeps a touch more content). Out-of-range
    values clamp to the ends.
    """
    if seconds <= VALID_DURATIONS[0]:
        return VALID_DURATIONS[0]
    if seconds >= VALID_DURATIONS[-1]:
        return VALID_DURATIONS[-1]
    # nearest; on a tie pick the larger (so 5 -> 6, not 4)
    return min(VALID_DURATIONS, key=lambda d: (abs(d - seconds), -d))
