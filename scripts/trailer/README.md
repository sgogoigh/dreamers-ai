# Trailer-generation pipeline

Turns a movie concept into a ~1-minute, chained Veo 3.1 trailer.

```
concept (+ optional adapter draft)
  │
  ├─ step 1  scripts/trailer/step1_generate_scene.py   LoRA adapter draft scene   [GPU only]
  ├─ step 2  scripts/trailer/step2_refine_script.py    → structured trailer script (gemini-3.5-flash)
  ├─ step 3  scripts/trailer/step3_build_prompts.py    → chained Veo segment prompts (gemini-3.5-flash)
  └─ step 4  scripts/trailer/step4_generate_video.py   → 8 × 8s clips, chained + stitched (veo-3.1)
```

## Why this shape
The fine-tuned adapter alone is **not** trailer-ready — its output is noisy and
fragmentary (see `../../LIMITATIONS.md`). So the adapter is used only as an
optional *draft seed*; `gemini-3.5-flash` cleans it, locks a consistent cast, and
restructures it into a proper trailer arc, then writes the Veo prompts.

A single Veo clip is max **8s**, so a trailer is built by **chaining 8 segments**
(≈64s). Continuation beats are seeded with the **last frame** of the previous clip
(or Veo's native video extension with `--native-extend`); hard-cut beats reuse an
**anchor reference image** so the protagonist/look stays consistent across cuts.

## Setup
```bash
pip install -r ../../requirements.txt          # google-genai, pydantic, imageio[-ffmpeg], pillow, …
# .env at repo root must contain GEMINI_API_KEY (and HF_TOKEN for step 1)
```

## Run
```bash
# from the repo root, module form (so relative imports resolve):

# Dry run — steps 2+3 only, no video cost. Inspect the JSON it writes.
python -m scripts.trailer.run_pipeline \
  --concept "A lighthouse keeper discovers the fog is alive" \
  --genre "Psychological Horror" --tone "dread, isolation" --dry-run

# Full run (generates video — billed per clip):
python -m scripts.trailer.run_pipeline --concept "…" --genre "…" --tone "…"

# With an adapter draft produced on a GPU:
python -m scripts.trailer.run_pipeline --concept "…" --raw-scene-file draft.txt
```

Each step is also runnable standalone (`python -m scripts.trailer.step2_refine_script --help`).

## Outputs (in `--out-dir`, default `trailer_out/`)
- `trailer_script.json` — structured beats (step 2)
- `trailer_prompts.json` — per-segment Veo prompts (step 3)
- `segment_NN.mp4` — individual clips (step 4)
- `trailer.mp4` — final stitched trailer

## Notes / gotchas
- **Cost & time:** 8 clips = 8+ Veo long-running ops. Use `--dry-run` first.
- **Step 1 needs a GPU** and an HF account that accepted the gated Llama-3.2-3B
  license. On CPU, skip it (Gemini writes from the concept) or supply a draft.
- **720p while chaining** — Veo extension input must be 720p/≤141s. Upscale at the end if you need 1080p.
- `--native-extend` uses Veo's `video=` extension for continuation beats instead
  of last-frame seeding; smoother, but every continuation must come from a 720p clip.
