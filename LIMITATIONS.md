# Dreamers-AI — Known Limitations & Observations

A running log of what we've found about the data, the fine-tuned model, and the
pipeline. Updated as we go. Newest sections at the bottom.

---

## 1. The fine-tuned adapter (`sgogoi/Llama-fine-tune-movies`)

**What it is:** QLoRA / PEFT adapter (r=64, α=128, all 7 attn+MLP proj modules,
`modules_to_save: null`) on base `meta-llama/Llama-3.2-3B-Instruct`. Trained 2
epochs / 4448 steps, train loss 1.758 → 1.135. Uploaded 2025-12-21.

**Verified by running it on a T4** (see `scripts/evaluate_adapter.ipynb`,
n=6 validation prompts — all from *10 Cloverfield Lane*, so treat percentages as
indicative, not population stats):

| Behaviour | Result | Verdict |
|---|---|---|
| Emits `<\|end_of_scene\|>` and stops cleanly | 100% | ✅ Solid |
| Runaway (never stops, hits token budget) | 0% | ✅ |
| Output is correctly-formatted screenplay (sluglines, cues, dialogue) | ~83% | ✅ Strong |
| Echoes form-feed (`\f`) OCR junk from training data | ~33% | ⚠️ Contamination |

**Strengths:** screenplay format and scene termination are genuinely well-learned.
The training recipe was sound. Usable as a *draft* scene generator.

**Limitations:**
- **L1 — OCR contamination bleeds into output.** ~33% of generations carry
  form-feed artifacts; the training corpus had ~43%. Source `raw_texts/` are
  OCR-grade scrapes that were never denoised.
- **L2 — Page-number leakage.** Generations contain stray standalone numbers
  mid-scene (e.g. `20.`, `76.`) — scanned-script page numbers the scene splitter
  never stripped. New finding from generation (the static scan missed it).
- **L3 — OCR character corruption.** e.g. `MICHelle`, `0.S.` (zero instead of
  `O.S.`). Straight from noisy source text.
- **L4 — Character hallucination / drift.** Invents characters absent from the
  film (e.g. `Ed`, `Nate` in a *10 Cloverfield Lane* scene). Inherent to the task
  design: the prompt only provides genre/theme/tone + previous scene — never a
  character list — so a 3B model cannot keep the cast consistent.
- **L5 — Instruction↔data mismatch.** The instruction says "write the next
  **scene**," but ~40% of training outputs are *paragraph* chunks (from
  `long_entries.py` chunking), so the model often produces short fragments rather
  than full scenes.
- **L6 — No held-out evaluation during training.** A val set exists but no eval
  loss was logged (`best_metric: null`); `load_best_model_at_end` was inert.
- **L7 — Stop token is plain text, not a registered special token**
  (`<\|end_of_scene\|>` is tokenized into subword pieces; `embed_tokens`/`lm_head`
  were not trained). Works, but inference must stop on the literal **string**, not
  an EOS id.

**Highest-leverage fix:** a corpus-cleaning pass (strip `\f`, standalone page
numbers, fix `0.S.`→`O.S.`-style OCR) + retrain. The training recipe itself is
fine — quality is capped by dirty data, not by hyperparameters.

---

## 2. Adapter output is not trailer-ready on its own

Raw adapter scenes are fragmentary, noisy (L1–L3), and character-inconsistent
(L4). They cannot be fed directly to a video model as prompts. **Decision:** use
the adapter only as a *seed/draft*, then refine into a structured trailer script
with `gemini-3.5-flash` before any video generation (see the `scripts/trailer/`
pipeline).

---

## 3. Video generation constraints (Veo 3.1)

- **V1 — 8s max per clip.** A trailer needs more, so we **chain** ~8 segments
  (≈64s). Mechanisms available: native video extension (`video=prev_clip`),
  first/last-frame interpolation (`image` + `config.last_frame`), and reference
  images (≤3) for cross-cut character consistency.
- **V2 — Extension input must be 720p, 16:9 or 9:16, ≤141s total.** Keep working
  clips at 720p when chaining; upscale only at the end if needed.
- **V3 — Character/style consistency across hard cuts is not automatic.** A
  trailer is a montage with cuts, not one continuous shot — so we carry a reusable
  "anchor" reference image (and/or last-frame seeding) across segments to keep the
  protagonist and look consistent.
- **V4 — Cost & latency.** Each 8s clip is a long-running operation (tens of
  seconds to minutes) and is billed per generation; an 8-segment trailer is 8+
  generations. Not free, not instant — batch/poll accordingly.

---

## 4. Environment / ops

- **E1 — Local box has no GPU and no torch/peft.** Adapter scene generation
  (step 1) must run on a GPU (Colab/endpoint). The Gemini + Veo steps are pure API
  calls and run anywhere.
- **E2 — Base model is gated.** `meta-llama/Llama-3.2-3B-Instruct` requires an
  accepted license on the HF account whose token is used.
