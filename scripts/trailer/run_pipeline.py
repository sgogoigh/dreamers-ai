"""End-to-end trailer pipeline orchestrator.

    concept (+ optional draft scene)
      -> [step 2] gemini-3.5-flash  : structured trailer script
      -> [step 3] gemini-3.5-flash  : chained Veo segment prompts
      -> [step 4] veo-3.1           : chained 8s clips -> stitched ~1min trailer

Step 1 (adapter draft) is optional and GPU-only. On a CPU box, generate the
draft separately (Colab / scripts/evaluate_adapter.ipynb) and pass it with
--raw-scene-file, or omit it entirely and let Gemini write from the concept.

Examples:
    python -m scripts.trailer.run_pipeline --concept "A lighthouse keeper discovers
        the fog is alive" --genre "Psychological Horror" --tone "dread, isolation"

    python -m scripts.trailer.run_pipeline --concept "..." --raw-scene-file draft.txt
        --segments 8 --native-extend
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import N_SEGMENTS, SEG_SECONDS
from .step2_refine_script import refine_to_trailer_script
from .step3_build_prompts import build_segment_prompts
from .step4_generate_video import generate_trailer


def main():
    p = argparse.ArgumentParser(description="Dreamers-AI trailer pipeline")
    p.add_argument("--concept", required=True, help="one-line movie concept")
    p.add_argument("--genre", default="")
    p.add_argument("--tone", default="")
    p.add_argument("--raw-scene-file", help="optional adapter draft scene (text)")
    p.add_argument("--gen-draft", action="store_true",
                   help="generate the draft with the LoRA adapter (GPU required)")
    p.add_argument("--segments", type=int, default=N_SEGMENTS)
    p.add_argument("--seconds", type=int, default=SEG_SECONDS)
    p.add_argument("--out-dir", default="trailer_out")
    p.add_argument("--native-extend", action="store_true")
    p.add_argument("--no-anchor", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="run steps 2-3 only (no video, no cost); writes JSON artifacts")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- draft (optional) --------------------------------------------------
    draft = ""
    if a.raw_scene_file:
        draft = Path(a.raw_scene_file).read_text(encoding="utf-8")
    elif a.gen_draft:
        from .step1_generate_scene import generate_raw_scene
        print("Step 1: generating draft scene with the LoRA adapter (GPU)…")
        draft = generate_raw_scene(a.concept, a.genre, theme=a.concept, tone=a.tone)
        (out / "raw_scene.txt").write_text(draft, encoding="utf-8")

    # --- step 2: trailer script -------------------------------------------
    print("Step 2: refining into a structured trailer script (gemini-3.5-flash)…")
    script = refine_to_trailer_script(a.concept, draft, a.genre, a.tone, n_segments=a.segments)
    (out / "trailer_script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    print(f"  -> {script.title}: {len(script.beats)} beats")

    # --- step 3: segment prompts ------------------------------------------
    print("Step 3: building chained Veo 3.1 segment prompts (gemini-3.5-flash)…")
    prompts = build_segment_prompts(script, seg_seconds=a.seconds)
    (out / "trailer_prompts.json").write_text(prompts.model_dump_json(indent=2), encoding="utf-8")
    print(f"  -> {len(prompts.segments)} segment prompts")

    if a.dry_run:
        print("\nDry run complete. Inspect trailer_script.json / trailer_prompts.json.")
        return

    # --- step 4: video -----------------------------------------------------
    print("Step 4: generating & chaining clips with Veo 3.1…")
    final = generate_trailer(
        prompts, out_dir=a.out_dir, native_extend=a.native_extend,
        use_anchor=not a.no_anchor, seg_seconds=a.seconds,
    )
    print(f"\nDone: {final}")


if __name__ == "__main__":
    main()
