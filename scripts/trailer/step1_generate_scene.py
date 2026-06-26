"""Step 1 — draft raw scene(s) from the fine-tuned LoRA adapter.

⚠️  Requires a GPU + torch/peft/bitsandbytes and an HF token whose account has
accepted the gated Llama-3.2-3B-Instruct license. On a CPU-only box this won't
run — generate the draft in Colab (see scripts/evaluate_adapter.ipynb) and feed
the text into step 2 via `run_pipeline.py --raw-scene-file`.

The adapter output is intentionally treated as a *rough draft*: it is noisy and
fragmentary (see LIMITATIONS.md). Step 2 cleans and restructures it.
"""
from __future__ import annotations

import json
from typing import Optional

from .config import ADAPTER_REPO, BASE_MODEL, STOP_STR

ALPACA_TMPL = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
INSTRUCTION = (
    "You are a screenwriter. Given the details of a movie and the content of the "
    "preceding scene, write the next scene for the script."
)


def _lazy_imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel
    return (torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
            StoppingCriteria, StoppingCriteriaList, PeftModel)


def load_model(load_4bit: bool = True):
    (torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
     _, _, PeftModel) = _lazy_imports()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()
    return model, tok


def generate_raw_scene(
    concept: str,
    genre: str = "",
    theme: str = "",
    tone: str = "",
    previous_scene: str = "",
    model=None,
    tok=None,
    max_new_tokens: int = 400,
) -> str:
    """Generate one draft scene. `concept` seeds `previous_scene` if none given."""
    (torch, _, _, _, StoppingCriteria, StoppingCriteriaList, _) = _lazy_imports()
    if model is None or tok is None:
        model, tok = load_model()

    input_obj = {
        "movie_details": {"genre": genre, "theme": theme or concept, "tone": tone},
        "previous_scene": previous_scene or concept,
    }
    prompt = ALPACA_TMPL.format(instruction=INSTRUCTION, input=json.dumps(input_obj, ensure_ascii=False))
    enc = tok(prompt, return_tensors="pt").to(model.device)
    plen = enc["input_ids"].shape[1]

    class StopOnString(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return STOP_STR in tok.decode(input_ids[0][plen:], skip_special_tokens=True)

    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.8, top_p=0.95, repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnString()]),
        )
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    return text.replace(STOP_STR, "").strip()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument("--genre", default="")
    p.add_argument("--theme", default="")
    p.add_argument("--tone", default="")
    p.add_argument("--out", default="raw_scene.txt")
    a = p.parse_args()
    scene = generate_raw_scene(a.concept, a.genre, a.theme, a.tone)
    print(scene)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(scene)
    print(f"\nSaved to {a.out}")
