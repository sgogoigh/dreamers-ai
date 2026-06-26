#!/usr/bin/env python3
"""
find_and_chunk_long_entries.py

1) Finds the top-K longest entries in a JSONL by tokenizer token count and prints them.
2) For entries whose (input+output) token length exceeds --max_tokens:
     - If the output contains paragraph breaks, chunk the output into paragraphs and emit
       multiple examples (optionally making previous_scene be the previous paragraph).
     - Otherwise, fall back to token-aware truncation of the output.
3) Writes a new JSONL with chunked/truncated entries (and untouched short entries).

Usage:
    python find_and_chunk_long_entries.py \
      --in finetuning_generation_dataset.jsonl \
      --out generation_chunked.jsonl \
      --model NousResearch/Meta-Llama-3-8B-Instruct \
      --max_tokens 3500 \
      --topk 20

Requirements:
    transformers (tokenizer only), tqdm (optional but helpful)

The script is robust to malformed JSON lines (skips them with a warning).
"""
import argparse
import json
import heapq
import re
from pathlib import Path
from typing import List
import sys

try:
    from transformers import AutoTokenizer
except Exception as e:
    print("ERROR: transformers library required. Install with `pip install transformers`")
    raise

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback no-progress

# ---------- helpers ----------
PARA_SPLIT_RE = re.compile(r'\n\s*\n+')  # paragraph splitter

def paragraphs(text: str, min_chars: int = 80) -> List[str]:
    paras = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    if min_chars:
        return [p for p in paras if len(p) >= min_chars]
    return paras

def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def token_length_of(tokenizer, text: str) -> int:
    # tokenizer(..., truncation=False) may raise on very long input for some fast tokenizers;
    # use try/except and fallback to a rough estimate by len(text)/4 if it fails.
    try:
        return len(tokenizer(text, truncation=False)["input_ids"])
    except Exception:
        # rough heuristic fallback if the tokenizer chokes
        return max(1, len(text) // 4)

def truncate_output_tokenwise(tokenizer, input_text: str, output_text: str, max_tokens: int) -> str:
    # Preserve as much of output_text as possible while keeping input+output <= max_tokens
    input_ids = tokenizer(input_text, truncation=False)["input_ids"]
    input_len = len(input_ids)
    allowed_for_output = max_tokens - input_len - 1
    if allowed_for_output <= 20:  # too small to be useful
        # fallback: return a short prefix of the output in characters
        return output_text[:1000]
    out_ids = tokenizer(output_text, truncation=False)["input_ids"]
    if len(out_ids) <= allowed_for_output:
        return output_text
    truncated_out_ids = out_ids[:allowed_for_output]
    try:
        truncated_out = tokenizer.decode(truncated_out_ids, skip_special_tokens=True)
    except Exception:
        # final fallback: character truncation
        return output_text[:max(1000, allowed_for_output * 2)]
    return truncated_out

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="Input JSONL path")
    p.add_argument("--out", dest="outfile", required=True, help="Output JSONL path (chunked/truncated)")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct", help="HF tokenizer/model id (tokenizer only)")
    p.add_argument("--max_tokens", type=int, default=3500, help="Max tokens allowed for input+output before chunking/truncation")
    p.add_argument("--topk", type=int, default=20, help="Show top-K longest entries before processing")
    p.add_argument("--min_para_chars", type=int, default=80, help="Minimum paragraph length to consider when chunking")
    p.add_argument("--make_prev_para", action="store_true", help="When chunking, set previous_scene to prior paragraph (recommended).")
    args = p.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        print("Input file not found:", infile)
        sys.exit(1)

    print("Loading tokenizer:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # First pass: find top-K longest entries (by token length) using a min-heap
    topk = []
    print("Scanning token lengths (this may take a while)...")
    total = 0
    with infile.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            total += 1
            line = line.rstrip("\n")
            if not line:
                continue
            obj = safe_json_load(line)
            if obj is None:
                continue
            inp = obj.get("input", "") or ""
            out = obj.get("output", "") or ""
            combined_text = inp + "\n" + out
            tlen = token_length_of(tokenizer, combined_text)
            # maintain min-heap of size topk
            if len(topk) < args.topk:
                heapq.heappush(topk, (tlen, total, line))
            else:
                if tlen > topk[0][0]:
                    heapq.heapreplace(topk, (tlen, total, line))

    # Report top-K (longest first)
    topk_sorted = sorted(topk, reverse=True)
    print(f"\nTop {args.topk} longest entries (token counts):")
    for tlen, idx, line in topk_sorted:
        obj = safe_json_load(line)
        if obj is None:
            print(f"Index {idx}: (malformed json) tokens={tlen}")
            continue
        inp = obj.get("input","")
        try:
            parsed_inp = json.loads(inp)
            md = parsed_inp.get("movie_details", {}) if isinstance(parsed_inp, dict) else {}
            movie_hint = md.get("title") or md.get("movie") or md.get("genre") or "UNKNOWN"
        except Exception:
            movie_hint = "UNKNOWN"
        out_preview = (obj.get("output","")[:400].replace("\n","\\n") + "...")
        print(f"Index {idx}: tokens={tlen}  movie_hint={movie_hint}\n  input_preview: {inp[:200].replace(chr(10),'\\n')}...\n  output_preview: {out_preview}\n")

    # Second pass: write chunked/truncated file
    print(f"\nProcessing and writing to {outfile} ... (entries with tokens > {args.max_tokens} will be chunked/truncated)")
    written = 0
    chunked_count = 0
    truncated_count = 0
    skipped_bad = 0
    processed = 0

    with infile.open("r", encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            processed += 1
            line = line.rstrip("\n")
            if not line:
                continue
            obj = safe_json_load(line)
            if obj is None:
                skipped_bad += 1
                continue

            inp = obj.get("input","") or ""
            out = obj.get("output","") or ""
            combined_text = inp + "\n" + out
            tlen = token_length_of(tokenizer, combined_text)

            if tlen <= args.max_tokens:
                # write unchanged
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                continue

            # tlen > max_tokens -> try paragraph chunking
            paras = paragraphs(out, min_chars=args.min_para_chars)
            if len(paras) >= 2:
                # create sequential chunks; for each chunk make a new example.
                # Option A: keep input as-is but set previous_scene optionally
                for i, para in enumerate(paras):
                    new_obj = {
                        "instruction": obj.get("instruction"),
                        "input": obj.get("input"),
                        "output": para
                    }
                    if args.make_prev_para:
                        # set previous_scene inside the input JSON if possible
                        try:
                            parsed_inp = json.loads(new_obj["input"])
                            # store previous scene as a string field
                            if i == 0:
                                parsed_inp["previous_scene"] = ""
                            else:
                                parsed_inp["previous_scene"] = paras[i-1]
                            new_obj["input"] = json.dumps(parsed_inp, ensure_ascii=False)
                        except Exception:
                            # if input isn't JSON, prepend a small textual prefix (fallback)
                            pref = f"Previous paragraph:\n{paras[i-1] if i>0 else ''}\n\n"
                            new_obj["input"] = pref + new_obj["input"]
                    fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                    written += 1
                chunked_count += 1
                continue

            # otherwise fallback to token-aware truncation
            truncated_out = truncate_output_tokenwise(tokenizer, inp, out, args.max_tokens)
            new_obj = dict(obj)
            new_obj["output"] = truncated_out
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            written += 1
            truncated_count += 1

    print("\nDONE.")
    print(f"Processed lines: {processed}")
    print(f"Written lines: {written}")
    print(f"Chunked entries (paragraph-split): {chunked_count}")
    print(f"Truncated entries (token-truncate fallback): {truncated_count}")
    print(f"Malformed/skipped lines: {skipped_bad}")
    print(f"Output saved to: {outfile}")

if __name__ == "__main__":
    main()