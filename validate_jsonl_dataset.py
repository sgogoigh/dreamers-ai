import json, statistics, collections, sys
from pathlib import Path

JSONL = "finetuning_generation_dataset.jsonl"  # change if different

if not Path(JSONL).exists():
    print("File not found:", JSONL); sys.exit(1)

count = 0
lens_chars_in = []
lens_chars_out = []
missing_keys = 0
empty_outputs = 0
duplicates = set()
dupe_count = 0
genre_counts = collections.Counter()
sample_examples = []

with open(JSONL, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print("Malformed JSON line (skipping):", line[:120])
            continue
        count += 1
        instr = obj.get("instruction")
        inp = obj.get("input")
        out = obj.get("output")
        if instr is None or inp is None or out is None:
            missing_keys += 1
        if not (out and out.strip()):
            empty_outputs += 1
        # character lengths
        lens_chars_in.append(len(inp or ""))
        lens_chars_out.append(len(out or ""))
        # detect duplicates by hashing a short signature
        sig = ( (inp or "")[:200] + "||" + (out or "")[:200] )
        if sig in duplicates:
            dupe_count += 1
        else:
            duplicates.add(sig)
        # try to read genre from input JSON string if present
        try:
            parsed_inp = json.loads(inp)
            md = parsed_inp.get("movie_details") if isinstance(parsed_inp, dict) else None
            if md and isinstance(md, dict):
                genre = md.get("genre") or "UNKNOWN"
                genre_counts[genre] += 1
        except Exception:
            pass
        # store some random samples
        if len(sample_examples) < 5:
            sample_examples.append(obj)

print("TOTAL LINES:", count)
print("Missing keys:", missing_keys)
print("Empty outputs:", empty_outputs)
print("Duplicate (approx) signatures:", dupe_count)
if lens_chars_in:
    print("Input chars: median", statistics.median(lens_chars_in), "mean", round(statistics.mean(lens_chars_in)))
    print("Output chars: median", statistics.median(lens_chars_out), "mean", round(statistics.mean(lens_chars_out)))
print("Unique genres (sample):", list(genre_counts.items())[:10])
print("\nSAMPLES (first 5 lines):")
for s in sample_examples:
    print("-", (s.get("input")[:180].replace("\n","\\n") + "..."))
    print("  -> output preview:", (s.get("output")[:180].replace("\n","\\n") + "..."))