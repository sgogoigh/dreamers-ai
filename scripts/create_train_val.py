import json
import random
from pathlib import Path
from collections import Counter

# Config
IN = "generation_cleaned_per_movie.jsonl"
TRAIN = "train.jsonl"
VAL = "val.jsonl"
STOP = "\n<|end_of_scene|>"
VAL_PCT = 0.05
SEED = 42  # reproducible shuffle

in_path = Path(IN)
if not in_path.exists():
    raise SystemExit(f"{IN} not found — run dedupe_and_cap_per_movie.py first.")

# Count total lines and collect offsets (low-memory: store line offsets)
print("Counting lines and sampling keys...")
total = 0
sample_missing_meta = 0
meta_counter = Counter()

with in_path.open("r", encoding="utf-8") as fh:
    for line in fh:
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        # try to probe movie metadata presence
        found_key = None
        if obj.get("source_file"):
            found_key = obj.get("source_file")
        else:
            try:
                parsed = json.loads(obj.get("input",""))
                found_key = parsed.get("movie_title") or parsed.get("movie_details",{}).get("title")
            except Exception:
                found_key = None
        if found_key:
            meta_counter.update([found_key])
        else:
            sample_missing_meta += 1

if total == 0:
    raise SystemExit("Input file empty or unreadable.")

print(f"Total lines: {total}")
print(f"Sample missing per-example movie metadata: {sample_missing_meta} (expected small)")
print("Top movie keys (sample):", meta_counter.most_common(10))

# Build deterministic shuffled indices
indices = list(range(total))
random.Random(SEED).shuffle(indices)

v = max(1, int(total * VAL_PCT))
val_idx = set(indices[:v])
# Write files streaming, using indices to decide train/val
print(f"Writing train/val split (val {v} lines, seed={SEED})...")

written_train = written_val = 0
with in_path.open("r", encoding="utf-8") as fin, \
     open(TRAIN, "w", encoding="utf-8") as ftrain, \
     open(VAL, "w", encoding="utf-8") as fval:
    for i, line in enumerate(fin):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        # append stop token to output (idempotent)
        obj["output"] = obj.get("output","").rstrip() + STOP
        out_line = json.dumps(obj, ensure_ascii=False)
        if i in val_idx:
            fval.write(out_line + "\n")
            written_val += 1
        else:
            ftrain.write(out_line + "\n")
            written_train += 1

print("Done.")
print(f"Wrote train: {written_train}, val: {written_val} (total: {written_train + written_val})")