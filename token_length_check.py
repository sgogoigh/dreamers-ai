from transformers import AutoTokenizer
import json, statistics, sys
from pathlib import Path

JSONL = "finetuning_generation_dataset.jsonl"
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"   # change if different

if not Path(JSONL).exists():
    print("File not found:", JSONL); sys.exit(1)

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
lengths = []
count = 0
with open(JSONL, "r", encoding="utf-8") as fh:
    for line in fh:
        obj = json.loads(line)
        txt = obj["input"] + "\n" + obj["output"]
        toks = tok(txt, truncation=False)["input_ids"]
        lengths.append(len(toks))
        count += 1
        if count % 5000 == 0:
            print(f"Processed {count} lines...")

lengths_sorted = sorted(lengths)
p90 = lengths_sorted[int(0.90 * len(lengths_sorted))]
p95 = lengths_sorted[int(0.95 * len(lengths_sorted))]
print(f"N samples: {len(lengths)}")
print("Median tokens:", int(statistics.median(lengths)))
print("Mean tokens:", int(statistics.mean(lengths)))
print("90th pct:", p90, "95th pct:", p95, "max:", max(lengths))