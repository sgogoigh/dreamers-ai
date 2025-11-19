# dedupe_and_cap_chunked.py
import json, hashlib, collections
from pathlib import Path

IN = "generation_chunked.jsonl"
OUT = "generation_cleaned.jsonl"
CAP = 1000   # change to 500 if you want stronger capping

def sig(inp,out):
    return hashlib.sha256((inp[:300]+"||"+out[:300]).encode("utf-8")).hexdigest()

counts = collections.Counter()
seen = set()

with open(IN,"r",encoding="utf-8") as fh, open(OUT,"w",encoding="utf-8") as fo:
    for line in fh:
        obj = json.loads(line)
        inp = obj.get("input","")
        out = obj.get("output","")
        # try to get movie key from input JSON if present
        movie = "UNKNOWN"
        try:
            md = json.loads(inp).get("movie_details", {})
            # prefer an explicit title if present, else genre as fallback
            movie = md.get("title") or md.get("movie") or md.get("genre") or "UNKNOWN"
        except Exception:
            pass
        h = sig(inp,out)
        if h in seen:
            continue
        if counts[movie] >= CAP:
            continue
        seen.add(h)
        counts[movie] += 1
        fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Wrote cleaned file:", OUT)
print("Per-movie cap:", CAP)
print("Top movie counts (sample):", counts.most_common(10))