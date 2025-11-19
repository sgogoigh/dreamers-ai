import os, json, re, argparse, difflib
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl", default="generation_chunked.jsonl")
parser.add_argument("--rawdir", default="raw_texts")
parser.add_argument("--out", default="generation_chunked_with_source.jsonl")
parser.add_argument("--substr_len", type=int, default=120, help="length of substring to match from output")
parser.add_argument("--fuzzy_cutoff", type=float, default=0.75)
args = parser.parse_args()

jsonl_path = Path(args.jsonl)
rawdir = Path(args.rawdir)
out_path = Path(args.out)

# Build a simple map from file -> content (lowercase) to speed substring search
print("Loading raw texts into memory (for substring search). If large, this may take a while...")
file_texts = {}
for p in rawdir.glob("*.txt"):
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = p.read_text(encoding="latin-1", errors="replace")
    file_texts[p.name] = text.lower()

print(f"Loaded {len(file_texts)} raw files.")

def find_file_by_substring(substr):
    s = substr.lower()
    # quick exact substring search
    for fname, text in file_texts.items():
        if s in text:
            return fname
    # fallback fuzzy: check filenames by similarity to substring (weak)
    names = list(file_texts.keys())
    matches = difflib.get_close_matches(substr[:50], names, n=1, cutoff=args.fuzzy_cutoff)
    if matches:
        return matches[0]
    return None

def inject_source(obj, fname):
    # add top-level source_file
    obj["source_file"] = fname
    # also add movie_title inside input JSON if input is JSON
    try:
        parsed = json.loads(obj.get("input",""))
        if isinstance(parsed, dict):
            # only set movie_title if not present
            if not parsed.get("movie_title"):
                # derive a title from filename (strip extension)
                parsed["movie_title"] = os.path.splitext(fname)[0]
                obj["input"] = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        # input isn't JSON — we can prefix it as fallback
        obj["input"] = f'{{"movie_title":"{os.path.splitext(fname)[0]}", "previous_scene":""}}' + obj.get("input","")
    return obj

with jsonl_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    not_found = 0
    found = 0
    for i, line in enumerate(fin, 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        outtext = obj.get("output","")
        if not outtext.strip():
            not_found += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            continue
        # pick a substring to search (the first substantial chunk)
        cand = outtext.strip()[:args.substr_len]
        fname = find_file_by_substring(cand)
        if fname:
            obj = inject_source(obj, fname)
            found += 1
        else:
            not_found += 1
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if i % 5000 == 0:
            print(f"Processed {i:,} lines. Found: {found}, Not found: {not_found}")

print("Done. Total:", i, "Found:", found, "Not found:", not_found)
print("Output written to:", out_path)