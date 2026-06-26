"""
Dedupe and Per-Movie Cap

- Prefers movie identifier fields (checked in order):
    1) source_file
    2) movie_title
    3) title
    4) MovieTitle
    5) movie (lowercase)
    6) meta 'movie_details' -> 'title' if present
- Falls back to 'genre' ONLY if none of the above exist.

Outputs: generation_cleaned_per_movie.jsonl
"""
import json
import hashlib
import collections
from pathlib import Path

IN = "generation_chunked_with_source.jsonl"   # ← use your backfilled file
OUT = "generation_cleaned_per_movie.jsonl"
CAP = 1000   # recommended per-movie cap (change if you want stricter/looser)

def sig(inp, out):
    # stable signature of a pair to dedupe exact duplicates
    return hashlib.sha256((inp[:400] + "||" + out[:400]).encode("utf-8")).hexdigest()

def extract_movie_key(obj):
    """
    Try multiple common keys/locations for a per-movie identifier.
    Return a string key (never None) — uses genre as last resort.
    """
    # 1) direct fields on object
    if isinstance(obj, dict):
        for k in ("source_file", "movie_title", "title", "MovieTitle", "movie"):
            if k in obj and obj[k]:
                return str(obj[k]).strip()
        # 2) sometimes input is a JSON string; try parsing it
        inp = obj.get("input")
        if isinstance(inp, str):
            try:
                parsed = json.loads(inp)
                # top-level movie fields in the parsed input
                for k in ("movie_title", "title", "MovieTitle", "movie", "source_file"):
                    if k in parsed and parsed[k]:
                        return str(parsed[k]).strip()
                # check nested movie_details.title or movie_title keys
                md = parsed.get("movie_details") if isinstance(parsed, dict) else None
                if isinstance(md, dict):
                    for k in ("title", "movie_title", "name"):
                        if k in md and md[k]:
                            return str(md[k]).strip()
                    # if there's no title, use a combination of title/genre/tone to at least group
                    genre = md.get("genre")
                    tone = md.get("tone")
                    if genre and tone:
                        return f"__GENRE_{genre}__TONE_{tone}"
                    if genre:
                        return f"__GENRE_{genre}"
            except Exception:
                pass
    # last resort: fallback to 'UNKNOWN' (but will be grouped under this)
    # --- but prefer to return genre if present in movie_details
    try:
        if isinstance(obj, dict):
            inp = obj.get("input","")
            parsed = None
            if isinstance(inp, str):
                parsed = json.loads(inp) if inp.strip().startswith("{") else None
            if parsed and isinstance(parsed, dict):
                md = parsed.get("movie_details", {})
                if isinstance(md, dict) and md.get("genre"):
                    return f"__GENRE_{md.get('genre')}"
    except Exception:
        pass
    return "UNKNOWN_MOVIE"

def main():
    inpath = Path(IN)
    outpath = Path(OUT)
    if not inpath.exists():
        print(f"Input file not found: {inpath}")
        return

    seen = set()
    counts = collections.Counter()
    written = 0
    skipped_dup = 0
    skipped_cap = 0

    with inpath.open("r", encoding="utf-8") as fin, outpath.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed
                continue

            inp = obj.get("input","")
            out = obj.get("output","")
            h = sig(inp, out)
            if h in seen:
                skipped_dup += 1
                continue

            movie_key = extract_movie_key(obj)
            # normalize movie_key for counting
            movie_key = movie_key.strip()
            if counts[movie_key] >= CAP:
                skipped_cap += 1
                continue

            # accept this example
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            seen.add(h)
            counts[movie_key] += 1
            written += 1

    print("Wrote cleaned file:", outpath)
    print("Per-movie cap:", CAP)
    print("Examples written:", written)
    print("Duplicates skipped:", skipped_dup)
    print("Skipped due to cap:", skipped_cap)
    print("Top movie counts (sample):", counts.most_common(20))

if __name__ == '__main__':
    main()