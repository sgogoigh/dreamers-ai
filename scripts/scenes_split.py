import pandas as pd
import os
import json
import re
import difflib
from pathlib import Path

def read_file_with_fallback(path):
    encodings = ("utf-8", "cp1252", "latin-1")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            # try next encoding
            continue
        except Exception as e:
            raise
    # Last resort: read as utf-8 but replace invalid bytes so we never crash
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()

def normalize_title(s: str):
    s = s.lower().strip()
    # remove common punctuation
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def find_best_title_match(filename_title, title_keys):
    """
    Try exact match, then normalized match, then fuzzy match (difflib).
    Returns matched key or None.
    """
    if filename_title in title_keys:
        return filename_title
    # normalized exact match
    norm_filename = normalize_title(filename_title)
    for k in title_keys:
        if normalize_title(k) == norm_filename:
            return k
    # fuzzy
    candidates = difflib.get_close_matches(filename_title, title_keys, n=1, cutoff=0.7)
    if candidates:
        return candidates[0]
    # try after removing trailing numbers or extra tokens (common filename patterns)
    alt = re.split(r"[_\-]", filename_title)[0].strip()
    if alt in title_keys:
        return alt
    # last attempt: try normalized against all keys and return first similar
    candidates = difflib.get_close_matches(norm_filename, [normalize_title(k) for k in title_keys], n=1, cutoff=0.7)
    if candidates:
        # map back to original key
        for k in title_keys:
            if normalize_title(k) == candidates[0]:
                return k
    return None

def split_script_into_scenes(script_content):
    pattern = re.compile(
        r'(?=(?:^|\n)(?:INT(?:/EXT)?\.?|EXT(?:/INT)?\.?|INT -|EXT -))',
        flags=re.IGNORECASE | re.MULTILINE
    )
    scenes = re.split(pattern, script_content)
    cleaned = [s.strip() for s in scenes if s and s.strip()]
    return cleaned

# Load movie details 
try:
    movies_df = pd.read_csv("dataset.csv")
    movie_details_dict = movies_df.set_index('MovieTitle').to_dict('index')
except FileNotFoundError:
    print("🔴 Error: 'dataset.csv' not found in the current directory.")
    print("   Please make sure the file with movie details is named correctly and is in the same folder.")
    raise SystemExit(1)

# Config / paths 
dir_path = "raw_texts"
output_file = "finetuning_generation_dataset.jsonl"

if not os.path.isdir(dir_path):
    print(f"🔴 Error: directory '{dir_path}' not found.")
    raise SystemExit(1)

file_names = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

# Process files 
with open(output_file, 'w', encoding='utf-8') as f_out:
    print(f"Processing {len(file_names)} scripts to create {output_file}...")
    skipped_no_details = 0
    skipped_small = 0
    total_pairs = 0

    title_keys = list(movie_details_dict.keys())

    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)
        # derive a likely movie title from filename (without extension)
        raw_title = Path(file_name).stem  # filename without extension
        # Many filenames contain extra metadata; try some heuristics
        # prefer full stem, then part before underscore or dash
        candidate_title = raw_title
        # Try splitting on underscore/dash if the exact stem not found
        candidate_title = re.split(r"[_\-]", candidate_title)[0].strip()

        match_title = find_best_title_match(candidate_title, title_keys)
        if not match_title:
            # give one more attempt using the whole stem
            match_title = find_best_title_match(Path(file_name).stem, title_keys)

        if not match_title:
            print(f"  - Warning: No details found for '{candidate_title}' (file: {file_name}) in the CSV. Skipping.")
            skipped_no_details += 1
            continue

        # robust read
        try:
            script_content = read_file_with_fallback(file_path)
        except Exception as e:
            print(f"  - Error reading '{file_name}': {e}. Skipping.")
            continue

        scenes = split_script_into_scenes(script_content)

        if len(scenes) < 2:
            print(f"  - Warning: Could not split '{match_title}' (file: {file_name}) into enough scenes. Skipping.")
            skipped_small += 1
            continue

        details = movie_details_dict[match_title]
        movie_details_for_input = {
            "genre": details.get('Genre', 'N/A'),
            "theme": details.get('Theme', 'N/A'),
            "tone": details.get('Tone', 'N/A')
        }

        # create previous->current pairs
        for i in range(1, len(scenes)):
            previous_scene = scenes[i-1]
            current_scene_as_output = scenes[i]

            input_data = {
                "movie_details": movie_details_for_input,
                "previous_scene": previous_scene
            }

            training_example = {
                "instruction": "You are a screenwriter. Given the details of a movie and the content of the preceding scene, write the next scene for the script.",
                "input": json.dumps(input_data, ensure_ascii=False),
                "output": current_scene_as_output
            }

            f_out.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            total_pairs += 1

    print("\nSummary:")
    print(f"  - Files processed: {len(file_names)}")
    print(f"  - Training pairs written: {total_pairs}")
    print(f"  - Skipped (no CSV details): {skipped_no_details}")
    print(f"  - Skipped (couldn't split into scenes): {skipped_small}")

print(f"\n✅ Generation dataset creation complete! Saved to {output_file}")
