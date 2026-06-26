import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import csv
import time

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except TypeError:
    print("🔴 Error: GEMINI_API_KEY not found. Please check your .env file.")
    exit()

model = genai.GenerativeModel('gemini-2.0-flash-lite')
dir_path = "raw_texts"
output_csv_file = "dataset.csv"

processed_titles = set()
try:
    with open(output_csv_file, 'r', newline='', encoding='utf-8') as f_read:
        reader = csv.reader(f_read)
        header = next(reader)
        for row in reader:
            if row:
                processed_titles.add(row[0])
    print(f"✅ Found {len(processed_titles)} movies already in '{output_csv_file}'. Will skip them.")
except FileNotFoundError:
    print(f"📄 '{output_csv_file}' not found. A new file will be created.")


all_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
with open(output_csv_file, 'a', newline='', encoding='utf-8') as f_write:
    csv_writer = csv.writer(f_write)
    if not processed_titles:
        header = ['MovieTitle', 'Genre', 'Theme', 'Tone', 'Length (min)']
        csv_writer.writerow(header)

    print(f"Found {len(all_files)} total scripts. Starting analysis...")
    for index, file_name in enumerate(all_files):
        movie_title = os.path.splitext(file_name)[0].split("_")[0]

        if movie_title in processed_titles:
            continue

        print(f"  ({index + 1}/{len(all_files)}) Querying for: {movie_title}")
        
        prompt = f"""
        Act as a movie database. For the movie titled "{movie_title}", provide the following information in a single, clean JSON object with these exact keys: "genre", "theme", "tone", and "runtime_minutes".
        - genre: The primary genre.
        - theme: A brief description of the central theme.
        - tone: The overall mood or feeling.
        - runtime_minutes: The official runtime in minutes as an integer.
        Do not include any text, notes, or markdown formatting before or after the JSON object.
        """

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip().replace('```json', '').replace('```', '')
            movie_info = json.loads(response_text)

            # Flexibly extract the length
            length = 0
            for key in ['runtime_minutes', 'length_minutes', 'runtime']:
                if key in movie_info:
                    length = int(movie_info.get(key, 0))
                    break

            # Prepare the data as a list for the CSV row
            new_row = [
                movie_title,
                movie_info.get('genre', 'N/A'),
                movie_info.get('theme', 'N/A'),
                movie_info.get('tone', 'N/A'),
                length
            ]

            # --- KEY CHANGE: Write the new row to the file immediately ---
            csv_writer.writerow(new_row)
            
            # Add the title to our set so we don't re-process it in this run
            processed_titles.add(movie_title)

        except Exception as e:
            print(f"    - 🔴 Could not process {movie_title}. Error: {e}. Skipping.")

        # handles gemini 2.0 flash lite rate limits - 30 RPM, 200 RPD
        time.sleep(1)

print("\n✅ Analysis complete!")