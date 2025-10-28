import pandas as pd
import os
import json
import re

try:
    movies_df = pd.read_csv("dataset.csv") 
    movie_details_dict = movies_df.set_index('MovieTitle').to_dict('index')
except FileNotFoundError:
    print("🔴 Error: 'dataset.csv' not found in the current directory.")
    print("   Please make sure the file with movie details is named correctly and is in the same folder.")
    exit()

def split_script_into_scenes(script_content):
    # Split the text at each scene heading, but keep the heading with the scene.
    scenes = re.split(r'(?=^INT\.|^EXT\.)', script_content, flags=re.MULTILINE)
    # Filter out any empty strings that might result from the split
    return [scene.strip() for scene in scenes if scene.strip()]

dir_path = "raw_texts"
output_file = "finetuning_generation_dataset.jsonl"
file_names = [f for f in os.listdir(dir_path) if f.endswith('.txt')]


with open(output_file, 'w', encoding='utf-8') as f_out:
    print(f"Processing {len(file_names)} scripts to create {output_file}...")
    
    for file_name in file_names:
        # Clean the movie title from the filename to match the CSV
        movie_title = os.path.splitext(file_name)[0].split("_")[0]

        if movie_title in movie_details_dict:
            script_path = os.path.join(dir_path, file_name)
            with open(script_path, 'r', encoding='utf-8') as f_in:
                script_content = f_in.read()
            
            scenes = split_script_into_scenes(script_content)
            
            if len(scenes) < 2:
                print(f"  - Warning: Could not split '{movie_title}' into enough scenes. Skipping.")
                continue

            details = movie_details_dict[movie_title]
            movie_details_for_input = {
                "genre": details.get('Genre', 'N/A'),
                "theme": details.get('Theme', 'N/A'),
                "tone": details.get('Tone', 'N/A')
            }

            # Loop through scenes to create pairs (previous_scene -> current_scene)
            for i in range(1, len(scenes)):
                previous_scene = scenes[i-1]
                current_scene_as_output = scenes[i]
                
                # Format the input for the model
                input_data = {
                    "movie_details": movie_details_for_input,
                    "previous_scene": previous_scene
                }
                
                # Create the final training example
                training_example = {
                    "instruction": "You are a screenwriter. Given the details of a movie and the content of the preceding scene, write the next scene for the script.",
                    "input": json.dumps(input_data),
                    "output": current_scene_as_output
                }
                
                # Write the JSON object as a new line in the file
                f_out.write(json.dumps(training_example) + "\n")
        else:
            print(f"  - Warning: No details found for '{movie_title}' in the CSV. Skipping.")

print(f"\n✅ Generation dataset creation complete! Saved to {output_file}")