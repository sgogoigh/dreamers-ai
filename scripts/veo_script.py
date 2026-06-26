import time
from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
load_dotenv()

print("Beginning the script ...")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """
A close up of two people staring at a cryptic drawing on a wall, torchlight flickering.
A man murmurs, 'This must be it. That's the secret code.'
The woman leans in and whispers excitedly, 'What did you find?'
"""

print("Loading model ...")
operation = client.models.generate_videos(
    model="veo-3.1-generate-preview",  # MAIN MODEL
    prompt=prompt,
    # config=types.GenerateVideosConfig(
    #     resolution="720p",
    #     aspectRatio="16:9"
    # )
)

# Poll until finished
while not operation.done:
    print("Waiting for video generation to complete...")
    time.sleep(10)
    operation = client.operations.get(operation)

# Get the generated video file reference
generated_video = operation.response.generated_videos[0]

# Download to local file
client.files.download(
    file=generated_video.video
)
generated_video.video.save("dialogue_example.mp4")
print("Generated video saved to dialogue_example.mp4")