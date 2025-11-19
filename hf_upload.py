from huggingface_hub import HfApi
import os

from dotenv import load_dotenv
load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="dataset",
    repo_id="sgogoi/movie-scripts",
    repo_type="dataset",
)