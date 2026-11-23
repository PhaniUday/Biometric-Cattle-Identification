import os
import zipfile
from huggingface_hub import hf_hub_download

# 1. DOWNLOAD PHASE
repo_id = "MingShuo666/CattleMuzzle"
# Use the exact path discovered in the repository
filename = "cattle_muzzle_datasets_/recognition datasets.zip"

try:
    print(f"Connecting to Hugging Face to download: {filename}")
    # This downloads the zip file into your current project folder
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir="."
    )
    print(f"Successfully downloaded to: {zip_path}")

    # 2. EXTRACTION PHASE
    print("Extracting files... This may take a minute.")
    # We extract into a dedicated folder to keep your workspace organized
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("CattleData")
    
    print("Extraction complete! Your data is now in 'CattleData/recognition data'.")

except Exception as e:
    print(f"An error occurred: {e}")