from huggingface_hub import HfApi
api = HfApi()
# This lists every file in the repository so we can see the exact path
files = api.list_repo_files(repo_id="MingShuo666/CattleMuzzle", repo_type="dataset")
for f in files:
    print(f)