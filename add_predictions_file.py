import json
from collections import defaultdict
from huggingface_hub import hf_hub_download, HfApi
import os
from pathlib import Path
import tempfile

def download_file(repo_id, filename, token):
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token, local_dir=tmpdirname, repo_type="dataset")
        with open(local_path, 'r') as file:
            return file.read()

def generate_empty_predictions(input_content):
    predictions = defaultdict(lambda: defaultdict(float))
    
    for line in input_content.splitlines():
        data = json.loads(line)
        qid = data['qid']
        pid = data['pid']
        predictions[qid][pid] = 0.0
    
    return predictions

def push_to_hub(content, filename, repo_id, token):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo="empty_scores.json",
        repo_id=repo_id,
        token=token,
        repo_type="dataset"
    )

def main():
    languages = ['rus', 'fas', 'zho']
    base_repo_id = "jhu-clsp/mFollowIR-{lang}-cl"
    token = os.environ.get("HF_TOKEN")  # Make sure to set this environment variable

    if not token:
        raise ValueError("Please set the HF_TOKEN environment variable")

    for lang in languages:
        repo_id = base_repo_id.format(lang=lang)
        input_filename = "top_ranked.jsonl"
        output_filename = f'mFollowIR-{lang}-cl_predictions.json'
        
        # Download input file
        print(f"Downloading {input_filename} from {repo_id}")
        input_content = download_file(repo_id, input_filename, token)
        
        # Generate predictions
        print(f"Generating predictions for {lang}")
        predictions = generate_empty_predictions(input_content)
        
        # Prepare output content
        output_content = json.dumps(predictions, indent=2)
        
        # Push the file to Hugging Face
        print(f"Pushing {output_filename} to {repo_id}")
        push_to_hub(output_content, output_filename, repo_id, token)
        
        print(f"Completed processing for {lang}")

if __name__ == "__main__":
    main()