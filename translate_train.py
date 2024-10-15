import json
from datasets import load_dataset
import uuid

# Load the dataset
dataset = load_dataset("jhu-clsp/FollowIR-train", split="train")

# Target languages
target_languages = ["Farsi", "Russian", "Chinese"]

def prepare_translation_request(text, field_name, target_language, followir_id):
    uuid_value = uuid.uuid4().hex
    custom_id = f"{uuid_value}-{target_language.lower()}-{followir_id}-{field_name}"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {"role": "system", "content": f"Translate the following {field_name} from English to {target_language}. Output the translated text only."},
                {"role": "user", "content": text}
            ],
            "max_tokens": 5000
        }
    }

def process_dataset(dataset, target_language):
    batch_requests = []
    for item in dataset:
        followir_id = item['id']
        for field in ['document', 'query', 'instruction']:
            if item[field]:  # Check if the field is not empty
                request = prepare_translation_request(item[field], field, target_language, followir_id)
                batch_requests.append(request)
    return batch_requests

# Process the dataset for each target language
for language in target_languages:
    batch_requests = process_dataset(dataset, language)
    
    # Save batch requests to a JSONL file
    output_filename = f"batch_requests_{language.lower()}.jsonl"
    with open(output_filename, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
    
    print(f"Saved {len(batch_requests)} translation requests for {language} to {output_filename}")

print("Batch request files created for all target languages.")