import json
import argparse
from datasets import load_dataset

def load_multilingual_data(filename):
    multilingual_data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            custom_id = item['custom_id']
            lang = custom_id.split('-')[1]
            data_type = custom_id.split('-')[-1]
            original_id = "-".join(custom_id.split('-')[1:]).replace(lang + "-", "").replace("-" + data_type, "")
           
            if original_id not in multilingual_data:
                multilingual_data[original_id] = {'lang': lang}
            
            content = item['response']['body']['choices'][0]['message']['content']
            multilingual_data[original_id][data_type] = content
    
    return multilingual_data

def transform_dataset(dataset_name, multilingual_dataset):
    # Load the dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Transform the dataset
    transformed_data = []
    for item in dataset:
        id = item["id"]
        multi_item = multilingual_dataset[id]
        instruction = multi_item["instruction"]
        query = multi_item["query"]
        document = multi_item["document"]
        lang = multi_item["lang"]
        
        transformed_item = {
            "id": f"{id}-{lang}",
            "instruction": f"You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false). Answer using only one word, one of those two choices.\n",
            "input": f"Query: {query.strip()} {instruction.strip()}\nDocument: {document.strip()}\nRelevant (only output one word, either \"true\" or \"false\"):",
            "output": str(item['label']).lower()
        }
        transformed_data.append(transformed_item)
    
    return transformed_data

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Transform HuggingFace dataset with multilingual support")
    parser.add_argument("--dataset", default="jhu-clsp/FollowIR-train", help="HuggingFace dataset name")
    parser.add_argument("--multilingual_file", required=True, help="Path to the multilingual data file")
    parser.add_argument("--output", default="transformed_dataset.json", help="Output JSON file name")
    
    args = parser.parse_args()
    
    multilingual_data = load_multilingual_data(args.multilingual_file)
    transformed_data = transform_dataset(args.dataset, multilingual_data)
    save_to_json(transformed_data, args.output)
    
    print(f"Dataset transformed and saved to {args.output}")

if __name__ == "__main__":
    main()

    # example usage:
    # python make_multilingual_dataset.py --multilingual_file chinese_output.jsonl --output chinese_train.json
    # python make_multilingual_dataset.py --multilingual_file russian_output.jsonl --output russian_train.json
    # python make_multilingual_dataset.py --multilingual_file farsi_output.jsonl --output farsi_train.json