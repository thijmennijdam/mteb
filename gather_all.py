import os
import json
import csv
import glob

def extract_values(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for result in data['scores']['test']:
        lang_pair = '-'.join(result['languages'])
        results[lang_pair] = {
            'ndcg@20': result['individual']['original']['ndcg_at_20'],
            'p-MRR': result['main_score']
        }
    
    return results

def process_directory(directory):
    cross_lingual_pattern = os.path.join(directory, '**', 'mFollowIRCrossLingualInstructionRetrieval.json')
    normal_pattern = os.path.join(directory, '**', 'mFollowIRInstructionRetrieval.json')
    
    results = {}
    
    cross_lingual_files = glob.glob(cross_lingual_pattern, recursive=True)
    if cross_lingual_files:
        results['cross_lingual'] = extract_values(cross_lingual_files[0])
    
    normal_files = glob.glob(normal_pattern, recursive=True)
    if normal_files:
        results['normal'] = extract_values(normal_files[0])
    
    return results

def main():
    results_dir = 'results_v2'
    output_file = 'results_summary.csv'
    missing_models = []
    
    all_results = {}
    
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            results = process_directory(model_path)
            if results:
                all_results[model_dir] = results
            else:
                missing_models.append(model_dir)
    
    # Write results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Type', 'Language Pair', 'ndcg@20', 'p-MRR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for model, data in all_results.items():
            for result_type, lang_pairs in data.items():
                for lang_pair, metrics in lang_pairs.items():
                    writer.writerow({
                        'Model': model,
                        'Type': result_type,
                        'Language Pair': lang_pair,
                        'ndcg@20': metrics['ndcg@20'],
                        'p-MRR': metrics['p-MRR']
                    })
    
    print(f"Results saved to {output_file}")
    
    if missing_models:
        print("\nModels missing results:")
        for model in missing_models:
            print(model)

if __name__ == "__main__":
    main()