import json
import requests
import numpy as np
import pytrec_eval

def load_run_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    run = {}
    for qid, doc_scores in data.items():
        run[qid] = {doc_id: float(score) for doc_id, score in doc_scores.items()}
    
    return run

def download_and_load_qrels(url):
    response = requests.get(url)
    qrels = {}
    
    for line in response.text.strip().split('\n'):
        data = json.loads(line)
        qid = data['query-id']
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][data['corpus-id']] = int(data['score'])
    
    return qrels

# Load your run file
lang = 'zho'
run_file_path = f'results_v2/samaya-ai_promptriever-llama2-7b-v1/mFollowIRCrossLingualInstructionRetrieval_eng-{lang}_predictions.json'
run = load_run_file(run_file_path)
print(f"Loaded run with {len(run)} queries")

# Download and load qrels
qrels_url = f'https://huggingface.co/datasets/jhu-clsp/mFollowIR-{lang}-cl/raw/main/qrels_og/test.jsonl'
qrels = download_and_load_qrels(qrels_url)
print(f"Loaded qrels with {len(qrels)} queries")

# Define the metrics you want to calculate
metrics = {'ndcg_cut.10', 'ndcg_cut.20'}

# Create evaluator
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

# Evaluate
results = evaluator.evaluate(run)

# Calculate and print overall results
mean_scores = {metric: np.mean([query_result[metric.replace(".", "_")] for query_result in results.values()]) for metric in metrics}
print("Mean results:")
for metric, score in mean_scores.items():
    print(f"{metric}: {score:.4f}")

# Print and analyze nDCG@20 scores
ndcg20_scores = {qid: query_result['ndcg_cut_20'] for qid, query_result in results.items()}
print(f"\nResults for nDCG@20: {ndcg20_scores}")

# Separate the qids by ones with "2022" and "2023" in the name, then compare
results_2022 = []
results_2023 = []
for qid, score in ndcg20_scores.items():
    if '2022' in qid:
        results_2022.append(score)
    elif '2023' in qid:
        results_2023.append(score)

avg_2022 = np.mean(results_2022) if results_2022 else 0
avg_2023 = np.mean(results_2023) if results_2023 else 0

print(f"\nResults for queries with '2022' in the name: {results_2022}")
print(f"Average for 2022 queries: {avg_2022:.4f}")
print(f"\nResults for queries with '2023' in the name: {results_2023}")
print(f"Average for 2023 queries: {avg_2023:.4f}")