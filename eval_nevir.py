
# any model that we already have in https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models
# or one you define
# model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# model_name = "mteb/models/paraphrase-multilingual-mpnet-base-v2"
# model_name = "GritLM/GritLM-7B"
# model_name = "GritLM/GritLM-8x7B"
# model_name = "castorini/rankllama-v1-7b-lora-passage"
# model_name = "castorini/repllama-v1-7b-lora-passage"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "samaya-ai/RepLLaMA-reproduced"
# model_name = "samaya-ai/promptriever-llama2-7b-v1"
# model_name = "castorini/monot5-base-msmarco"
# model_name = "castorini/monot5-base-msmarco-10k"
# model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# from mteb.encoder_interface import PromptType
# model_name = "text-embedding-3-large"

import argparse
import mteb
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MTEB evaluation with a specified model.")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the model to use.")
    
    parser.add_argument(
        "--previous_results",
        type=str,
        default=None,
        help="Path to the previous results file (optional). Default is None."
    )
    args = parser.parse_args()
    model = mteb.get_model(args.model)

    # Define tasks and evaluation
    tasks = mteb.get_tasks(tasks=["NevIR"])
    evaluation = mteb.MTEB(tasks=tasks)

    # remove / from model name
    model_path = args.model.replace("/", "_")
    
    # Run evaluation
    print(f"Running evaluation with model {args.model}")
    evaluation.run(
        model,
        output_folder=f"results/{model_path}",
        batch_size=32,
        save_predictions=True,
        previous_results=args.previous_results
    )

if __name__ == "__main__":
    main()