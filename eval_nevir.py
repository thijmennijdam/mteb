import mteb
# any model that we already have in https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models
# or one you define
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# model_name = "mteb/models/paraphrase-multilingual-mpnet-base-v2"
# model_name = "GritLM/GritLM-7B"
# model_name = "GritLM/GritLM-8x7B"
# model_name = "castorini/repllama-v1-7b-lora-passage"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "samaya-ai/RepLLaMA-reproduced"
# model_name = "samaya-ai/promptriever-llama2-7b-v1"
# model_name = "castorini/monot5-base-msmarco"
# model_name = "castorini/monot5-base-msmarco-10k"
# model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# from mteb.encoder_interface import PromptType
# model_name = "text-embedding-3-large"

model = mteb.get_model(model_name)
tasks = mteb.get_tasks(tasks=["NevIR"])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(
    model,
    output_folder="results",
    batch_size=32,
    save_predictions=True,
    # previous_results="results/NevIR_default_predictions.json"
)