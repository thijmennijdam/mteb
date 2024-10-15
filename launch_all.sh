#!/bin/bash

# List of non-7B models
non_7b_models=(
    # "intfloat/multilingual-e5-small"
    # "intfloat/multilingual-e5-base"
    # "intfloat/multilingual-e5-large"
    # "bm25s"
    # "facebook/mcontriever-msmarco"
    # "castorini/mdpr-tied-pft-msmarco-ft-all"
)

# List of 7B models
models_7b=(
    # "GritLM/GritLM-7B"
    # "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
    # "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised"
    # "Salesforce/SFR-Embedding-2_R"
    # "samaya-ai/promptriever-llama2-7b-v1"
    # "samaya-ai/promptriever-llama3.1-8b-instruct-v1"
    # "samaya-ai/promptriever-mistral-v0.1-7b-v1"
    # "samaya-ai/RepLLaMA-reproduced"
    # "intfloat/e5-mistral-7b-instruct"
    # "nomic-ai/nomic-embed-text-v1.5"
)

reranker_models=(
    "unicamp-dl/mt5-base-mmarco-v2"
    # "unicamp-dl/mt5-13b-mmarco-100k"
    # "jhu-clsp/mFollowIR-7B-all"
    "jhu-clsp/mFollowIR-7B-fas"
    "jhu-clsp/mFollowIR-7B-zho"
    "jhu-clsp/mFollowIR-7B-rus"
    # "jhu-clsp/FollowIR-7B"
    "mistralai/Mistral-7B-Instruct-v0.2"
)

# Function to create a safe job name
create_safe_job_name() {
    echo "mteb_$(echo $1 | sed 's/[^a-zA-Z0-9]/_/g')"
}

# Function to submit a job
submit_job() {
    local model=$1
    local dataset=$2
    local partition=$3
    local job_name=$(create_safe_job_name "$model")


# if partition is ba100, use the following command
if [ "$partition" == "ba100" ]; then
    sbatch --gpus 1 -J "$job_name" -p "$partition" << EOF
#!/bin/bash
#SBATCH --output=/brtx/605-nvme1/orionw/InstructionsInIR/neuclir-mteb/mteb/logs/${job_name}_%j.log
#SBATCH --error=/brtx/605-nvme1/orionw/InstructionsInIR/neuclir-mteb/mteb/logs/${job_name}_%j.log

# Use the specific Python environment
export PATH=~/anaconda3/envs/mteb-dev/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64
# print the time, formatted nicely
echo "Current time: $(date +"%Y-%m-%d %H:%M:%S")"

echo "Current GPU: $CUDA_VISIBLE_DEVICES"

# Run the Python script with the current model
echo "$(which python)"
~/anaconda3/envs/mteb-dev/bin/python launch_job.py "$model" "$dataset"
EOF

# if partition is brtx6, use the following command
else
    sbatch --gpus 1 -J "$job_name" -p "$partition" << EOF
#!/bin/bash
#SBATCH --output=/brtx/605-nvme1/orionw/InstructionsInIR/neuclir-mteb/mteb/logs/${job_name}_%j.log
#SBATCH --error=/brtx/605-nvme1/orionw/InstructionsInIR/neuclir-mteb/mteb/logs/${job_name}_%j.log

# Use the specific Python environment
export PATH=~/anaconda3/envs/mteb-dev-lower/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64

# print the time, formatted nicely
echo "Current time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Current GPU: $CUDA_VISIBLE_DEVICES"


# Run the Python script with the current model
echo "$(which python)"
~/anaconda3/envs/mteb-dev-lower/bin/python launch_job.py "$model" "$dataset"
EOF

fi


    echo "Submitted job for model: $model on partition: $partition"
}

for dataset in mFollowIRCrossLingualInstructionRetrieval mFollowIRInstructionRetrieval; do
    # Submit jobs for non-7B bi-encoder models
    # echo "Submitting jobs for non-7B models on brtx6 partition..."
    # for model in "${non_7b_models[@]}"; do
    #     submit_job "$model"  "$dataset" "brtx6"
    # done

    # Submit jobs for 7B bi-encoder models
    # echo "Submitting jobs for 7B models on ba100 partition..."
    # for model in "${models_7b[@]}"; do
    #     submit_job "$model" "$dataset" "ba100"
    # done

    # submit jobs for reranker models
    echo "Submitting jobs for reranker models on ba100 partition..."
    for model in "${reranker_models[@]}"; do
        submit_job "$model" "$dataset" "ba100"
    done
done

echo "All jobs submitted."