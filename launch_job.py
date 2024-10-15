from __future__ import annotations

import argparse
import logging
import os

import mteb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_safe_folder_name(model_name):
    return model_name.replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation for a specified model."
    )
    parser.add_argument("model", type=str, help="The model to evaluate.")
    parser.add_argument("dataset", type=str, help="The dataset to evaluate.")
    args = parser.parse_args()

    model_name = args.model
    logger.info(f"Running evaluation for model: {model_name}")

    model = mteb.get_model(model_name)
    tasks = mteb.get_tasks(tasks=[args.dataset])

    # Create a safe folder name based on the model name
    safe_folder_name = get_safe_folder_name(model_name)
    output_folder = os.path.join("results_v2", safe_folder_name)

    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, batch_size=16, output_folder=output_folder, save_corpus_embeddings=True, save_predictions=True)

    logger.info(f"Evaluation completed. Results saved in {output_folder}")


if __name__ == "__main__":
    main()
