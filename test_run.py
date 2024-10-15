from __future__ import annotations

import json
import logging

import mteb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    model_name = "unicamp-dl/mt5-base-mmarco-v2"
    model = mteb.get_model(model_name)
    tasks = mteb.get_tasks(
        tasks=[
            # "mFollowIRCrossLingualInstructionRetrieval",
            "mFollowIRInstructionRetrieval",
        ]
    )
    for i in range(len(tasks)):
        metadata = tasks[i].calculate_metadata_metrics()
        print(json.dumps(metadata))
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, 
        output_folder="results",
        batch_size=2,
)


if __name__ == "__main__":
    main()
