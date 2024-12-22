import argparse
import mteb

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
    
    if "promptriever" in args.model:
         model = mteb.get_model(
            model_name = args.model,
            model_prompts={
                "query": "query:  ",
                "passage": "passage:  "
            }
        )
    else:
        model = mteb.get_model(args.model)

    # Define tasks and evaluation
    tasks = mteb.get_tasks(tasks=["NevIR"])
    evaluation = mteb.MTEB(tasks=tasks)
    model_path = args.model.replace("/", "_")
    
    # Run evaluation
    print(f"Running evaluation with model {args.model}")
    evaluation.run(
        model,
        output_folder=f"results/{model_path}",
        batch_size=32,
        save_predictions=True,
        overwrite_results=True,
        previous_results=args.previous_results
        # previous_results="results/jinaai_jina-embeddings-v3/NevIR_default_predictions.json"
    )

if __name__ == "__main__":
    main()