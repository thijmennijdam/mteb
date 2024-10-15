import json
import sys

# Constants for pricing
INPUT_PRICE_PER_MILLION = 1.25
OUTPUT_PRICE_PER_MILLION = 5.00

def process_jsonl(file_path):
    total_input_tokens = 0
    total_output_tokens = 0

    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                usage = data['response']['body']['usage']
                total_input_tokens += usage['prompt_tokens']
                total_output_tokens += usage['completion_tokens']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
                continue

    return total_input_tokens, total_output_tokens

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_jsonl_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    input_tokens, output_tokens = process_jsonl(file_path)
    input_cost, output_cost, total_cost = calculate_cost(input_tokens, output_tokens)

    print(f"Total input tokens: {input_tokens}")
    print(f"Total output tokens: {output_tokens}")
    print(f"Input cost: ${input_cost:.2f}")
    print(f"Output cost: ${output_cost:.2f}")
    print(f"Total cost: ${total_cost:.2f}")

if __name__ == "__main__":
    main()