import json
import random
import argparse


def sample_from_large_file(input_path, num_samples, num_read_lines):
    # Read in the first 1000 items from the JSON Lines file
    with open(input_path, "r") as f:
        lines = [next(f) for _ in range(num_read_lines)]

    # Shuffle the lines
    random.shuffle(lines)

    # Select a sample of 100 items
    sample = lines[:num_samples]

    return sample


"""

python sample_to_inspect.py --input_file ../new_qa_datasets/mrqa_train.jsonl --output_file ../new_qa_datasets/mrqa_train_sample.jsonl

"""


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="the path to the JSON Lines input file")
    parser.add_argument("--output_file", help="the path to the JSON Lines output file")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="number of samples to draw"
    )
    parser.add_argument(
        "--num_read",
        type=int,
        default=1000,
        help="number of samples to read in from input",
    )
    args = parser.parse_args()

    sample = sample_from_large_file(args.input_file, args.num_samples, args.num_read)

    # Write the sample to a new file
    with open(args.output_file, "w") as f:
        for line in sample:
            f.write(line)
