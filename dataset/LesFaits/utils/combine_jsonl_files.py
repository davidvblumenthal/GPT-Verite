import json
import os
import glob
import argparse


def combine_jsonl(input_dir, output_file):
    # Open the output file for writing
    with open(output_file, 'w') as f_out:
        # Iterate over the JSON Lines files in the input directory
        for jsonl_file in glob.glob(os.path.join(input_dir, '*.jsonl')):
            # Open the input file for reading
            with open(jsonl_file, 'r') as f_in:
                # Iterate over each line in the input file
                for line in f_in:
                    # Load the JSON object from the line
                    data = json.loads(line)
                    # Write the JSON object to the output file
                    json.dump(data, f_out)
                    f_out.write('\n')
            # Delete the input file after processing
            os.remove(jsonl_file)


"""

python combine_jsonl_files.py --dataset_path ../../books1/ \
       --output_file_name books1.jsonl

"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="markdown files to jsonl.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Name of the dataset to load.",
    )

    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Path to save location."
    )

    args = parser.parse_args()

    combine_jsonl(args.dataset_path, args.output_file_name)
