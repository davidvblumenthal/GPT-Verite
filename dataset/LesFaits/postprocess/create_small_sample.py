import argparse
import jsonlines
import random
from tqdm import tqdm

def random_sample(input_file, output_file, percentage):


    # Open the input file and read the lines
    with open(input_file, 'r') as input_file:
        lines = input_file.readlines()

    # Determine the number of lines to sample
    num_lines = len(lines)
    num_to_sample = int(num_lines * percentage)

    # Shuffle the lines and select a random sample
    random.shuffle(lines)
    sampled_lines = random.sample(lines, num_to_sample)

    # Open the output file and write the sampled lines
    with open(output_file, 'w') as output_file:
        for line in sampled_lines:
            output_file.write(line)


"""

python create_small_sample.py \
      --input_file /pfs/work7/workspace/scratch/ukmwn-les_faits/les_faits_final/v2/no_sc_loss.jsonl \
      --output_file /pfs/work7/workspace/scratch/ukmwn-les_faits/les_faits_final/hp_sweep_ds/no_sc_loss.jsonl \
      --percentage 0.01 

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a shuffled sample of a JSON Lines file')
    parser.add_argument('--input_file', type=str, help='path to input JSON Lines file')
    parser.add_argument('--output_file', type=str, help='path to output JSON Lines file')
    parser.add_argument('--percentage', type=float, help='percentage of input file to sample')
    args = parser.parse_args()

    # Call the random_sample function with command-line arguments
    random_sample(args.input_file, args.output_file, args.percentage)



