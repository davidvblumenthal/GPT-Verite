import json
import random

# specify the input and output file paths
input_file_path = "/home/kit/stud/ukmwn/master_thesis/data/Wikipedia/Wikipedia_20221201.jsonl"
output_file_path = "./sample_Wikipedia_20221201.jsonl"

# read all lines in the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# sample 100 lines from the input file
sample_lines = random.sample(lines, 10000)

# write the sampled lines to the output file
with open(output_file_path, "w") as output_file:
    for line in sample_lines:
        output_file.write(line)
