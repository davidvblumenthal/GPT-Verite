#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=tokenize-wiki-clean
#SBATCH --time=2:00:00
#SBATCH --mem=60gb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc



# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

python create_dataset.py --method no_packing --save_path ../data/tokenized_wiki_no_packing

