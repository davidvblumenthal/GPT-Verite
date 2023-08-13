#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=clean_wiki
#SBATCH --time=48:30:00
#SBATCH --mem=180000mb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans


python count_entities.py --corpus_path /pfs/work7/workspace/scratch/ukmwn-les_faits/staging_area/train/glm_wikipedia.jsonl \
        --save_path ./freq_entities.pickle \
        --entities ./entity_trie_original_strings.jsonl


