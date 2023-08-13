#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --job-name=clean_wiki
#SBATCH --time=48:30:00
#SBATCH --mem=180000mb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

WORKING_DIR="/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/entity_tokenizer"
pushd $WORKING_DIR



python count_factuality_prompts_entities.py \
       --save_path ./entity_count.pickle \
       --num_proc 12


