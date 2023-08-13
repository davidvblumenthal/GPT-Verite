#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=train_tokenizer
#SBATCH --time=6:00:00
#SBATCH --mem=100gb


#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans


WORKING_DIR=/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/entity_tokenizer
pushd $WORKING_DIR

python train_tokenizer.py \
       --dataset_dir /pfs/work7/workspace/scratch/ukmwn-les_faits/staging_area/train \
       --save_dir /pfs/work7/workspace/scratch/ukmwn-les_faits/trained_tokenizers/gpt-vérité_entity_tokenizer \
       --path_entity_special_tokens /pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/entity_tokenizer/final_token_entities.jsonl \
       --train_new