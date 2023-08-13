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


WORKING_DIR=/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/preprocessing/tokenizer/python_script/

pushd $WORKING_DIR

python train_tokenizer.py \
       --dataset_dir ../../../../staging_area/train \
       --save_dir ../../../../trained_tokenizers/gpt-vérité_tokenizer \
       --train_new