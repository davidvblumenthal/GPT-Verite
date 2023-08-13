#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=train_1.3B_standard_wiki_3
#SBATCH --time=12:00:00
#SBATCH --mem=60gb
#SBATCH --gres=gpu:4

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

# RUN BENCHMARK
torchrun \
  --nproc_per_node 4 training_script.py
