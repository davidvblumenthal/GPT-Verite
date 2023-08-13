#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=build_books3
#SBATCH --time=6:00:00
#SBATCH --mem=180000mb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc



# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

python get_datasets.py --datasets books3 --single_dataset --save_dir ../



