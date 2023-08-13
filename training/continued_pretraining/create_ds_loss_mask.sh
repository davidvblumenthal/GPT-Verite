#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70
#SBATCH --job-name=create_ds_loss_standard_wiki
#SBATCH --time=5:00:00
#SBATCH --mem=120gb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc



# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

python create_loss_mask.py \
      --input_file /home/kit/stud/ukmwn/master_thesis/data/Wikipedia/Coref_Wikipedia_20221201.jsonl \
      --output_file ./data/sc_loss_standard_wiki \
      --save_as dataset \
      --concatenate