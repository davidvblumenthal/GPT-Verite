#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=aggregate_datasets
#SBATCH --time=10:30:00
#SBATCH --mem=180000mb


#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans


WORKING_DIR=/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/preprocessing/tokenizer/python_script/

pushd $WORKING_DIR

python david_aggregate_datasets.py --dataset_configuration_path ./train_meta.csv \
                                   --load_num_proc 5 \
                                   --save_path ../../../../les_faits_final/v2 \
                                   --save_num_proc 10
