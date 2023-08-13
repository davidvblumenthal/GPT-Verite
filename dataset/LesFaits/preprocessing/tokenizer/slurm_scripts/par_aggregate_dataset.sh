#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=aggregate-dataset
#SBATCH --time=2:30:00
#SBATCH --mem=50gb
#SBATCH --array=0-1

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc

LIST_DATASET=(
        "yes"
        "no"
)

DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}

echo "DATASET_NAME "$DATASET_NAME
echo "SLURM_ARRAY_TASK_ID "$SLURM_ARRAY_TASK_ID



# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

WORKING_DIR=/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/preprocessing/tokenizer/python_script/

pushd $WORKING_DIR

python par_david_aggregate_datasets.py --dataset_configuration_path ./train_meta.csv \
                                   --subset $DATASET_NAME \
                                   --load_num_proc 2 \
                                   --save_path ../../../../les_faits_final/final \
                                   --save_num_proc 1