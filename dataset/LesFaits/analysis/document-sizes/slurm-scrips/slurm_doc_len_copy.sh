#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=count_doc_len
#SBATCH --time=3:00:00
#SBATCH --mem=30gb
#SBATCH --array=0

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

WORKING_DIR="/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/analysis/document-sizes/python-scripts"

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=(
        "glm_europarl.jsonl"
        )



DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}
echo "DATASET_NAME "$DATASET_NAME
echo "SLURM_ARRAY_TASK_ID "$SLURM_ARRAY_TASK_ID

pushd $WORKING_DIR

python count_len_doc.py --dataset-path ../../../../staging_area/train/$DATASET_NAME \
        --save-path ../../../../statistics_dataset/$DATASET_NAME \
        --num-proc 20 \
        --batch-size 1000