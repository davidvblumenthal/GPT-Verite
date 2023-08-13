#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=count_doc_len
#SBATCH --time=4:00:00
#SBATCH --mem=40gb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

WORKING_DIR="/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/analysis/document-sizes/python-scripts"

pushd $WORKING_DIR

python count_len_doc_final_ds.py --dataset-dir ../../../../les_faits_final \
        --save-path ../../../../les_faits_final/statistics \
        --num-proc 20 \
        --batch-size 200