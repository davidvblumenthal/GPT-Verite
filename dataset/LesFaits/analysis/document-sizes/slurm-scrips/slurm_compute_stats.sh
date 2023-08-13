#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=tokenize-wiki-clean
#SBATCH --time=2:30:00
#SBATCH --mem=20gb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

WORKING_DIR=/"pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/analysis/document-sizes/python-scripts"
pushd $WORKING_DIR

python compute_stats.py --doc_length_dir ../../../../statistics_dataset/doc_len \
                        --doc_num_tokens_dir ../../../../statistics_dataset/num_tokens \
                        --statistics-pickle-file_length ../../../../statistics_dataset/doc_len/stats.pickle \
                        --statistics-pickle-file_tokens ../../../../statistics_dataset/num_tokens/stats.pickle


