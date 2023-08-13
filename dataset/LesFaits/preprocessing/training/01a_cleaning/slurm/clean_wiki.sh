#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=clean_wiki
#SBATCH --time=2:30:00
#SBATCH --mem=50gb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


WORKING_DIR="/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/preprocessing/training/01a_catalogue_cleaning_and_filtering"

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

pushd $WORKING_DIR

python clean.py --dataset-path ../../../../staging_area/train/glm_wikipedia.jsonl \
        --preprocessings dedup_document filter_remove_empty_docs filter_wiki_non_text_type filter_small_docs_bytes_1024 \
        --save-path ../../../../filter_test/glm_wikipedia.jsonl \
        --num-proc 10 \
        --batch-size 100 \
        --save-to-json



#python main_filtering.py --data_files ../../../../uncorpus.jsonl \
#        --path_dir_save_dataset ../../../../filter_test/uncorpus.jsonl \
#        --lang_dataset_id en \
#        --path_sentencepiece_model ../../../../pre_pro_models/en.sp.model \
#        --path_kenlm_model ../../../../pre_pro_models/en.arpa.bin \
#        --num_proc 5