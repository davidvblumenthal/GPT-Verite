#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=clean_books
#SBATCH --time=8:30:00
#SBATCH --mem=120gb
#SBATCH --array=0

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

LIST_DATASET=(
        "glm_wikipedia.jsonl"
)

DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}

python main_filtering.py --data_files ../../../../staging_area/train/$DATASET_NAME \
        --path_dir_save_dataset ../../../../filter_test/$DATASET_NAME \
        --lang_dataset_id en \
        --path_sentencepiece_model ../../../../pre_pro_models/en.sp.model \
        --path_kenlm_model ../../../../pre_pro_models/en.arpa.bin \
        --path_fasttext_model ../../../../pre_pro_models/lid.176.bin \
        --num_proc 20



#python main_filtering.py --data_files ../../../../uncorpus.jsonl \
#        --path_dir_save_dataset ../../../../filter_test/uncorpus.jsonl \
#        --lang_dataset_id en \
#        --path_sentencepiece_model ../../../../pre_pro_models/en.sp.model \
#        --path_kenlm_model ../../../../pre_pro_models/en.arpa.bin \
#        --num_proc 5