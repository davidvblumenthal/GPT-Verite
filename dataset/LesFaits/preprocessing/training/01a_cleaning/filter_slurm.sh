#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=tokenize-wiki-clean
#SBATCH --time=2:30:00
#SBATCH --mem=30gb
#SBATCH --array=0-2

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc

LIST_DATASET=(
        "eli5_category_train.jsonl"
        "qa_lfqa_train.jsonl"
        "qa_mrqa_train.jsonl"
)

DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}

echo "DATASET_NAME "$DATASET_NAME
echo "SLURM_ARRAY_TASK_ID "$SLURM_ARRAY_TASK_ID


DATASET_ID=$SLURM_ARRAY_TASK_ID
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

python clean.py --dataset-path ../../../../new_qa_datasets/$DATASET_NAME \
        --preprocessings dedup_document filter_remove_empty_docs filter_small_docs \
        --save-path ../../../../new_qa_datasets/filtered/$DATASET_NAME \
        --num-proc 2 \
        --batch-size 100 \
        --save-to-json



#python main_filtering.py --data_files ../../../../uncorpus.jsonl \
#        --path_dir_save_dataset ../../../../filter_test/uncorpus.jsonl \
#        --lang_dataset_id en \
#        --path_sentencepiece_model ../../../../pre_pro_models/en.sp.model \
#        --path_kenlm_model ../../../../pre_pro_models/en.arpa.bin \
#        --num_proc 5