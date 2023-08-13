#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=count_doc_len
#SBATCH --time=3:00:00
#SBATCH --mem=30gb
#SBATCH --array=0-19

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


# max value for single partion memory 180000mb
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

WORKING_DIR="/pfs/work7/workspace/scratch/ukmwn-les_faits/LesFaits/analysis/document-sizes/python-scripts"

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=(
        "glm_books1.jsonl"
        "glm_books3.jsonl"
        "glm_FreeLaw.jsonl"
        "glm_europarl.jsonl"
        "glm_pubmed_abstracts.jsonl"
        "glm_ted_talks.jsonl"
        "glm_uncorpus.jsonl"
        "glm_uspto.jsonl"
        "glm_wikihow_corpus_version.jsonl"
        "glm_wikipedia.jsonl"
        "glm_wikiquotes.jsonl"
        "nli_qnli_train.jsonl"
        "pp_paws_train.jsonl"
        "qa_eli5_category_train.jsonl"
        "qa_lfqa_train.jsonl"
        "qa_mrqa_train.jsonl"
        "qa_wikihow_qa_version.jsonl"
        "sum_big_patent_summary_version_train.jsonl"
        "sum_scitldr_train.jsonl"
        "topic_ag_news_train.jsonl"
        "topic_dbpedia14_train.jsonl"
        )



DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}
echo "DATASET_NAME "$DATASET_NAME
echo "SLURM_ARRAY_TASK_ID "$SLURM_ARRAY_TASK_ID

pushd $WORKING_DIR

python count_len_doc.py --dataset-path ../../../../staging_area/train/$DATASET_NAME \
        --save-path ../../../../statistics_dataset/$DATASET_NAME \
        --num-proc 10 \
        --batch-size 100