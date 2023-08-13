#!/bin/bash
#SBATCH --job-name generation-2.7B_standard_wiki # Name for your job
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=60gb
#SBATCH --gres=gpu:1

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/evaluation/lm-evaluation-harness"
pushd $WORKING_DIR

HUG_NAMESPACE="davidvblumenthal/"

MODEL_NAME="GPT-Verite-125M-prototype"

MODEL=$HUG_NAMESPACE$MODEL_NAME


python main.py \
    --model hf-causal \
    --model_args pretrained=${MODEL} \
    --tasks wikitext \
    --device cuda:0 \
    --output_path ${MODEL_NAME}.json