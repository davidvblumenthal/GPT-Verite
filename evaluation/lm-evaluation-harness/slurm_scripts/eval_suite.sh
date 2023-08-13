#!/bin/bash
#SBATCH --job-name eval_suite # Name for your job
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=50gb
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

MODEL_NAME="1.4B-GPT-Verite_no_sc"

MODEL=$HUG_NAMESPACE$MODEL_NAME


python main.py \
    --model hf-causal \
    --model_args pretrained=${MODEL} \
    --tasks lambada_openai,piqa,winogrande,wsc273,arc_easy,arc_challenge,sciq,logiqa,hellaswag,openbookqa,truthfulqa_gen,mnli,triviaqa \
    --device cuda:0 \
    --output_path results/${MODEL_NAME}.json



# ,toxigen
