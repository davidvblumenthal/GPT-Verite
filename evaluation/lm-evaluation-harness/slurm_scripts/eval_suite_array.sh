#!/bin/bash
#SBATCH --job-name eval_suite # Name for your job
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=60gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate torch_trans

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/evaluation/lm-evaluation-harness"
pushd $WORKING_DIR


MODEL_LIST=("EleutherAI/pythia-1.4b" "davidvblumenthal/GPT-Verite_1.4B" "davidvblumenthal/160M_GPT-Verite" "davidvblumenthal/160M_padding_v1")

MODEL=${MODEL_LIST[${SLURM_ARRAY_TASK_ID}]}

# Get model name without namespace
# Split the string at the "/" character and retrieve the part after it
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)

echo "Evaluating model: $MODEL_NAME"



python main.py \
    --model hf-causal \
    --model_args pretrained=${MODEL} \
    --tasks lambada_openai,piqa,winogrande,wsc273,arc_easy,arc_challenge,sciq,logiqa,hellaswag,openbookqa,truthfulqa_gen,mnli,triviaqa,toxigen \
    --device cuda:0 \
    --output_path results/${MODEL_NAME}.json



# 
