#!/bin/bash
#SBATCH --job-name toxicity_prompts # Name for your job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=20gb

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/evaluation/toxicity_prompts"
pushd $WORKING_DIR


python evaluate_toxicity.py \
      --model_name EleutherAI/pythia-1.4b \
      --output_dir /home/kit/stud/ukmwn/master_thesis/evaluation/toxicity_prompts/evaluations \
      --perspective_rate_limit 30 \
      --max_len 50
      



# --trained_with_padding