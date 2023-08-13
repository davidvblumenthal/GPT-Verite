#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=calc_metrics
#SBATCH --time=03:00:00
#SBATCH --mem=10gb

#SBATCH --array=0-1

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc

# module load devel/cuda/10.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
# conda activate factuality_prompts
conda activate fact_prompts

promptTypes=("factual" "nonfactual")

GEN_FOLDER="standard_wiki_125m"


PROMPT_TYPE=${promptTypes[${SLURM_ARRAY_TASK_ID}]}

GEN_TO_EVALUATE_NAME="${promptTypes[${SLURM_ARRAY_TASK_ID}]}-gen.jsonl"


PYTHONPATH=. python src/auto_evaluation.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME} \
        --gen_dir ${GEN_FOLDER} \
        --use_additional_ne \
        --dedub_generation \
        --entailment_model large_mnli
        
        