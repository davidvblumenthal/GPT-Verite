#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=train_1.3B_no_packing
#SBATCH --time=20:00:00
#SBATCH --mem=80gb
#SBATCH --gres=gpu:4

#SBATCH --mail-user david.blumenthal@partner.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


#module load devel/cuda/11.6
module purge

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

# RUN BENCHMARK
accelerate launch --config_file ../configs/acc_config_deepspeed.yml \
                       ../accelerate_deepspeed.py \
                       --tokenized_dataset ../../data/tokenized_wiki_no_packing \
                       --model_name_or_path EleutherAI/gpt-neo-1.3B \
                       --per_device_train_batch_size 1 \
                       --per_device_eval_batch_size 1 \
                       --learning_rate 2e-6 \
                       --weight_decay 0.01 \
                       --num_warmup_steps 100 \
                       --num_train_epochs 1 \
                       --gradient_accumulation_steps 32 \
                       --output_dir ../../model_checkpoints/1.3B_no_packing \
                       --checkpointing_steps 6000 \
                       --with_tracking \
                       --report_to wandb \
                       --exit_duration 1195 \
                       --resume_from_checkpoint ../../model_checkpoints/1.3B_no_packing/                   
                       


# ../../data/tokenized_wiki_no_packing
#--resume_from_checkpoint ../../model_checkpoints/1.3B_no_packing/step_400