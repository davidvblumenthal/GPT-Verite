#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=train_125m_loss_mask
#SBATCH --time=30:00:00
#SBATCH --mem=80gb
#SBATCH --gres=gpu:4

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate trans

# RUN BENCHMARK
accelerate launch --config_file ./acc_config.yml \
                       train_acc.py \
                       --tokenized_dataset_path ./data/sc_loss_standard_wiki \
                       --model_name_or_path EleutherAI/gpt-neo-125M \
                       --output_dir ./125m_sc_loss_v2 \
                       --loss_mask_multiple 2.0 \
                       --per_device_train_batch_size 3 \
                       --learning_rate 2e-4 \
                       --weight_decay 0.01 \
                       --num_warmup_steps 1000 \
                       --num_train_epochs 1 \
                       --gradient_accumulation_steps 16 \
                       --mixed_precision fp16 \
                       --checkpointing_steps 4000 \
                       --report_to wandb