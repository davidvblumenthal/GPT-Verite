from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AdamW

from datasets import load_from_disk
from datasets import concatenate_datasets
from datasets.utils import disable_progress_bar

from transformers import DataCollatorForLanguageModeling

from transformers import Trainer
from transformers import TrainingArguments

import torch
import transformers

### Sepcify paths here
CONTEXT_LENGTH = 2048
DATA_PATH = "../data/tokenized_Wiki_clean"
checkpoint = "../EleutherAI/gpt-neo-1.3B"

checkpoint_folder = "1.3B_clean_tokenization"

use_A100 = True

# DATA realted stuff
tokenized_dataset = load_from_disk(DATA_PATH)

'''
tokenized_dataset = concatenate_datasets(
    [tokenized_dataset["train"], tokenized_dataset["test"]]
)
'''
# tokenized_dataset = tokenized_dataset.select(range(0, 200))

print(f"Dataset found with the follwing values: {tokenized_dataset}")


##### TOKENIZER RELATED STUFF #####
# Tnitialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# set eos token
tokenizer.pad_token = tokenizer.eos_token


#### DATA COLLATOR ####
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="pt"
)

# test data_collator
'''
out = data_collator([tokenized_dataset[i] for i in range(2)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
'''

# Define model
model = AutoModelForCausalLM.from_pretrained(checkpoint)


"""
    Gradient Accumulaton ->     does forward, backward pass in small batches and accumulated gradient
                                saves memory, slows down training only slightly
    
    Gradient Checkpointing ->   saves stragegically selected activiations to deacrease memory needs
                                slows down training about 20%
    
    FP16 Training         ->    Mixed percision training saves activiations in fp16 instead of fp32
                                model is present on gpu in 16-bit and 32-bit, which needs more memory
                                1.5x more but increases training speed
    BF16                    ->  datatype with larger dynamic range than fp16 (available when using
                                A100 GPUs
    TF32                    ->  datatype that can increase throughput by 3x (available when usung A100)
                                import torch
                                torch.backends.cuda.matmul.allow_tf32 = True
    
effieciency_args = (gradient_accumulation_steps=4,
                    gradient_checkpointing=True,
                    fp16=True,
                    tf32=True,
                    bf16=True)
"""
transformers.logging.set_verbosity_info()

default_args = {
    "learning_rate": 2e-4,  # 1.3b and 1e-5 530B
    "weight_decay": 0.01,
    "warmup_steps": 1_000,
    "save_steps": 40,
    "save_strategy": "steps",
    "evaluation_strategy": "no",
    # "eval_steps": 5_000,
    "save_total_limit": 2,
    # "load_best_model_at_end": True,
    "report_to": "wandb",
    "run_name": "1.3B_clean_tokenization"
}


### HUGGINGFACE TRAINER SPECIFICS ###

training_args = TrainingArguments(
    checkpoint_folder,
    # Pytorch 2.0
    # torch_compile=True,
    per_device_train_batch_size=2,
    # per_device_eval_batch_size=12,
    num_train_epochs=1,
    fp16=True,
    #bf16=True,
    #tf32=True,
    gradient_checkpointing=False,
    # use_cache=False, # cache not competible with gradient checkpointing
    gradient_accumulation_steps=64,
    **default_args,
)
"""
if use_A100:
    training_args.tf32 = True
    training_args.fp16 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("\n Utilizing datatype tf32!!!")
"""
# Define Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset,  # ["train"]
    # eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start trainign loop
trainer.train(resume_from_checkpoint=True)  # "./1.3B_coref_wiki/checkpoint-5000" 

# Save the last model inside the Trainer
trainer.save_model(checkpoint_folder)
