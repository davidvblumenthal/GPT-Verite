from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import datasets
import torch

from tqdm.auto import tqdm
from itertools import dropwhile
import json
import argparse, os


def huggingface_model_pipeline(max_len, model_name_or_path, trained_with_padding):
    




    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    generate_kwargs={
                        'do_sample': False,  # generate_kwargs do_sample=True, max_length=50, top_k=50
                        'num_beams': 1,
                        'pad_token_id': tokenizer.pad_token_id
                        }

    # set pad token for batch inference   
    if trained_with_padding:
        print(f"Using the dedicated padding token!")
        tokenizer.pad_token = tokenizer.pad_token
    else:
        print("Using EOS token as pad token!")
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side="left"
    
    print(f"Using padding side: {tokenizer.padding_side} and pad token: {tokenizer.pad_token}")
    print(f"Pad token has id: {tokenizer.pad_token_id}")


    # create pipeline
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device="cuda:0", # "cpu",   #"cuda:0", #  cpu
                    max_new_tokens=max_len,
                    batch_size=64,
                    **generate_kwargs
                    ) 
    

    return pipe



