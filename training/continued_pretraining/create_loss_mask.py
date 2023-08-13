from transformers import AutoTokenizer

from datasets import load_dataset
from datasets.utils import disable_progress_bar

import argparse
from itertools import groupby

from utils.loss_mask_utils import split_ids_at_SENT_END_token
from utils.loss_mask_utils import construct_loss_mask
from utils.loss_mask_utils import chunks
from utils.loss_mask_utils import get_dataset
from utils.loss_mask_utils import preprocess_loss_mask




# Tokenize function for dataset.map()
def tokenize(element):
    # get outputs -> batch

    # do the preprocessing
    preprocessed_batch = preprocess_loss_mask(documents=element["text"])

    batch_ids = []
    batch_loss_mask = []

    outputs = tokenizer(preprocessed_batch)

    for input_ids in outputs["input_ids"]:

        input_ids = split_ids_at_SENT_END_token(input_ids)

        input_ids, loss_mask = construct_loss_mask(input_ids)

        batch_ids.append(input_ids)
        batch_loss_mask.append(loss_mask)

    return {"input_ids": batch_ids, "loss_mask": batch_loss_mask}


def tokenize_conc_chunk(element):
    
    # do the preprocessing
    preprocessed_batch = preprocess_loss_mask(documents=element["text"])
    
    
    batch_ids = []
    batch_loss_mask = []

    outputs = tokenizer(preprocessed_batch)

    for input_ids in outputs["input_ids"]:

        input_ids = split_ids_at_SENT_END_token(input_ids)

        input_ids, loss_mask = construct_loss_mask(input_ids)

        # append eos token at end of document; for loss mask zero
        input_ids.append(50256)
        loss_mask.append(0)

        # concatenate all examples in batch together by extending list
        batch_ids.extend(input_ids)
        batch_loss_mask.extend(loss_mask)

    # create chunks according to context window size
    batch_ids = chunks(batch_ids, n=2048, m=1500)
    batch_loss_mask = chunks(batch_loss_mask, n=2048, m=1500)

    return {"input_ids": batch_ids, "loss_mask": batch_loss_mask}
        



"""
python create_loss_mask.py \
      --input_file ./bigsample_Wikipedia_20221201.jsonl \
      --output_file ./lm_tokenized.jsonl \
      --concatenate

python create_loss_mask.py \
      --input_file /home/kit/stud/ukmwn/master_thesis/data/Wikipedia/Coref_Wikipedia_20221201.jsonl \
      --output_file ./data/sc_loss_standard_wiki \
      --save_as dataset \
      --concatenate
"""


        
if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default='./prepro_loss_mask.jsonl')
    parser.add_argument("--output_file", type=str, default='./tokenized_data.jsonl')
    parser.add_argument("--save_as", type=str, choices=["jsonl", "dataset"], default="dataset")

    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-125M", 
        help="Tokenizer to use")

    parser.add_argument("--concatenate", action="store_true", 
        help="If flagged concatenates batch together and splits into chunks of length context window")
    


    args = parser.parse_args()
    
    # specify the tokenizer model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # specify the special token
    special_token_end = '<|SENT_END|>'
    # special_token_mid = "|<SENT_MID>|"

    # add the special token to the tokenizer's vocabulary
    tokenizer.add_tokens([special_token_end], special_tokens=True)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    # tokenizer.add_special_tokens({'middle_of_sentence': [special_token_mid]})

    # save the tokenizer to a directory
    #tokenizer.save_pretrained("./new_tokenizer/")



    # Get dataset
    dataset = get_dataset(args.input_file, sample=False)
    
    # do the actual tokenization
    disable_progress_bar()

    if args.concatenate:
        tokenize = tokenize_conc_chunk

    tokenized_dataset = dataset.map(tokenize,
                                    keep_in_memory=True,
                                    batched=True,
                                    num_proc=60,
                                    batch_size=500,
                                    remove_columns=dataset.column_names # ["train"]
                                    )

    
    if args.save_as == "jsonl":
        # Write to jsonl
        tokenized_dataset.to_json(args.output_file, lines=True)

    if args.save_as == "dataset":
        tokenized_dataset.save_to_disk(args.output_file)


