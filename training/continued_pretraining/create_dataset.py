from transformers import AutoTokenizer
from datasets import load_dataset
from datasets.utils import disable_progress_bar

import argparse
import random


##### TOKENIZER RELATED STUFF #####
# Tnitialize tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#    checkpoint, model_max_length=2048, truncation="longest_first"
# )


# Custom tokenization functions
# Custom truncation function
def chunks(lst, n, m=1500):
    """
    Takes a list containing conc tokens and returns:
    sublists with a maximum len of n and minimum lenght of m
    """
    results = []
    for i in range(0, len(lst), n):
        if len(lst[i : i + n]) > m:
            results.append(lst[i : i + n])

    return results


# Tokenization function for dataset.map()
def tokenize(element):
    """
    Takes a huggingface dataset batch as input and
    concates all tokens in that batch with appending eos token;
    chunks the concatenated tokens using above func:chunk() and returns
    the resulting input_ids

    """
    outputs = tokenizer(element["text"])

    # print(f'\n länge von outputs[input_ids] {len(outputs["input_ids"])}')

    input_batch = []

    for input_ids in outputs["input_ids"]:
        # print(f'\n before insertion {input_ids}')
        input_ids.append(50256)
        # print(f'\n after insertion: {input_ids}')

        input_batch.extend(input_ids)

    # create chunks according to context size
    input_batch = chunks(input_batch, CONTEXT_LENGTH)

    return {"input_ids": input_batch}


def tokenize_clean(element):
    outputs = tokenizer(element["text"])

    current_len = 0
    current_sample = (
        []
    )  # list where samples are continuously concatenated until max context window size is reached
    input_batch = []  # list where the final samples are stored

    # Loop over examples
    for input_ids in outputs["input_ids"]:
        # track length of current example
        current_len += len(input_ids)

        # Check if len of current sample still smaller than max_context (2048)
        if current_len < 2047:
            input_ids.append(50256)  # append end of sequence token

            # Extend the current sample under construction with the current sample in iteration
            current_sample.extend(input_ids)
            continue

        # Check if len is already larger than the max_context (2048)
        if current_len > 2047:
            # Append the current constructed sample to the final samples

            if len(current_sample) > 0 and len(current_sample) <= 2047:
                # A
                if len(current_sample) > 2047:
                    print(f"Inside A {len(current_sample)}")
                input_batch.append(current_sample)

            # Reset sample size tracker only to current sample
            current_len = len(input_ids)
            # Reset the construction list
            current_sample = []

            # Check if current sample is smaller than max_context window
            if current_len < 2047:
                # If so append sample to the construction list
                input_ids.append(50256)
                current_sample.extend(input_ids)

            # Check if current sample is bigger than max_context on its own
            if current_len > 2047:
                # If so -> truncate to max_length
                truncated_input_ids = chunks(input_ids, 2047, 0)

                # print(f"Single Sample größer 2047 {truncated_input_ids}")

                # Append the the truncated samples to the final list
                # B
                if len(truncated_input_ids[0]) > 2047:
                    print(f"Inside B {len(truncated_input_ids[0])}")
                input_batch.append(truncated_input_ids[0])

                # //TODO handle the overflowing tokens
                for idx, trunc_sample in enumerate(truncated_input_ids):
                    # Check if last item
                    if idx == len(truncated_input_ids) - 1:
                        if len(trunc_sample) >= 1000:
                            trunc_sample.append(50256)
                            # C
                            if len(trunc_sample) > 2048:
                                print(f"Inside C {len(trunc_sample)}")
                            input_batch.append(trunc_sample)

                    if len(trunc_sample) == 2047:
                        # D
                        if len(trunc_sample) > 2047:
                            print(f"Inside D {len(trunc_sample)}")
                        input_batch.append(trunc_sample)

                # Reset current len
                current_len = 0

    return {"input_ids": input_batch}


def tokenize_no_packing_trunc(element):
    outputs = tokenizer(
        element["text"], truncation=True
    )  # , return_overflowing_tokens=True)

    return outputs


def tokenize_no_packing_return_overflow(element):
    context_length = 2048
    
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length and length > 10:
            input_batch.append(input_ids)
    
    return {"input_ids": input_batch}



def detokenize(element):
    """
    Takes a huggingface dataset batch as input and
    detokenizes the input_ids and returns the resulting text
    """
    texts = []
    outputs = tokenizer.batch_decode(element["input_ids"])
    
    for output in outputs:
        texts.append(output)

    return {"decoded": texts}



"""
number_of_tokens = 0
num_samples = len(tokenized_dataset["input_ids"])


for input_ids in tokenized_dataset["input_ids"]:
    number_of_tokens += len(input_ids)

print(f"Average Sequence Length: {number_of_tokens/num_samples} Tokens!")
"""

"""

python create_dataset.py \
           --method no_packing \
           --data_path /Users/davidblumenthal/Documents/Master_Thesis/Evaluation/gpt-ver/data/sample_Wikipedia_20221201.jsonl \
           --save_path ./inspect_tokens.jsonl \
           --tokenizer_checkpoint davidvblumenthal/GPT-Verite

"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create a tokenized dataset")

    parser.add_argument(
        "--method",
        type=str,
        default="no_packing",
        help="Tokenization type possilbe values [full_packing, carefull_packing, no_packing]",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/kit/stud/ukmwn/master_thesis/data/Wikipedia/Wikipedia_20221201.jsonl",
    )
    parser.add_argument("--save_path", type=str)

    parser.add_argument(
        "--tokenizer_checkpoint", type=str, default="EleutherAI/gpt-neo-1.3B"
    )

    parser.add_argument("--create_small_eval_set", type=bool, default=False)

    args = parser.parse_args()

    # intialize tokenizer

    ### Sepcify paths here
    CONTEXT_LENGTH = 2048

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)

    # DATA realted stuff
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    # shuffle dataset
    dataset = dataset.shuffle(seed=42)

    # Create small eval dataset from the same samples
    if args.create_small_eval_set:
        eval_dataset = dataset.select(random.sample(range(0, len(dataset)), 250))
        eval_dataset = eval_dataset.map(
            tokenize_no_packing_return_overflow,
            batched=True,
            num_proc=1,
            batch_size=250,
            remove_columns=eval_dataset.column_names,
        )

        # Construct save path
        eval_save_path = args.save_path + "_eval"
        eval_dataset.save_to_disk(eval_save_path)

        print(f"Evaluation dataset saved with the follwing values: {eval_dataset}")

    # create a train test split
    # dataset = dataset.train_test_split(test_size=0.05, shuffle=False)

    # dataset = dataset.select(range(0, 6000))
    print(f"Dataset found with the follwing values: {dataset}")


    # set eos token
    # tokenizer.pad_token = tokenizer.

    tokenize_method = {
        "full_packing": tokenize,
        "carefull_packing": tokenize_clean,
        "no_packing": tokenize_no_packing_return_overflow,
    }

    # do the actuall tokenization
    # disable_progress_bar()

    # map dataset function
    tokenized_dataset = dataset.map(
        tokenize_method[args.method],
        keep_in_memory=False,
        batched=True,
        num_proc=20,
        batch_size=500,
        remove_columns=dataset.column_names,  # ["train"]
    )

    print(f"Dataset after tokenization: {tokenized_dataset}")

    print("Saving dataset to: " + args.save_path)

    tokenized_dataset.save_to_disk(args.save_path)

    """
    tokenized_dataset = tokenized_dataset.map(
        detokenize,
        keep_in_memory=False,
        batched=True,
        num_proc=1,
        batch_size=500,
    )
    """
    # tokenized_dataset.to_json(args.save_path)
