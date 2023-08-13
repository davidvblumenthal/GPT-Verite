import os
import argparse
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset
from datasets.utils.logging import set_verbosity_info
from datasets import concatenate_datasets

from transformers import AutoTokenizer

set_verbosity_info()
logger = logging.getLogger(__name__)


tokenizer = AutoTokenizer.from_pretrained("../../../../trained_tokenizers/gpt-vérité_tokenizer")

def get_file_paths(directory):
    """
    Gets the paths of all files in a directory and returns them as a list.
    :param directory: The path to the directory containing the files.
    :return: A list of file paths.
    """
    # Initialize an empty list to store the file paths
    file_paths = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Get the full path to the file
        file_path = os.path.join(directory, filename)

        # Check if the file is a regular file
        if os.path.isfile(file_path):

            # Append the file path to the list
            file_paths.append(file_path)

    # Return the list of file paths
    return file_paths


def merge_datasets(file_paths):

    datasets = []

    for file_path in file_paths:

        name = file_path.split("/")[-1].replace(".jsonl", "")
        logger.info(f"Getting dataset: {name}")

        dataset = load_dataset("json", data_files=file_path, split="train")#, streaming=True)
        datasets.append(dataset)

    # construct the corpus with iterleave
    logger.info("Starting to merge subsets!!")
    logger.info(f"Number of Datasets loaded: {len(datasets)}")

    print(datasets)
    
    #final_corpus = interleave_datasets(datasets)
    final_corpus = concatenate_datasets(datasets)

    del datasets

    return final_corpus


def compute_num_tokens(sample):
    length = []
    
    outputs = tokenizer(sample["text"])
    
    for token_ids in outputs["input_ids"]:
        length.append(len(token_ids))
    
    return {"len": length}


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(
        f"** The job is runned with the following arguments: **\n{args}\n **** "
    )

    logger.info(f" ===== Loading files in: {args.dataset_dir} =====")

    ds_paths = get_file_paths(args.dataset_dir)

    ds = merge_datasets(ds_paths)

    #ds = load_dataset("json", data_files=str(args.dataset_path))
    #ds = load_dataset("json", data_files=args.dataset_path)

    logger.info(f"ds info: {ds}")

    logger.info(f" ===== Starting to count the number of tokens =====")

    ds_num_tokens = ds.map(
        compute_num_tokens,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=["text"]
    )

    # Create save_path if not exists

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    logger.info(f" ===== Saving results to disk at: {args.save_path} =====")

    ds_num_tokens.save_to_disk(args.save_path)