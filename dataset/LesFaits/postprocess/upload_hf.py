import argparse
import json
import logging

import pandas as pd

import datasets
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import login


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset_configuration_path",
        type=str,
        required=True,
        help="path to csv file containing input dataset ratios.",
    )
    
    parser.add_argument(
        "--load_num_proc", type=int, default=1, help="number of procs to use for loading datasets, default 1"
    )
    # Save
    
    
    # Parse args
    args = parser.parse_args()

    return args



def read_configuration(args):
    csv_path = args.dataset_configuration_path

    config = pd.read_csv(csv_path)

    return config


def process_single_dataset(config_row, load_num_proc=1):

    dataset = load_dataset("json", data_files=config_row["path"], split="train", num_proc=load_num_proc)

    # Append some meta columns
    category = [config_row["category"]] * len(dataset)
    task = [config_row["task"]] * len(dataset)
    name = [config_row["name"]] * len(dataset)

    dataset = dataset.add_column("category", category)
    dataset = dataset.add_column("task", task)
    dataset = dataset.add_column("name", name)

    
    return dataset


def process_all(args):

    les_faits_datasets = []

    # get dataset config
    config = read_configuration(args)

    for index, row in config.iterrows():
        log_name = row["name"]
        logger.info(f"Processing: {log_name}")

        curr_dataset = process_single_dataset(row, args.load_num_proc)
        les_faits_datasets.append(curr_dataset)

        

    logger.info("Starting to concatenate the datasets....")

    les_faits = concatenate_datasets(les_faits_datasets)

    logger.info("Finished concatenating..")

    return les_faits


def main():
    args = parse_args()

    les_faits = process_all(args)

    login()
    repo_name = input("Provide a repository name for the HF Hub: ")
    
    les_faits.push_to_hub(repo_name)


if __name__ == "__main__":
    main()


