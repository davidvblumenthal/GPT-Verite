import argparse
import json
import logging
import os
import multiprocessing
import re
from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, Union, Optional, List

import datasets
from numpy import log10
from numpy.random import default_rng, SeedSequence

import random

import pandas as pd

from datasets import concatenate_datasets, load_dataset, utils, Features, Value, Dataset
import sys

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
        "--subset",
        type=str,
        help="Subset to process for slurm jobarray [sc_loss, no_sc_loss]"
    )
    
    parser.add_argument(
        "--load_num_proc", type=int, default=1, help="number of procs to use for loading datasets, default 1"
    )
    # Save
    parser.add_argument("--save_path", type=str, default=".", help="path to save the dataset, default '.'")
    parser.add_argument("--save_num_proc", type=int, default=1, help="number of procs to use for saving, default 1")
    
    # Parse args
    args = parser.parse_args()
    
    # Post-process args
    args.dataset_configuration_path = Path(args.dataset_configuration_path)
    args.save_path = Path(args.save_path)
    return args




def read_configuration(args):
    csv_path = args.dataset_configuration_path

    config = pd.read_csv(csv_path)

    return config



def process_single_dataset(config_row, seed, load_num_proc=1):

    dataset = load_dataset("json", data_files=config_row["path"], split="train", num_proc=load_num_proc)

    if config_row["ratio"] < 1:

        num_samples = int(len(dataset) * config_row["ratio"])
        
        if num_samples == 0:
            return None
        
        #rng = default_rng(seed)
        #indices = rng.choice(len(dataset), size=num_samples, replace=False, shuffle=False)
        """
            Range, list or 1D-array of integer indices for indexing.
            If the indices correspond to a contiguous range, the Arrow table is simply sliced.
            However passing a list of indices that are not contiguous creates indices mapping, which is much less efficient,
            but still faster than recreating an Arrow table made of the requested rows.
            https://github.com/huggingface/datasets/blob/2.10.0/src/datasets/arrow_dataset.py#L3584
        """
        #indices = random.sample(list(range(len(dataset))), num_samples)
        indices = list(range(num_samples))
        
        logger.info("Currently flattening indices...")
        dataset = dataset.select(indices, writer_batch_size=5000)

    if config_row["ratio"] > 1:
        log_oversampling = int(config_row["ratio"])
        logger.info(f"Oversampling Dataset {log_oversampling} times...")

        dataset = [dataset] * int(config_row["ratio"])
        dataset = concatenate_datasets(dataset)


    
    # Append some meta columns
    category = [config_row["category"]] * len(dataset)
    task = [config_row["task"]] * len(dataset)
    name = [config_row["name"]] * len(dataset)

    dataset = dataset.add_column("category", category)
    dataset = dataset.add_column("task", task)
    dataset = dataset.add_column("name", name)

    
    return dataset


def process_all(args, seed):

    subpart = []

    # get dataset config
    config = read_configuration(args)

    for index, row in config.iterrows():
        
        if row["sc_loss"] == args.subset:
        
            log_name = row["name"]
            logger.info(f"Processing: {log_name}")

            curr_dataset = process_single_dataset(row, seed, args.load_num_proc)
            subpart.append(curr_dataset)
        

    logger.info("Starting to concatenate the datasets....")

    subpart = concatenate_datasets(subpart)

    logger.info("Finished concatenating..")

    logger.info("Starting to shuffle ....")

    #subpart = subpart.shuffle(seed=44, writer_batch_size=10000)

    logger.info("Finished Shuffling ....")




    return subpart



def save_as_jsonl(args, ds_name, dataset):
    ds_name = f"{ds_name}.jsonl"
    
    final_path = args.save_path / Path(ds_name)

    logger.info(f"Saving dataset to: {final_path}")

    dataset.to_json(path_or_buf=final_path, lines=True, batch_size=10000, num_proc=args.save_num_proc)






def main():
    args = parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    config = read_configuration(args)

    # Random generator
    seed = SeedSequence(42)

    dataset = process_all(args, seed)
    
    # configure the save name
    if args.subset == "yes":
        dataset_name = "sc_loss"
    elif args.subset == "no":
        dataset_name = "no_sc_loss"
    else:
        print("Something wrong in argument --subset")
        sys.exit()
    
    logger.info("Starting to save jsonl...")

    # Starting to save
    save_as_jsonl(args, dataset_name, dataset)



if __name__ == "__main__":
    main()
