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

    if config_row["ratio"] == 2:
        
        dataset_duplicate = dataset
        dataset = concatenate_datasets([dataset, dataset_duplicate])


    
    # Append some meta columns
    category = [config_row["category"]] * len(dataset)
    task = [config_row["task"]] * len(dataset)
    name = [config_row["name"]] * len(dataset)

    dataset = dataset.add_column("category", category)
    dataset = dataset.add_column("task", task)
    dataset = dataset.add_column("name", name)

    
    return dataset


def process_all(args, seed):

    sc_loss = []
    no_sc_loss = []

    # get dataset config
    config = read_configuration(args)

    for index, row in config.iterrows():
        log_name = row["name"]
        logger.info(f"Processing: {log_name}")

        curr_dataset = process_single_dataset(row, seed, args.load_num_proc)

        if row["sc_loss"] == "yes":
            sc_loss.append(curr_dataset)
        elif row["sc_loss"] == "no":
            no_sc_loss.append(curr_dataset)
        
        else:
            sys.exit('There is probably a mapping missing in the "sc_loss" column')

    logger.info("Starting to concatenate the datasets....")

    sc_loss = concatenate_datasets(sc_loss)
    no_sc_loss = concatenate_datasets(no_sc_loss)

    logger.info("Finished concatenating..")

    logger.info("Starting to shuffle ....")

    """
    # Shuffle -> using iterable is faster 
    iterable_sc_loss = sc_loss.to_iterable_dataset(num_shards=256)
    iterable_no_sc_loss = no_sc_loss.to_iterable_dataset(num_shards=256)
    

    iterable_sc_loss = iterable_sc_loss.shuffle(seed=42, buffer_size=1000)
    iterable_no_sc_loss = iterable_no_sc_loss.shuffle(seed=42, buffer_size=1000)

    sc_loss = iterable_sc_loss
    no_sc_loss = iterable_no_sc_loss
    """

    sc_loss = sc_loss.shuffle(seed=44, writer_batch_size=10000)
    no_sc_loss = no_sc_loss.shuffle(seed=44, writer_batch_size=10000)

    logger.info("Finished Shuffling ....")

    logger.info("Flattening indices from dataset: sc_loss ....")
    cache_path_sc = "/pfs/work7/workspace/scratch/ukmwn-les_faits/les_faits_final/hug_cache/sc_loss"
    sc_loss = sc_loss.flatten_indices(num_proc=args.save_num_proc)
    logger.info("Finished ....")

    logger.info("Flattening indices from dataset: no_sc_loss ....")
    cache_path_no_sc = "/pfs/work7/workspace/scratch/ukmwn-les_faits/les_faits_final/hug_cache/no_sc_loss"
    no_sc_loss = no_sc_loss.flatten_indices(num_proc=args.save_num_proc)
    logger.info("Finished ....")


    return [sc_loss, no_sc_loss]



def save_as_jsonl(args, ds_name, dataset):
    ds_name = f"{ds_name}.jsonl"
    
    final_path = args.save_path / Path(ds_name)

    logger.info(f"Saving dataset to: {final_path}")

    dataset.to_json(path_or_buf=final_path, lines=True, batch_size=1000, num_proc=1)






def main():
    args = parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    config = read_configuration(args)

    # Random generator
    seed = SeedSequence(42)

    datasets = process_all(args, seed)
    dataset_names = ["sc_loss", "no_sc_loss"]

    logger.info("Starting to save jsonl...")

    for idx, (dataset, ds_name) in enumerate(zip(datasets, dataset_names)):
        logger.info(f"Saving {idx + 1}/{len(datasets)} | current: {ds_name}")
        # Actual saving
        save_as_jsonl(args, ds_name, dataset)


if __name__ == "__main__":
    main()
