import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--doc_length_dir", type=Path, required=True)
    parser.add_argument("--doc_num_tokens_dir", type=Path, required=True)
    
    parser.add_argument("--statistics-pickle-file_length", type=Path, required=True)
    parser.add_argument("--statistics-pickle-file_tokens", type=Path, required=True)
    args = parser.parse_args()

    return args


def normalize_lang_codes(lang):
    # Normalise chinese languages, so that we only consider simplified and traditional chinese as the two chinese
    # languages
    if lang in ["zh", "zhs", "zh-cn"]:
        lang = "zhs"
    elif lang in ["zht", "zh-tw"]:
        lang = "zht"
    return lang



def get_datasets_per_task(dataset_dir):
    dataset_paths = list(dataset_dir.iterdir())

    """
    Python dictionary throws a KeyError if you try to get an item with a key that is not currently in the dictionary. 
    The defaultdict in contrast will simply create any items that you try to access (provided of course they do not exist yet)
    """
    dataset_paths_per_task = defaultdict(list)
    for dataset_path in dataset_paths:

        task = dataset_path.name.split("_")[0]
        print(f"Debugging - extracted task: {task}")

        dataset_paths_per_task[task].append(dataset_path)
    
    return dataset_paths_per_task


def get_datasets_per_lang(dataset_dir):
    dataset_paths = list(dataset_dir.iterdir())

    dataset_paths_per_lang = defaultdict(list)
    for dataset_path in dataset_paths:

        lang = dataset_path.name[len("cleaned_lm_") :].split("_")[0]
        lang = normalize_lang_codes(lang)

        dataset_paths_per_lang[lang].append(dataset_path)
    return dataset_paths_per_lang


def compute_stats_per_ds(dataset_path, task): #lang):
    ds = load_from_disk(str(dataset_path))
    data_points_list = ds["len"][:]
    return (
        np.mean(data_points_list),
        np.median(data_points_list),
        data_points_list,
        dataset_path.name.split("_", 1)[1] #dataset_path.name[len(f"cleaned_lm_{lang}_") :],
    )

def compute_stats_per_ds_tokens(dataset_path, task):
    ds = load_from_disk(str(dataset_path))
    data_points_list = ds["len"][:]
    return (
        np.mean(data_points_list),
        np.median(data_points_list),
        data_points_list,
        dataset_path.name.split("_", 1)[1]
    )

def main():
    args = get_args()
    dataset_dir = args.doc_length_dir
    #dataset_paths_per_lang = get_datasets_per_lang(dataset_dir)
    dataset_paths_per_task = get_datasets_per_task(dataset_dir)

    all_data_point = {}
    for task in dataset_paths_per_task.keys():
        print(f"Processing {task}")
        data_points = []
        for dataset_path in dataset_paths_per_task[task]:
            data_points.append(compute_stats_per_ds(dataset_path, task))
        all_data_point[task] = data_points

    with open(args.statistics_pickle_file_length, "wb") as handle:
        pickle.dump(all_data_point, handle, protocol=pickle.HIGHEST_PROTOCOL)

# --------------------------------------------------------------------------------- #
    dataset_dir = args.doc_num_tokens_dir

    all_data_point_tokens = {}
    for task in dataset_paths_per_task.keys():
        print(f"Processing {task}")
        data_points = []
        for dataset_path in dataset_paths_per_task[task]:
            data_points.append(compute_stats_per_ds_tokens(dataset_path, task))
        all_data_point_tokens[task] = data_points

    with open(args.statistics_pickle_file_tokens, "wb") as handle:
        pickle.dump(all_data_point_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
