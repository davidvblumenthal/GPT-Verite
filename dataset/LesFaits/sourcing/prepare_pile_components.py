from datasets import load_dataset
from datasets import load_from_disk
from datasets.utils import disable_progress_bar
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import Sequence


import zstandard
import tarfile
import pathlib

import argparse
import jsonlines, json


def prepare_phil_papers(args):
    path = args.dataset_path
    save_path = args.save_dir
    num_samples = 0
    filtered_list = []
    with jsonlines.open(path) as input:
        for line in input:
            num_samples += 1
            if line["meta"]["language"] == "en" or line["meta"]["language"] == "uk":
                line = {"title": line["meta"]["title"], "text": line["text"]}
                filtered_list.append(line)
    print(f"Number samples before filtering: {num_samples}")
    print(f"\nNumber samples after filtering: {len(filtered_list)}")

    with jsonlines.open(path, "w") as writer:
        for sample in filtered_list:
            writer.write(sample)


def prepare_pubmed_abstracts(args):
    path = args.dataset_path
    save_path = args.save_dir

    num_samples = 0
    num_samples_after = 0

    with jsonlines.open(save_path, "w") as writer:
        with jsonlines.open(path) as input: 
            for line in input:
                num_samples += 1
                if line["text"] != "" and line["meta"]["language"] == "eng":
                    num_samples_after += 1
                    line = {"text": line["text"]}
                    writer.write(line)

    print(f"Number of samples initial: {num_samples}")
    print(f"\nNumber of samples after filtering: {num_samples_after}")



def prepare_free_law(args):
    path = args.dataset_path
    save_path = args.save_dir

    num_samples = 0
    num_samples_after = 0

    with jsonlines.open(save_path, "w") as writer:
        with jsonlines.open(path) as input:
            for line in input:
                num_samples += 1
                if line["text"] != "":
                    num_samples_after += 1
                    line = {"text": line["text"]}
                    writer.write(line)

    print(f"Number of samples initial: {num_samples}")
    print(f"\nNumber of samples after filtering: {num_samples_after}")


def decompress_zstandard_to_folder(args):
    input_file = args.dataset_path
    destination_dir = args.save_dir

    input_file = pathlib.Path(input_file)
    with open(input_file, "rb") as compressed:
        decomp = zstandard.ZstdDecompressor()
        output_path = pathlib.Path(destination_dir) / input_file.stem
        with open(output_path, "wb") as destination:
            decomp.copy_stream(compressed, destination)

def decompress_tar_gz(args):
    input_file = args.dataset_path
    destination_dir = args.save_dir

    if input_file.endswith("tar.gz"):
        tar = tarfile.open(input_file, "r:gz")
        tar.extractall(path=destination_dir)
        tar.close()
    elif input_file.endswith("tar"):
        tar = tarfile.open(input_file, "r:")
        tar.extractall(path=destination_dir)
        tar.close()


task_mapping = {
    "decompress_zst": decompress_zstandard_to_folder,
    "decompress_tar": decompress_tar_gz,
    "prepare_phil": prepare_phil_papers,
    "prepare_free_law": prepare_free_law,
    "prepare_pubmed_abstracts": prepare_pubmed_abstracts
}

"""
Unpack .zst file
python prepare_pile_components.py --task decompress_zst --dataset_path ../PUBMED_title_abstracts_2019_baseline.jsonl.zst --save_dir ../
python prepare_pile_components.py --task decompress_tar --dataset_path ../PMC_extracts.tar.gz --save_dir ../pubmed_central

Prepare datasets

python prepare_pile_components.py --task prepare_pubmed_abstracts --dataset_path ../PUBMED_title_abstracts_2019_baseline.jsonl --save_dir ../pubmed_abstracts.jsonl

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        "--task", type=str, help="Type of task to do e.g decompress zst"
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")

    parser.add_argument("--save_dir", type=str, help="path to save directory")

    parser.add_argument(
        "--num_proc", type=int, default=1, help="Number of processes to use"
    )

    args = parser.parse_args()

    task_mapping[args.task](args)
