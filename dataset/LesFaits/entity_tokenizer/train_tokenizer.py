from datasets import concatenate_datasets
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers, processors

import os
import logging
import argparse
import jsonlines

set_verbosity_info()
logger = logging.getLogger(__name__)


def read_entities(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    
    return data


def construct_corpus(file_paths):

    datasets = []

    for file_path in file_paths:

        name = file_path.split("/")[-1].replace(".jsonl", "")
        logger.info(f"Getting dataset: {name}")

        dataset = load_dataset("json", data_files=file_path, split="train")#, streaming=True)
        datasets.append(dataset)

    # construct the corpus with iterleave
    logger.info("Starting to merge subsets using interleave!!")
    logger.info(f"Number of Datasets loaded: {len(datasets)}")

    print(datasets)
    
    #final_corpus = interleave_datasets(datasets)
    final_corpus = concatenate_datasets(datasets)

    del datasets

    return final_corpus


def get_batches(final_corpus):
    """
        returns 1000 examples at a time
    """
    for i in final_corpus:
        yield i["text"]


def batch_iterator(dataset, batch_size=1000):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]    
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
    """


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


def build_gpt_verite_tokenizer(path_entity_special_tokens):
    # Build own tokenizer
    new_tokenizer = Tokenizer(models.BPE(fuse_unk=False))
    new_tokenizer.normalizer = normalizers.NFC()
    new_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
    new_tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False, trim_offsets=True)
    new_tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=True)
    
    # Add and construct the special tokens
    special_tokens = read_entities(path_entity_special_tokens)
    special_tokens = ["<|endoftext|>", "<|padding|>"] + special_tokens
    
    # Vocab size
    vocab_size = 50304
    # Build trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
        )
    
    return new_tokenizer, trainer


def main():
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", required=True, type=str, help="Where to save the tokenizer."
    )
    parser.add_argument(
        "--dataset_dir",
        help="path to where the jsonl files are located",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--path_entity_special_tokens",
        help="Path to the selected entities to add as tokens.",
        required=True,
        type=str
    )

    parser.add_argument(
        "--train_new",
        help="Train with new configured tokenizer. Else Train new from GPT2 configuration",
        action="store_true"
    )

    args = parser.parse_args()

    # get the paths of all datafiles
    logger.info("Getting the file paths....")
    file_paths = get_file_paths(args.dataset_dir)

    # Construct the corpus
    logger.info("Constructing the corpus....")
    corpus = construct_corpus(file_paths)
    logger.info(f"Final corpus constructed. Number of samples: {len(corpus)}")

    # Get the batches to train the tokenizer
    #training_corpus = get_batches(corpus)
    if args.train_new:
        logger.info("Building GPT-Vérité from new config...")

        tokenizer, trainer = build_gpt_verite_tokenizer(args.path_entity_special_tokens)

        logger.info("Starting to train ...")
        tokenizer.train_from_iterator(batch_iterator(corpus), trainer=trainer)

        logger.info(f"Finished Training! Wrapping tokenizer and saving to: {args.save_dir} ...")

        wrapped_tokenizer = PreTrainedTokenizerFast(
            model_max_length=2048,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|padding|>",
            unk_token="<|endoftext|>",
            tokenizer_object=tokenizer
        )

        wrapped_tokenizer.save_pretrained(args.save_dir)

        logger.info("Saved! Now uploading to Huggingface!")
        wrapped_tokenizer.push_to_hub("GPT-Verite")


    """
    # Reuse the configuration from the gpt-2 tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")  #()  "EleutherAI/gpt-neox-20b"

    # Contruct the new vocabulary
    logger.info("Beginning to train the tokenizer....")
    tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(corpus), vocab_size=50_257)

    # Saving the resulting tokenizer
    logger.info(f"Saving tokenizer to: {args.save_dir}....")
    tokenizer.save_pretrained(args.save_dir)

    logger.info("Finished")
    """

"""

python train_tokenizer.py --dataset_dir ../../../../staging_area/train \
                          --save_dir ../../../../gpt-vérité_tokenizer

"""

if __name__ == "__main__":
    main()