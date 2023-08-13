from datasets import load_dataset
import argparse





def prepend_title_to_text(args):

    dataset = load_dataset("json", data_files=args.datasets, split="train")

    dataset = dataset.map(lambda sample: {"text": sample["title"] + "\n\n" + sample["text"]}, remove_columns=["title"])

    dataset.to_json(path_or_buf=args.save_dir)

    print("Finished!!")



"""

python prepend_title_to_text.py --datasets ../../filter_test/glm_books3.jsonl \
          --save_dir ../../filter_test/books3_new.jsonl

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("--datasets", type=str, help="datasets to use")

    parser.add_argument("--save_dir", type=str, help="path to save directory")

    args = parser.parse_args()

    prepend_title_to_text(args)