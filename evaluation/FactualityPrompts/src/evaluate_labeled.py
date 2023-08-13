import argparse

import spacy
from nltk.corpus import stopwords

from utils.read_write_utils import read_jsonl, write_jsonl
from utils.labeled_utils import calc_metrics_per_generation_list
from utils.labeled_utils import calc_metrics_all


# Initialise Libraries
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


def main(args):
    # Read in labeled samples
    label_objects = read_jsonl(args.label_file)

    # Calculate metrics for all generations
    list_results = calc_metrics_per_generation_list(
        label_objects, calc_type=args.calc_type
    )

    # Pass result_list of results per generation
    final_result = calc_metrics_all(list_results)

    print(final_result)

    # Save to disk
    save_suffix = f"_{args.calc_type}_result.jsonl"
    save_path = args.label_file.replace(".jsonl", save_suffix)

    write_jsonl(save_path, [final_result.__dict__])


"""

python evaluate_labeled.py --label_file ../../joined_labels.jsonl --calc_type full
python evaluate_labeled.py --label_file ../../labeled_test-set-gen.jsonl --calc_type full

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label_file",
        type=str,
        help="path to the file containing labeled examples",
    )

    parser.add_argument(
        "--calc_type",
        type=str,
        default="full",
        help="How to calculate the metrics ['full', 'single']. Full generation or only first sentence",
    )

    args = parser.parse_args()

    # Run main method
    main(args)
