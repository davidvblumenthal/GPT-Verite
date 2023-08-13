from datasets import load_dataset
from datasets import load_from_disk
from datasets.utils import disable_progress_bar
from datasets import Dataset
from datasets import concatenate_datasets

import random
import argparse
import os
import requests

from utils.qa_utils import construct_extractive_qa_sample, contruct_closed_book_qa


dataset_mapping = {
    "ted_talks": "bigscience-data/roots_en_ted_talks_iwslt",
    "eli5_category": "eli5_category",
    "eli5": "Pavithree/eli5",
    "mrqa": "mrqa",
    "uspto": "bigscience-data/roots_en_the_pile_uspto",
    "europarl": "bigscience-data/roots_en_the_pile_europarl",
    "united_nations_corpus": "bigscience-data/roots_en_uncorpus",
    "pubmed_abstracts": "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst",
    "pubmed_central": "https://the-eye.eu/public/AI/pile_preliminary_components/PMC_extracts.tar.gz",
    "wikibooks": "bigscience-data/roots_en_wikibooks",
    "wikiquotes": "bigscience-data/roots_en_wikiquote",
    "books1": "https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz",
    "ag_news": "ag_news",
    "paws": "paws",
    "dbpedia14": "dbpedia_14",
    "wikihow": "wikihow",
    "big_patent": "big_patent",
    "scitldr": "allenai/scitldr",
    "qnli": "SetFit/qnli",
    "lfqa": "LLukas22/lfqa_preprocessed",
    "books3": "the_pile_books3"
}

def get_qnli(args):
    dataset = load_dataset(dataset_mapping["qnli"])

    # test split is not labeled, hence not needed
    

    def create_prompt(sample):
        
        premise = sample["text2"]
        question = sample["text1"]
        
        if sample["label"] == 1:
            label = "no"
        elif sample["label"] == 0:
            label = "yes"
        else:
            print(sample["label"])
            print("Label-error!!!!!!!!!!")
        
        
        text = f"Premise: {premise};\n\nHypothesis: {question};\n\nDoes this premise entail the hypothesis?\n\nAnswer: {label}"

        return {"text": text}


    
    save_dir = args.save_dir

    for split in ["train", "validation"]:
        temp_dataset = dataset[split]
        temp_dataset = temp_dataset.map(create_prompt, num_proc=1)
        temp_dataset = temp_dataset.remove_columns([column for column in temp_dataset.column_names if column != "text"])

        dataset_name = f"qnli_{split}.jsonl"    
        save_path = os.path.join(save_dir, dataset_name)

        print(f"Constructing {split} split!!!")
                  
        temp_dataset.to_json(path_or_buf=save_path, lines=True)


def get_books3(args):

    dataset = load_dataset(dataset_mapping["books3"])

    save_dir = args.save_dir

    dataset_name = "books3.jsonl"    
    save_path = os.path.join(save_dir, dataset_name)

    print(dataset)
    dataset = dataset["train"]
    dataset.to_json(path_or_buf=save_path, lines=True, num_proc=1)






def get_scitldr(args):
    configs = ["AIC", "Abstract", "FullText"]
    

    def construct(sample):
        source = sample["source"]
        source = " ".join(source)

        target = sample["target"]
        target = " ".join(target)

        text = f"Summarise this text: {source}; \n\nSummary: {target}"

        return {"text": text}



    save_dir = args.save_dir

    for split in ["train", "test", "validation"]:
        temp_datasets = []
        
        for config in configs:
            
            dataset = load_dataset(dataset_mapping["scitldr"], config)
            dataset = dataset[split]
            dataset = dataset.remove_columns([column for column in dataset.column_names if column != "source" and column != "target"])
            print(f"Dataset: {dataset}")

            temp_datasets.append(dataset)
        
        print(f"List of datasets: {temp_datasets}")
        
        dataset = concatenate_datasets(temp_datasets)


        dataset = dataset.map(construct, num_proc=1)
        dataset = dataset.remove_columns([column for column in dataset.column_names if column != "text"])
        
        dataset_name = f"scitldr_{split}.jsonl"    
        save_path = os.path.join(save_dir, dataset_name)

        print(f"Constructing {split} split!!!")
                  
        dataset.to_json(path_or_buf=save_path, lines=True)






def get_big_patent(args):

    dataset = load_dataset(dataset_mapping["big_patent"], "all")

    def construct_summary_version(sample):
        text = sample["description"]
        summary = sample["abstract"]

        # construct the text
        text = f"Summarise this text: {text}; \n\nSummary: {summary}"

        return {"text": text}

    def construct_corpus_version(sample):
        
        return {"text": sample["description"]}

    
    save_dir = args.save_dir
    
    for version in ["corpus_version", "summary_version"]:
        for split in ["train", "validation", "test"]:
            print(f"Contructing {version}..split {split}... ")
            
            dataset_name = f"big_patent_{version}_{split}.jsonl"
            
            save_path = os.path.join(save_dir, dataset_name)

            if version == "corpus_version":
                temp_dataset = dataset[split].map(construct_corpus_version, remove_columns=dataset[split].column_names)
            if version == "summary_version":
                temp_dataset = dataset[split].map(construct_summary_version, remove_columns=dataset[split].column_names)
            
            temp_dataset.to_json(path_or_buf=save_path, lines=True)
    




def get_wikihow(args):
    dataset = load_dataset(dataset_mapping["wikihow"], "all", data_dir="/pfs/work7/workspace/scratch/ukmwn-les_faits/wikihow")

    def contruct_qa_version(sample):
        question = sample["title"]
        answer = sample["text"]

        # sometimes question ends with a number
        if question[-1].isdigit():
            question = question[:-1]
        
        # append question mark
        question = question + "?"
        # construct text
        text = f"{question} \n\nAnswer: {answer}"

        return {"text": text}

    def contruct_corpus_version(sample):
        title = sample["title"]
        body = sample["text"]

        # sometimes title ends with a number
        if title[-1].isdigit():
            title = title[:-1]

        text = f"{title} \n\n{body}"

        return {"text": text}

    dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

    save_dir = args.save_dir
    
    for version in ["corpus_version", "qa_version"]:
        print(f"Contructing {version}..... ")
        dataset_name = "wikihow_{}.jsonl".format(version)
        save_path = os.path.join(save_dir, dataset_name)

        if version == "corpus_version":
            temp_dataset = dataset.map(contruct_corpus_version, remove_columns=dataset.column_names)
        if version == "qa_version":
            temp_dataset = dataset.map(contruct_qa_version, remove_columns=dataset.column_names)
        
        temp_dataset.to_json(path_or_buf=save_path, lines=True)





def get_dbppedia14(args):
    
    mapping = {
                0: "Company",
                1: "EducationalInstitution",
                2: "Artist",
                3: "Athlete",
                4: "OfficeHolder",
                5: "MeanOfTransportation",
                6: "Building",
                7: "NaturalPlace",
                8: "Village",
                9: "Animal",
                10: "Plant",
                11: "Album",
                12: "Film",
                13: "WrittenWork"
                }
    dataset = load_dataset(dataset_mapping["dbpedia14"])

    def turn_to_prompt(sample):

        content = sample["content"]
        label = sample["label"]

        string_label = mapping[label]

        text = f"What is the topic of this text? {content}; \n\nTopic: {string_label}"

        return {"text": text}

    sets = ["train", "test"]
    save_dir = args.save_dir
    
    for subset in sets:
        dataset_name = "dbpedia14_{}.jsonl".format(subset)
        save_path = os.path.join(save_dir, dataset_name)

        temp_dataset = dataset[subset].map(turn_to_prompt, remove_columns=dataset[subset].column_names)

        temp_dataset.to_json(path_or_buf=save_path, lines=True)



def get_paws(args):
    # Label 1 == is a paraphrase; Label 0 == not a paraphrase
    # labeled_final and labeled_swap include human judgment; unlabeled_final not
    dataset = load_dataset(dataset_mapping["paws"], "labeled_final")
    print(f"Dataset before filtering only paraphrases: {dataset}")
    dataset = dataset.filter(lambda sample: sample["label"] == 1)
    print(f"Dataset after filtering only paraphrases: {dataset}")
    
    dataset_train = load_dataset(dataset_mapping["paws"], "labeled_swap", split="train")

    dataset_train = concatenate_datasets([dataset["train"], dataset_train])
    print(f"Dataset Train before filtering only paraphrases: {dataset_train}")
    
    dataset_train = dataset_train.filter(lambda sample: sample["label"] == 1)
    print(f"Dataset after filtering only paraphrases: {dataset_train}")
    
    def remove_false_paraphrase(sample):

        sample = "Paraphrase this text: " + sample["sentence1"] + "; \n\nParaphrase: " + sample["sentence2"]

        return {"text": sample}
    
    #dataset = dataset.map(remove_false_paraphrase, remove_columns=dataset.column_names)
    dataset_train_new = dataset_train.map(remove_false_paraphrase, remove_columns=dataset_train.column_names)

    print(f"Datasetnew {dataset_train_new}")
    

    splits = ["train", "validation", "test"]

    save_dir = args.save_dir

    for subset in splits:
        dataset_name = "paws_{}.jsonl".format(subset)
        save_path = os.path.join(save_dir, dataset_name)
        
        if subset != "train":     
            
            temp_dataset = dataset[subset].map(remove_false_paraphrase, remove_columns=dataset[subset].column_names)

            temp_dataset.to_json(path_or_buf=save_path, lines=True)
        
        if subset == "train":
            dataset_train_new.to_json(path_or_buf=save_path, lines=True)






def get_ag_news(args):
    
    dataset = load_dataset(dataset_mapping["ag_news"])

    dataset = dataset.remove_columns(["label"])

    save_dir = args.save_dir
    
    sets = ["train", "test"]


    
    for subset in sets:
        dataset_name = "ag_news{}.jsonl".format(subset)
        save_path = os.path.join(save_dir, dataset_name)

        temp_dataset = dataset[subset]

        temp_dataset.to_json(path_or_buf=save_path, lines=True)
 


def get_wikiquotes(args):
    dataset = load_dataset(dataset_mapping["wikiquotes"])

    dataset = dataset["train"]
    dataset = dataset.remove_columns("meta")

    # save dataset as jsonl to disk
    # Get save_path
    save_dir = args.save_dir
    dataset_name = "wikiquotes.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    dataset.to_json(path_or_buf=save_path, lines=True)


def get_wikibooks(args):
    dataset = load_dataset(dataset_mapping["wikibooks"])

    dataset = dataset["train"]
    dataset = dataset.remove_columns("meta")

    # save dataset as jsonl to disk
    # Get save_path
    save_dir = args.save_dir
    dataset_name = "wikibooks.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    dataset.to_json(path_or_buf=save_path, lines=True)


def get_ted_talks(args):
    dataset = load_dataset(dataset_mapping["ted_talks"])

    dataset = dataset["train"]
    dataset = dataset.remove_columns("meta")

    # save dataset as jsonl to disk
    # Get save_path
    save_dir = args.save_dir
    dataset_name = "ted_talks.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    dataset.to_json(path_or_buf=save_path, lines=True)

def get_pubmed_abstracts(args):
    filename = dataset_mapping["pubmed_abstracts"].split("/")[-1]
    filename = args.save_dir + filename
    
    with requests.get(dataset_mapping["pubmed_abstracts"], stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def get_pubmed_cental(args):
    filename = dataset_mapping["pubmed_central"].split("/")[-1]
    filename = args.save_dir + filename
    
    with requests.get(dataset_mapping["pubmed_central"], stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def get_books1(args):
    filename = dataset_mapping["books1"].split("/")[-1]
    filename = args.save_dir + filename
    
    with requests.get(dataset_mapping["books1"], stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)    


def get_eli5_category(args):
    dataset = load_dataset(dataset_mapping["eli5_category"])

    dataset = dataset.remove_columns(
        ["q_id", "selftext", "category", "subreddit", "title_urls", "selftext_urls"]
    )

    dataset = dataset.rename_column("title", "question")
    #dataset = dataset.rename_column("answers", "answer")
    # Only use the first answers -> highest rating
    dataset = dataset.map(lambda sample: {"answer": sample["answers"]["text"][0]}, remove_columns=["answers"])

    dataset = dataset.map(contruct_closed_book_qa)

    dataset = dataset.remove_columns(["question", "answer"])

    dataset_train = dataset["train"]
    dataset_val = concatenate_datasets([dataset["validation1"], dataset["validation2"]])
    dataset_test = dataset["test"]

    save_paths = []

    for dataset_sub in ["train", "validation", "test"]:
        dataset_name = f"eli5_category_{dataset_sub}.jsonl"

        save_paths.append(os.path.join(args.save_dir, dataset_name))

    
    dataset_train.to_json(path_or_buf=save_paths[0], lines=True)
    dataset_val.to_json(path_or_buf=save_paths[1], lines=True)
    dataset_test.to_json(path_or_buf=save_paths[2], lines=True)


def get_eli5(args):
    dataset = load_dataset(dataset_mapping["eli5"])
    dataset = dataset.remove_columns(
        [
            "q_id",
            "selftext",
            "document",
            "subreddit",
            "url",
            "title_urls",
            "selftext_urls",
            "answers_urls",
        ]
    )



    save_dir = args.save_dir
    dataset_name = "eli5.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    print(f"Dataset values: {dataset}")

    dataset.to_json(path_or_buf=save_path, lines=True)


def get_lfqa(args):
    dataset = load_dataset(dataset_mapping["lfqa"])

    dataset = dataset.map(construct_extractive_qa_sample, num_proc=2)

    save_dir = args.save_dir
    for split in ["train", "validation"]:

        dataset_name = "lfqa{}.jsonl".format(split)
        save_path = os.path.join(save_dir, dataset_name)
        
        temp_dataset = dataset[split]
        temp_dataset = temp_dataset.remove_columns([column for column in temp_dataset.column_names if column != "text"])
        temp_dataset.to_json(path_or_buf=save_path, lines=True)

    

def get_mrqa(args):
    dataset = load_dataset(dataset_mapping["mrqa"])
    dataset = dataset.remove_columns(
        ["subset", "context_tokens", "qid", "question_tokens", "detected_answers"]
    )

    dataset = dataset.map(construct_extractive_qa_sample)
    dataset = dataset.remove_columns(["context", "answers", "question"])

    dataset_train = dataset["train"]
    dataset_val = dataset["validation"]
    dataset_test = dataset["test"]

    save_dir = args.save_dir
    
    dataset_name_train = "mrqa_train.jsonl"
    dataset_name_val = "mrqa_validation.jsonl"
    dataset_name_test = "mrqa_test.jsonl"
    
    save_path_t = os.path.join(save_dir, dataset_name_train)
    save_path_v = os.path.join(save_dir, dataset_name_val)
    save_path_test = os.path.join(save_dir, dataset_name_test)

    #print(f"Dataset values {dataset}")

    dataset_train.to_json(path_or_buf=save_path_t, lines=True)
    dataset_val.to_json(path_or_buf=save_path_v, lines=True)
    dataset_test.to_json(path_or_buf=save_path_test, lines=True)


def get_uspto(args):
    dataset = load_dataset(dataset_mapping["uspto"], split="train")
    dataset = dataset.remove_columns("meta")

    save_dir = args.save_dir
    dataset_name = "uspto.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    print(f"Dataset values {dataset}")

    dataset.to_json(path_or_buf=save_path, lines=True)


def get_europarl(args):
    dataset = load_dataset(dataset_mapping["europarl"], split="train")

    save_dir = args.save_dir
    dataset_name = "europarl.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    print(f"Dataset values {dataset}")

    dataset.to_json(path_or_buf=save_path, lines=True)


def get_united_nations_corpus(args):
    dataset = load_dataset(dataset_mapping["united_nations_corpus"], split="train")
    dataset = dataset.remove_columns("meta")

    save_dir = args.save_dir
    dataset_name = "uncorpus.jsonl"
    save_path = os.path.join(save_dir, dataset_name)

    print(f"Dataset values {dataset}")

    dataset.to_json(path_or_buf=save_path, lines=True)   


get_dataset_mapping = {
    "ted_talks": get_ted_talks,
    "eli5_category": get_eli5_category,
    "eli5": get_eli5,
    "mrqa": get_mrqa,
    "uspto": get_uspto,
    "europarl": get_europarl,
    "united_nations": get_united_nations_corpus,
    "pubmed_abstracts": get_pubmed_abstracts,
    "pubmed_central": get_pubmed_cental,
    "wikibooks": get_wikibooks,
    "wikiquotes": get_wikiquotes,
    "books1": get_books1,
    "ag_news": get_ag_news,
    "paws": get_paws,
    "dbpedia14": get_dbppedia14,
    "wikihow": get_wikihow,
    "bigpatent": get_big_patent,
    "scitldr": get_scitldr,
    "qnli": get_qnli,
    "books3": get_books3,
    "lfqa": get_lfqa
}

"""

    python get_datasets.py --datasets mrqa --single_dataset --save_dir ../new_qa_datasets

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("--datasets", type=str, help="datasets to use")
    parser.add_argument(
        "--single_dataset", action="store_true", help="Just get a single dataset"
    )

    parser.add_argument("--save_dir", type=str, help="path to save directory")

    args = parser.parse_args()

    if args.single_dataset:
        dataset_name = args.datasets

        print(f"Getting dataset: {dataset_name}")
        get_dataset_mapping[dataset_name](args)

        print("Finished!")
