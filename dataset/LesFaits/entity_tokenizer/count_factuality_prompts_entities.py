import jsonlines
import codecs
import json
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
import pickle


FAC_PATH = "/home/kit/stud/ukmwn/master_thesis/evaluation/FactualityPrompt/crawl_wikipedia/factual_wiki_crawl_imp_ne.jsonl"
NON_FACT_PATH = "/home/kit/stud/ukmwn/master_thesis/evaluation/FactualityPrompt/crawl_wikipedia/nonfactual_wiki_crawl_imp_ne.jsonl"


def save_as_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def read_jsonlines_file(file_path):
    """
    Reads in a JSONLines file and returns a list of Python dictionaries.
    
    Args:
        file_path (str): Path to the JSONLines file.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries, where each dictionary represents a single line
        from the JSONLines file.
    """
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data.append(line)
    return data


def clean_unicode_escape(item):
    """
    Cleans a list of strings from any Unicode escape codes.
    
    Args:
        lst (List[str]): List of strings to be cleaned.
        
    Returns:
        List[str]: List of cleaned strings.
    """

    cleaned_item = item
    cleaned_item = cleaned_item.replace('\u200b', '')
    # Remove zero-width spaces encoded as '\x80\x8b'
    cleaned_item = cleaned_item.replace('\x80\x8b', '')
    cleaned_item = cleaned_item.replace("the ", '')
        

    return cleaned_item


def count_entities_in_article(article, entities):
    counts = Counter()
    for entity in entities:
        counts[entity] += article.count(entity)
    return counts

def count_entities_in_articles(entities, articles, num_proc):
    print(f"Using {num_proc} procecces")
    with Pool(processes=num_proc) as pool:
        entity_counts = Counter()
        args = zip(articles, [entities]*len(articles))
        for counts in tqdm(pool.imap_unordered(_count_entities_in_article_unpack_args, args), total=len(articles)):
            entity_counts += counts
    return entity_counts

def _count_entities_in_article_unpack_args(args):
    return count_entities_in_article(*args)





"""

python count_factuality_prompts_entities.py \
       --save_path save_path \
       --num_proc 40

"""


def main():
    parser = argparse.ArgumentParser(description='Count the occurrences of names in a corpus.')
    parser.add_argument('--save_path', type=str, help='The path to save the name counts as a pickle file.')
    parser.add_argument('--num_proc', type=int)
    
    args = parser.parse_args()

    factual = read_jsonlines_file(FAC_PATH)
    non_factual = read_jsonlines_file(NON_FACT_PATH)

    entities = factual + non_factual

    cleaned_entities = []

    for sample in entities:
        for entity in sample["important_ne"]:
            
            cleaned_entity = clean_unicode_escape(entity)
            cleaned_entities.append(cleaned_entity)


    unique_clean_entities = list(set(cleaned_entities))


    all_articles = []

    for article in entities:
        
        for single_article in article["articles"]:
            all_articles.append(single_article["text"])

    
    del factual
    del non_factual
    del entities
    del cleaned_entities


    result = count_entities_in_articles(unique_clean_entities, all_articles, args.num_proc)

    save_as_pickle(result, args.save_path)


if __name__ == '__main__':
    main()