import argparse
import re
import json
import pickle
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def read_jsonl_to_list(file_path):

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_corpus(filename, chunk_size=10000):
    with open(filename, 'r') as file:
        while True:
            chunk = file.readlines(chunk_size)
            if not chunk:
                break
            for line in chunk:
                yield json.loads(line)

def count_names_worker(document, names):
    text = document['text']
    name_counts = Counter()
    for name in names:
        pattern = r"{name}"
        count = len(re.findall(pattern, text))
        name_counts[name] += count
    return name_counts

def count_names(corpus_path, save_path, names):
    num_cpus = 20
    print(f"Number of cpus used: {num_cpus}")
    
    with tqdm(total=3012121, desc='Counting names') as pbar:
        with Pool(num_cpus) as pool:
            results = []
            for document in read_corpus(corpus_path):
                results.append(pool.apply_async(count_names_worker, (document, names)))
            pool.close()
            pool.join()
            name_counts = Counter()
            for r in results:
                name_counts += r.get()
                pbar.update(1)

    with open(save_path, 'wb') as file:
        pickle.dump(dict(name_counts), file)

"""

python count_entities.py --corpus_path /pfs/work7/workspace/scratch/ukmwn-les_faits/staging_area/train/glm_wikipedia.jsonl \
        --save_path ./freq_entities.pickle \
        --entities ./entity_trie_original_strings.jsonl

"""


def main():
    parser = argparse.ArgumentParser(description='Count the occurrences of names in a corpus.')
    parser.add_argument('--corpus_path', type=str, help='The path to the corpus file.')
    parser.add_argument('--save_path', type=str, help='The path to save the name counts as a pickle file.')
    parser.add_argument('--entities', type=str)
    args = parser.parse_args()

    entities = read_jsonl_to_list(args.entities)

    count_names(args.corpus_path, args.save_path, entities)

if __name__ == '__main__':
    main()
