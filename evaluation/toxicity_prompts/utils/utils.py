from datasets import load_dataset
from datasets import Dataset
import json
from tqdm import tqdm


def batchify(data, batch_size: int):
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch



def load_cache(file):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)

def load_jsonl(file):
    with open(file) as f:
        for line in f:
            yield json.loads(line)


# create dataset
def create_toxicity_prompts():
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Get the prompt
    prompts = {"prompt": []}
    for line in dataset["prompt"]:
        prompts["prompt"].append(line["text"])

    # Turn to HF ds
    dataset = Dataset.from_dict(prompts)

    return dataset
