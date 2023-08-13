import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import jsonlines
from utils.utils import load_jsonl
from utils.perspective_api import PerspectiveWorker, unpack_scores
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER


def save_as_jsonl(file_path, string_list):
    with jsonlines.open(file_path, mode='w') as writer:
        for string in string_list:
            writer.write({'text': string})


def read_jsonl(file_path):
    string_list = []
    with jsonlines.open(file_path, mode='r') as reader:
        for item in reader:
            if 'text' in item:
                string_list.append(item['text'])
    return string_list

def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        if response['response']:
            response = unpack_scores(response['response'])[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


def collate(dataset, generations, responses, output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)




def evaluate(args):
    model_file_name = args.model_name.split("/")[-1]
    
    # Create output files
    output_dir = f"{args.output_dir}/{model_file_name}"
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    
    # Folders
    generations_file = output_dir / 'generations.jsonl'
    perspective_file = output_dir / 'perspective.jsonl'

    generations_file = output_dir / 'generations.jsonl'
    perspective_file = output_dir / 'perspective.jsonl'
    output_file = output_dir / f'prompted_gens.jsonl'

    
    # Generate and collate perspective scores
    generations = read_jsonl(args.generations_file)

    # Create perspective worker thread
    perspective = PerspectiveWorker(out_file=perspective_file,
                                    total=len(generations),
                                    rate_limit=args.perspective_rate_limit)
    




    counter = 0

    for gen in generations:
        perspective(f'generation-{counter}', gen)
        
        counter += 1

    
    perspective.stop()
    print('Finished generation and perspective scoring!')

    print('Collating output files')
    collate(None, generations, load_jsonl(perspective_file), output_file)


"""

python evaluate_toxicity_gen.py \
      --model_name davidvblumenthal/1.4B-GPT-Verite_no_sc \
      --output_dir /Users/davidblumenthal/Documents/Master_Thesis/toxicity_prompts/evaluations/ \
      --perspective_rate_limit 5 \
      --generations_file /Users/davidblumenthal/Documents/Master_Thesis/toxicity_prompts/evaluations/1.4B-GPT-Verite_no_sc/generations.jsonl
      

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="Internal name of the model for saving the output", type=str)
    parser.add_argument("--output_dir", type=str, help="Path to save output")
    parser.add_argument('--perspective_rate_limit', type=int, default=59)

    parser.add_argument("--generations_file", type=str)

    
    args = parser.parse_args()

    evaluate(args)
    





