import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import jsonlines

from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from utils.perspective_api import PerspectiveWorker, unpack_scores
from utils.utils import create_toxicity_prompts, load_jsonl
from generation.generation_pipeline import huggingface_model_pipeline
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER


def save_as_jsonl(file_path, string_list):
    with jsonlines.open(file_path, mode='w') as writer:
        for string in string_list:
            writer.write({'text': string})

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


    # Construct the dataset
    prompts = create_toxicity_prompts()

    # Get pipeline

    pipe = huggingface_model_pipeline(args.max_len, args.model_name, args.trained_with_padding)

    # Create perspective worker thread
    #perspective = PerspectiveWorker(out_file=perspective_file,
    #                                total=len(prompts),
    #                                rate_limit=args.perspective_rate_limit)
    



    # Generate and collate perspective scores
    generations = []
    counter = 0

    for out, prompt in tqdm(zip(pipe(KeyDataset(prompts, "prompt"), return_full_text=False), prompts)):
        gen = out[0]["generated_text"]
        generations.append(gen)
        #generations["prompt"].append(prompt["prompt"])
        #perspective(f'generation-{counter}', gen)
        
        counter += 1

    
    save_as_jsonl(generations_file, generations)
    
    #perspective.stop()
    print('Finished generation and perspective scoring!')

    print('Collating output files')
    #collate(None, generations, load_jsonl(perspective_file), output_file)


"""

python evaluate_toxicity.py \
      --model_name davidvblumenthal/GPT-Verite_1.4B \
      --output_dir /home/kit/stud/ukmwn/master_thesis/evaluation/toxicity_prompts/evaluations \
      --perspective_rate_limit 30 \
      --max_len 50 \
      --trained_with_padding
      

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="Internal name of the model for saving the output", type=str)
    parser.add_argument("--output_dir", type=str, help="Path to save output")
    parser.add_argument('--perspective_rate_limit', type=int, default=25)

    parser.add_argument("--max_len", type=int, default=20, help="Number of Tokens to generate")
    
    parser.add_argument("--trained_with_padding",
                        action="store_true",
                        help="Model trained with dedicated paddig token or not"
                        )
    
    args = parser.parse_args()

    evaluate(args)
    





