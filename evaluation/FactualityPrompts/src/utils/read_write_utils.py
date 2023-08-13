import json
import jsonlines

from .const import HOME_DIR, GEN_DIR
from .helpers import remove_repetitions


def construct_paths(args):
    prompt_type = args.prompt_type

    # Construct path to prompts
    prompt_path = "{}/prompts/fever_{}_final.jsonl".format(HOME_DIR, prompt_type)

    if args.gen_path != None:
        gen_path = "{}/{}".format(GEN_DIR, args.gen_path)
    else:
        print("No generation path provided. Using template based path")
        exit(0)

    ### added ###
    if args.gen_dir != None:
        gen_path = "{}/{}/{}".format(GEN_DIR, args.gen_dir, args.gen_path)

    return gen_path, prompt_path


def read_prompts_and_generations(prompt_path: str, gen_path: str, dedubed=True):
    # Read in the Prompts and the Generations
    prompts, gens = [], []

    with open(prompt_path, "r") as infile:
        for line in infile:
            fever_obj = json.loads(line.strip())
            prompts.append(fever_obj)

    with open(gen_path, "r") as infile:
        for line in infile:
            gen_obj = json.loads(line.strip())
            
            if dedubed:
                gen_obj["text"] = remove_repetitions(gen_obj["text"])

                #print(gen_obj)
                gens.append(gen_obj)
            else:
                gens.append(gen_obj)

    return prompts, gens


def write_jsonl(path, list_of_dicts):
    with jsonlines.open(path, "w") as writer:
        for sample in list_of_dicts:
            writer.write(sample)


def append_jsonl(path, list_of_dicts):
    with jsonlines.open(path, "a") as writer:
        for sample in list_of_dicts:
            writer.write(sample)


def read_jsonl(path: str) -> list:
    samples = []
    with jsonlines.open(path) as input:
        for line in input:
            samples.append(line)

    return samples


def read_addtional_ne(args):
    # Construct path to prompts
    path = "{}/crawl_wikipedia/{}_wiki_crawl_imp_ne.jsonl".format(HOME_DIR, args.prompt_type)
    
    return read_jsonl(path)
