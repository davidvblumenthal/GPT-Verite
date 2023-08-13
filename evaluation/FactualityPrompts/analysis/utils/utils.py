import jsonlines


def write_jsonl(path, list_of_dicts):
    with jsonlines.open(path, "w") as writer:
        for sample in list_of_dicts:
            writer.write(sample)
