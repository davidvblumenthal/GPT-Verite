from fastcoref import spacy_component
import spacy

from copy import deepcopy


def coref_generations(generations: list, model_type="large"):
    # Load Spacy
    # Setting up Spacy
    print("Setting up Spacy ...")
    spacy.require_gpu()
    nlp = spacy.load(
        "en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]
    )

    if model_type == "large":
        nlp.add_pipe(
            "fastcoref",
            config={
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": "cuda",
            },
        )
    elif model_type == "small":
        nlp.add_pipe("fastcoref")

    else:
        print(f"Model Type: {model_type} not implemented")

    generations = deepcopy(generations)
    # Get all generations into a list
    texts = [sample["text"] for sample in generations]
    # Process the generations
    docs = nlp.pipe(texts, component_cfg={"fastcoref": {"resolve_text": True}})

    # //TODO extract resolved texts from docs
    texts = [doc._.resolved_text for doc in list(docs)]

    for sample, text in zip(generations, texts):
        sample["text"] = text

    return generations


"""
for doc in nlp.pipe(small,
                    component_cfg={"fastcoref": {'resolve_text': True}}, 
                    n_process=1, 
                    batch_size=batch_size):
                                    
    #assert doc.has_annotation("TAG")
    assert doc._.trf_data == None
    # do some other processing, but don't save the whole doc
    results.append(doc._.resolved_text)
"""
