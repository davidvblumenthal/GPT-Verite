from fastcoref import spacy_component
import spacy

from copy import deepcopy

# Load Spacy
nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
nlp.add_pipe("fastcoref",
    config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
    )






def coref_generations(generations: list):
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
doc = nlp(, component_cfg={"fastcoref": {'resolve_text': True}})
cor_chunk = doc._.resolved_text

def resolve_text(input_text):

    text = input_text['text']

    docs = nlp.pipe(
                    texts, 
                    component_cfg={"fastcoref": {'resolve_text': True}}
                    )
"""