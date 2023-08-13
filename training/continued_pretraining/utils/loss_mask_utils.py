import time

import nltk


from datasets import load_dataset
from datasets.utils import disable_progress_bar

from itertools import groupby


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""



"""
    Helper function that turns list of strings to one string
"""
def flatten_str(l):
    s = ''.join(l)
    return s


"""
    Helper function that flattens a list of lists
"""
def flatten(l):
    return [item for sublist in l for item in sublist]


"""
    Helper function that chunks a long list into sublists with max length
        Custom truncation function

        Takes a list containing conc tokens and returns:
        sublists with a maximum len of n and minimum lenght of m
"""
def chunks(lst, n, m=1500):
    results = []
    for i in range(0, len(lst), n):
        if len(lst[i:i + n]) > m:
            results.append(lst[i:i + n])

    return results


"""
    Helper function that reads in jsonl file and returns a HuggingFace
    Dataset
"""
def get_dataset(jsonl_path: str, sample=False):
    # read jsonl files as HuggingFace datasets
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    if sample:
        dataset = dataset.select(range(5))

    return dataset



def insert_special_tokens_single(sentence: str, end_sentence_token='<|SENT_END|>') -> str:
    # append  |<SENT_END>| token
    sentence += end_sentence_token

    return sentence


def reconstruct_document(doc_sentences: list) -> str:
    return(flatten_str(doc_sentences))



# Initialize
splitter = nltk.load("tokenizers/punkt/english.pickle")
splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(train_text = splitter._params,
                                                      lang_vars = CustomLanguageVars())
def preprocess_loss_mask(documents: list) -> list:
    
    resulting_documents = []
    
    for doc in documents:
        doc_sentences = []
        for sent in splitter.tokenize(doc):
            # append special token at the end of the sentence
            sent = insert_special_tokens_single(sent, '<|SENT_END|>')
            # append result
            doc_sentences.append(sent)

        # reconstruct document; 
        # append to final result list
        resulting_documents.append(reconstruct_document(doc_sentences))


    return resulting_documents


def split_ids_at_SENT_END_token(token_ids: list, special_token_id=50257, implementation='itertools') -> list:
   
   if implementation == 'itertools':
    sublists = [list(group) for key, group in groupby(token_ids, lambda x: x == special_token_id) if not key]
    
    return sublists
   
   else:
    sublists = [token_ids[i:j] for i, j in zip([0]+[i+1 for i, x in enumerate(token_ids) if x == special_token_id], [i for i, x in enumerate(token_ids) if x == special_token_id]+[None])]

    return sublists


def construct_loss_mask(token_ids_sublists: list, multiple=2) -> list:
    doc_loss_mask = []
    doc_token_ids = []
    # iterate over the sublists, -> they represent one sentence
    for sentence in token_ids_sublists:
        # Get index where to split
        split_idx = len(sentence) // 2 # // -> floor division operator rounds down to the nearest integer
        
        # Different case for even and uneven length
        if len(sentence) % 2 == 0:
            # exactly half so length will match
            # loss_mask is first half
            loss_mask = [1] * split_idx
            sh_loss_mask = [multiple] * split_idx
        
        else:
            # due to floor division one idx would get lost
            loss_mask = [1] * split_idx
            sh_loss_mask = [multiple] * (split_idx + 1)
        
        # get the final loss mask by extending the two lists together
        loss_mask.extend(sh_loss_mask)

        assert len(sentence) == len(loss_mask), "loss_mask and sentence should have same length"

        doc_loss_mask.extend(loss_mask)
        doc_token_ids.extend(sentence)


    return doc_token_ids, doc_loss_mask