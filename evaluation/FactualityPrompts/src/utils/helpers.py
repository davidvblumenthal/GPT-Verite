from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

"""
    index 0 = contradict, index 1 = neutral, index 2 = entail
"""


def remove_nee_element(lst):
    return lst[:1] + lst[2:]


def index_of_list_with_highest_value(lists, use_true_false=True):
    if use_true_false:
        lists = list(map(remove_nee_element, lists))

    max_list_index, max_value = max(enumerate(map(max, lists)), key=lambda x: x[1])
    for i, lst in enumerate(lists):
        if max_value in lst and i != max_list_index:
            return i
    return max_list_index


def is_complete_sentence(sentence):
    # Check if the last character of the sentence is a period, question mark or exclamation mark.
    return sentence[-1] in [".", "?", "!"]


def remove_repetitions(generation_text: str) -> str:

    # First split into individual sentences
    single_sentences = []
    for idx, sent in enumerate(sent_tokenize(generation_text)):
        # check if last sentence, if so delete it no complete sentence e.g. The first actor was the actor who played the role of the King of
        if idx == len(sent_tokenize(generation_text)) - 1:
            clean_sent = sent.strip()
            if is_complete_sentence(clean_sent):
                single_sentences.append(clean_sent)
        else:
            clean_sent = sent.strip()
            single_sentences.append(clean_sent)
    
    # Remove the duplicates
    dedubed_sentences = list(set(single_sentences))
    
    # Rejoin the dedubed sentences into on text/string
    dedubed_generation = " ".join(dedubed_sentences)


    return dedubed_generation

