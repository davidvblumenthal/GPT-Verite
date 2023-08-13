import re
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import spacy

# spacy.prefer_gpu()
spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


IMPORTANT_ENT_TYPE = set(
    ["ORG", "GPE", "PERSON", "WORK_OF_ART", "PRODUCT", "EVENT"]
)  # added GPE
REMOVE_ENT_TYPE = set(["ORDINAL", "CARDINAL"])
# Cardinal numbers tell 'how many' of something, they show quantity. Ordinal numbers tell the order of how things are set, they show the position or the rank of something.


def obtain_important_ne(gen, include_capitalized_words_as_ents=True):
    important_words = []

    doc = nlp(gen)

    # print("GEN: ", gen)
    # print([(token.text, token.pos_, token.tag_, token.dep_) for token in doc if token.pos_ in ['NOUN', 'PRON', 'PROPN']])
    # print("\n")

    ents = [(ent.text, ent.label_) for ent in doc.ents]

    if include_capitalized_words_as_ents and len(ents) == 0:
        capitalized_words = re.findall("(?<!^)([A-Z][a-z]+)", gen)

        if len(capitalized_words) > 0:
            capitalized_words = [
                (word, "CAPITALIZED")
                for word in capitalized_words
                if word.lower() not in stop_words
            ]
            ents.extend(capitalized_words)

    important_words.extend([ent for ent in ents if ent[1] in IMPORTANT_ENT_TYPE])
    remaining_ne_all = [ent for ent in ents if ent[1] not in IMPORTANT_ENT_TYPE]

    # filter out some ne
    remaining_ne = []
    for ent in remaining_ne_all:
        if ent[1] in REMOVE_ENT_TYPE:
            continue
        if ent[1] == "DATE" and (
            "year" in ent[0] or "day" in ent[0]
        ):  # not bool(re.search(r'\d', ent[0])):
            # if "DATE" entity contains NO number at all (e.g., ``the year''), meaningless
            continue
        remaining_ne.append(ent)

    gens_with_ne = {
        "gen": gen,
        "important_ne": important_words,
        "unimportant_ne": remaining_ne,
        "subject": set(
            [token.text for token in doc if token.dep_ in ["nsubj", "nsubjpass"]]
        ),
        # "all_analysis": [(token.text, token.pos_, token.tag_, token.dep_) for token in doc]
    }

    return gens_with_ne


def has_incorrect_style(gen_obj):
    # case 1: contains first person -- I, we
    if gen_obj["subject"].intersection(set(["i", "I", "You", "you", "We", "we"])):
        return True

    # case 2: question?
    if "?" in gen_obj["gen"]:
        return True

    # case 3:

    # remove "my", "say"/"said", future ("will")

    # filter prompts that are too short --> Wales' population changed... this is so short

    return False


TYPES = {"NO_FACT": 1, "HAS_FACT": 2, "OFF_TOPIC": 3}


def identify_sentence_type(claim_obj, wiki_names_txt):
    """
    obj: LM generation object
    prompt_wiki_names (list): Wikipedia list from the FEVER dataset (evidence)

    """
    assigned_type = None

    # case 1: no facts -- i.e., no NE, incorrect_style, no SUBJECT
    if (
        len(claim_obj["important_ne"]) + len(claim_obj["unimportant_ne"]) == 0
        or has_incorrect_style(claim_obj)
        or len(claim_obj["subject"]) == 0
    ):
        assigned_type = TYPES["NO_FACT"]

    # case 2: no off-topic, but contains facts (unimportant_ne) about target-topic
    elif len(claim_obj["important_ne"]) == 0 and len(claim_obj["unimportant_ne"]) > 0:
        assigned_type = TYPES["HAS_FACT"]

    # case 3: tricky scenario. important_ne could be relevant to the target-topic, or could indicate off-topic
    else:
        # 1. filter out any extra_ne that is same as wikiname -- e.g., wiki_name = Barak Obama, ne = Obama
        extra_ne = [
            ne[0] for ne in claim_obj["important_ne"] if ne[0] not in wiki_names_txt
        ]

        # 2. check if any of the extra_ne is the "SUBJECT" of the generation
        overlap_between_extraNE_and_subj = claim_obj["subject"].intersection(
            set(" ".join(extra_ne).split(" "))
        )

        if len(overlap_between_extraNE_and_subj) > 0:  # contains off-topic NE!!
            assigned_type = TYPES["OFF_TOPIC"]
        else:
            assigned_type = TYPES["HAS_FACT"]

    return assigned_type


def check_assigned_types_single(gen_obj, prompt_obj, additional_ne=None):
    """
    gen_obj = dict of form
        {"prompt": str, "text": str}
    """
    # Get the text from the generation object
    gen_text = gen_obj["text"]

    # Get the main Named Entities from the prompts object
    """
        "evidence_info": [["Pilot (Supernatural)", "25017933"], ["Eric Kripke", "7399831"]]
    """

    wiki_names = [ev_infos[0] for ev_infos in prompt_obj["evidence_info"]]

    if additional_ne:
        wiki_names.extend(additional_ne["important_ne"])

    # Get the NE included in the corresponding WIKI article from the prompt["evidence_object"]
    wiki_names = " ".join(wiki_names)

    # Result list holding the tuples (sentence, type)
    sent_type_list = []

    for sent in sent_tokenize(gen_text):
        # Get Named Entities appearing in the sentence
        sent_with_ne = obtain_important_ne(sent)

        # Identify if sentence is check worthy or not
        """
            return: integer with the meaning:
                TYPES = {
                        'NO_FACT': 1,
                        'HAS_FACT': 2,
                        'OFF_TOPIC': 3
                        }
        """
        sent_type = identify_sentence_type(sent_with_ne, wiki_names)

        if sent_type == 2:
            sent_type_list.append((sent, "fact"))

        if sent_type == 1:
            sent_type_list.append((sent, "no_fact"))
        if sent_type == 3:
            sent_type_list.append((sent, "off_topic"))

    return {
        "prompt": gen_obj["prompt"],
        "text": gen_obj["text"],
        "single_sentences": sent_type_list,
    }


def check_assigned_types(list_gen_objs, list_prompts, additional_nes):
    result_list = []

    if additional_nes:
        for gen_obj, prompt_obj, additional_ne in tqdm(
            zip(list_gen_objs, list_prompts, additional_nes)
        ):
            result_list.append(
                check_assigned_types_single(gen_obj, prompt_obj, additional_ne)
            )

    else:
        for gen_obj, prompt_obj in tqdm(zip(list_gen_objs, list_prompts)):
            result_list.append(check_assigned_types_single(gen_obj, prompt_obj))

    return result_list


def obtain_trust_worthy_sents(text, wiki_names):
    wiki_names_txt = " ".join(wiki_names)

    text = text.strip().replace("\n", ". ")
    sents = sent_tokenize(text)

    sents_with_ne = [obtain_important_ne(sent.strip()) for sent in sents]

    no_fact_gen_cnt, no_fact_gens = 0, []
    checkworthy_gen_cnt, checkworthy_gens = 0, []
    off_topic_gen_cnt, off_topic_gens = 0, []

    for sent_obj in sents_with_ne:
        # case 1: no facts -- i.e., no NE, incorrect_style, no SUBJECT
        if (
            len(sent_obj["important_ne"]) + len(sent_obj["unimportant_ne"]) == 0
            or has_incorrect_style(sent_obj)
            or len(sent_obj["subject"]) == 0
        ):
            no_fact_gen_cnt += 1
            # no_fact_gens.append(sent_obj['gen'])
            # print("[NO FACT]", sent_obj['gen'])

        # case 2 v1: no off-topic, but contains facts (unimportant_ne) about target-topic
        elif len(sent_obj["important_ne"]) == 0 and len(sent_obj["unimportant_ne"]) > 0:
            checkworthy_gen_cnt += 1
            checkworthy_gens.append(sent_obj)
            # print('[CHECK-WORTHY]', sent_obj['gen'])

        # case 3: tricky scenario. important_ne could be relevant to the target-topic, or could indicate off-topic
        else:
            # 1. filter out any extra_ne that is same as wikiname -- e.g., wiki_name = Barak Obama, ne = Obama
            extra_ne = [
                ne[0] for ne in sent_obj["important_ne"] if ne[0] not in wiki_names_txt
            ]

            # 2. check if any of the extra_ne is the "SUBJECT" of the generation
            overlap_between_extraNE_and_subj = sent_obj["subject"].intersection(
                set(" ".join(extra_ne).split(" "))
            )

            if len(overlap_between_extraNE_and_subj) > 0:  # contains off-topic NE!!
                off_topic_gen_cnt += 1
                # off_topic_gens.append(sent_obj['gen'])
                # print('[OFF-TOPIC]', sent_obj['gen'])
            else:
                checkworthy_gen_cnt += 1
                checkworthy_gens.append(sent_obj)
                # print('[CHECK-WORTHY]', sent_obj['gen'])

    return checkworthy_gens
