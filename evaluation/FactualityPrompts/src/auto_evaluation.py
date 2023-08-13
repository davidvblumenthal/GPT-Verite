from ipaddress import _BaseAddress
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import Counter
import argparse
import jsonlines

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

from utils.retriever import obtain_relevant_evidences, get_wiki_from_db
from utils.factuality_metric import (
    ner_metric,
    nli_metric_batch_vitc,
)  # nli_metric_batch

from utils.const import DATA_DIR, HOME_DIR, GEN_DIR, ADDITIONAL_NE_TRAIN, ADDITIONAL_NE_TEST
from utils.claim_handling import obtain_important_ne
from utils.claim_handling import check_assigned_types_single, check_assigned_types

from utils.result import SingleFactualResult, FactualResult

from utils.read_write_utils import read_prompts_and_generations
from utils.read_write_utils import write_jsonl, read_jsonl, construct_paths, read_addtional_ne

from utils.coref_utils import coref_generations
from utils.analysis import reconstruct_generation_with_labels

from utils.factuality_metric import NliModel

from utils.helpers import index_of_list_with_highest_value


def count_sentences_types_return_fact(checked_generation: dict):
    num_fact, num_no_fact_and_off = 0, 0

    sents_to_check = []

    for sent in checked_generation["single_sentences"]:
        # sent is tuple of form (sentence, label)
        if sent[1] == "fact":
            num_fact += 1
            sents_to_check.append(sent[0])

        if sent[1] == "off_topic" or sent[1] == "no_fact":
            num_no_fact_and_off += 1

    return num_fact, num_no_fact_and_off, sents_to_check


def entailment_single_sentence(gt_sentences, single_claim, nli_model=None):
    # Get the relevant evidence for claim checking
    """
    Use Sentence Transforer to embeedd the claim and the Sentences from Wikipedia
    find the Wikipedia Sentence that has the highest similarity with the claim

    returns:
        2 x k sentences
    """
    evidence = obtain_relevant_evidences(
        single_claim, gt_sentences, k=8, method="emb_sim" #emb_sim  combined
    )

    #print(f"Evidence: {evidence}")

    # Identify the evidence sentence that gives the highest entailment score
    premise_hypothesis_pairs = [
        [ev[0], single_claim] for ev in evidence
    ]  # Construct [evidence, claim] pairs

    #print(f"Premise Hypothesis Pair: {premise_hypothesis_pairs}")

    # Use NLI model on the the [evidence, claim] pairs
    """
        label NLI model:
            0 -> contradict
            1 -> neutral
            2 -> entail
        returns:
            list: class_probabilities, list: labels
    """
    nli_probs, labels = nli_model.nli_metric_batch(premise_hypothesis_pairs)

    # Find which pair had highest entailment probability
    #entailment_argmax = np.argmax([nli_s[2] for nli_s in nli_probs])
    entailment_argmax = index_of_list_with_highest_value(nli_probs, use_true_false=True)

    # Get the probabilities of (contradict, neutral, entail) and the final label
    # for the (premise, hypothesis) pair with the highest entail probability
    max_prob = nli_probs[entailment_argmax]
    max_label = labels[entailment_argmax]

    # Get all probabilities for the above sample
    nli_contradict_prob = max_prob[0]
    nli_neutral_prob = max_prob[1]
    nli_entail_prob = max_prob[2]

    used_evidence = evidence[entailment_argmax]

    return (
        max_label,
        (nli_neutral_prob, nli_contradict_prob, nli_entail_prob),
        used_evidence,
        evidence,
    )


def entail_sentences(gt_sentences, claims, save_labels=False, nli_model=None):
    entail, false, neutral = 0, 0, 0
    label_sentence_pair = []
    for single_claim in claims:
        """
        returns:
            label, (nli_neutral_prob, nli_contradict_prob, nli_entail_prob)
        """
        label, probs, used_evidence, evidence = entailment_single_sentence(
            gt_sentences, single_claim, nli_model=nli_model
        )

        label_sentence_pair.append(
            (single_claim, label, probs, used_evidence, evidence)
        )

        if label == 2:
            entail += 1
        if label == 1:
            neutral += 1
        if label == 0:
            false += 1

    if save_labels:
        return entail, false, neutral, label_sentence_pair

    else:
        return entail, false, neutral


def construct_text_from_fact_sents(checked_sentences: dict):
    result_text = ""
    sentences = checked_sentences["single_sentences"]

    for sentence in sentences:
        if sentence[1] == "fact":
            result_text = result_text + sentence[0] + " "

    return result_text


def calc_metrics_per_generation(
    gen_obj, prompt_obj, additional_nes=None, save_labels=False, nli_model=None
):

    if gen_obj["text"] == "":
        print("-----------------EMPTY GENERATION-----------------")
        """
        with jsonlines.open("./analysis/empty_generations/empty.jsonl", "a") as writer:
            writer.write(gen_obj)
        """
        return SingleFactualResult(num_empty_generations=1)
    # Label sentences as checkworthy, no_fact and off_topic
    """
    returns:
    {"prompt": str: prompt, "text": str: generation_text, "single_sentences": list of tuples: (sentence, label)}
    """
    if additional_nes:
        checked_sentences = check_assigned_types_single(
            gen_obj, prompt_obj, additional_nes
        )

    else:
        checked_sentences = check_assigned_types_single(gen_obj, prompt_obj)

    num_fact, num_no_fact_and_off, sents_to_check = count_sentences_types_return_fact(
        checked_sentences
    )

    # Get the content of the Wikipedia Article serving as ground truth
    article_names = [
        article[0] for article in prompt_obj["evidence_info"]
    ]  # Get Article Names from prompts
    gt_sentences = get_wiki_from_db(article_names)

    # ---------------------------------------------Calculate Entailment Ratio----------------------------------------#

    # Get the entailment labels for all sentences
    """
        returns:
            num_entail, num_false, num_neutral
    """
    if save_labels:
        """
        generation_with_labels = list: (sentence, label, probabilities, used_evidence, evidence)
        """
        num_entail, num_false, num_neutral, generations_with_labels = entail_sentences(
            gt_sentences, sents_to_check, save_labels=save_labels, nli_model=nli_model
        )
        """
            generation_with_labels now dict{"prompt": "", "text": "", "single_sentences": [[sentence, label, probabilities, used_evidence, evidence]]}
        """
        generations_with_labels = reconstruct_generation_with_labels(
            generations_with_labels, checked_sentences
        )
        # Appending the gt_sentences -> wiki article
        generations_with_labels["gt_wiki_sentences"] = gt_sentences

        # print(generations_with_labels)

    else:
        num_entail, num_false, num_neutral = entail_sentences(
            gt_sentences, sents_to_check, nli_model=nli_model
        )
        generations_with_labels = None

    # Calculate the ratios
    true_ratio = num_entail / len(checked_sentences["single_sentences"])
    false_ratio = num_false / len(checked_sentences["single_sentences"])
    neutral_ratio = num_neutral / len(checked_sentences["single_sentences"])

    off_topic_ratio = num_no_fact_and_off / len(checked_sentences["single_sentences"])

    # ------------------------------------------Calculate Named Entity Error----------------------------------------#

    # Calculate the hallucinated named entity error
    text_to_check = construct_text_from_fact_sents(checked_sentences)

    # print(text_to_check)

    # Get named entities mentioned in the lm generation
    named_entities_to_check = obtain_important_ne(text_to_check)
    named_entities_to_check = named_entities_to_check["important_ne"]

    # Check all Named Entities appearing in the generated text
    correct_ner_ratio, num_correct_named_entities = ner_metric(
        named_entities_to_check, gt_sentences
    )
    hallu_ner_ratio = 1 - correct_ner_ratio

    # Construct the results object
    result_factuality = SingleFactualResult(
        id=prompt_obj["id"], #gen_obj["id"],
        wiki_article=prompt_obj["evidence_info"][0][1], #gen_obj["wiki_ne"],
        entail_ratio=true_ratio,
        false_ratio=false_ratio,
        neutral_ratio=neutral_ratio,
        off_topic_ratio=off_topic_ratio,
        num_entail=num_entail,
        num_false=num_false,
        num_neutral=num_neutral,
        num_off_topic=num_no_fact_and_off,
        hallu_ner=hallu_ner_ratio,
        num_total_mentioned_ne=len(named_entities_to_check),
        num_correct_mentioned_ne=num_correct_named_entities,
        num_sentences=len(checked_sentences["single_sentences"]),
        analysis_content=generations_with_labels,
    )

    return result_factuality


def calc_metrics(
    list_gen_obj,
    list_prompt_obj,
    additional_nes=False,
    save_labels=False,
    nli_model=None,
):
    # Result object storing the results
    final_result = FactualResult()


    # Calculate metrics for each generation, prompt pair and append to list
    if additional_nes:
        list_result_factuality = [
            calc_metrics_per_generation(
                gen_obj, prompt_obj, additional_ne, save_labels, nli_model=nli_model
            )
            for gen_obj, prompt_obj, additional_ne in zip(
                list_gen_obj, list_prompt_obj, additional_nes
            )
        ]
    else:
        list_result_factuality = [
            calc_metrics_per_generation(
                gen_obj=gen_obj,
                prompt_obj=prompt_obj,
                additional_nes=additional_nes,
                save_labels=save_labels,
                nli_model=nli_model,
            )
            for gen_obj, prompt_obj in zip(list_gen_obj, list_prompt_obj)
        ]

    # print(f"Debugging List of Results: {list_result_factuality}")
    # Construct file with sentences and labels
    if save_labels:
        final_result.analysis_content = []

    # Sum up all individual results
    for result_factuality in list_result_factuality:
        # Set all absolute values
        final_result.num_entail += result_factuality.num_entail
        final_result.num_false += result_factuality.num_false
        final_result.num_neutral += result_factuality.num_neutral
        final_result.num_off_topic += result_factuality.num_off_topic

        final_result.num_correct_mentioned_ne += (
            result_factuality.num_correct_mentioned_ne
        )
        final_result.num_total_mentioned_ne += result_factuality.num_total_mentioned_ne

        final_result.num_sentences += result_factuality.num_sentences

        final_result.num_empty_generations += result_factuality.num_empty_generations

        if save_labels:
            final_result.analysis_content.append(result_factuality.analysis_content)

    # Call class method to calculate ratios
    final_result.calculate_ratios()

    # Print Values
    print(
        f"Hallu-ner {final_result.hallu_ner} Entail Ratio: {final_result.entail_ratio} \
            False Ratio: {final_result.false_ratio} \
            Off Topic Ratio: {final_result.off_topic_ratio}"
    )

    return final_result


# -------------------------------------------Main Methods------------------------------------------#


def calculate_metrics_main(args):
    gen_path, prompt_path = construct_paths(args)

    # Read in the Prompts and the Generations
    prompts, gens = read_prompts_and_generations(
        prompt_path=prompt_path, gen_path=gen_path, dedubed=args.dedub_generation
    )

    # Check for the use of coref resolution
    if args.use_coref_resolution:
        gens = coref_generations(gens)

    nli_model = NliModel(model_type=args.entailment_model)

    # Get the final result
    if args.use_additional_ne:
        additional_nes = read_addtional_ne(args) #read_jsonl(ADDITIONAL_NE_TEST) #ADDITIONAL_NE_TEST ADDITIONAL_NE_TRAIN
        final_result = calc_metrics(
            gens, prompts, additional_nes, args.save_labels, nli_model=nli_model
        )
    else:
        print("Running evaluation without using addtional NEs")
        final_result = calc_metrics(
            list_gen_obj=gens,
            list_prompt_obj=prompts,
            additional_nes=False,
            save_labels=args.save_labels,
            nli_model=nli_model,
        )

    # Construct save path and save as file

    save_path = gen_path.replace(".jsonl", f"_{args.entailment_model}.jsonl")

    if args.use_coref_resolution:
        save_path = save_path.replace(".jsonl", "_v_coref_results.jsonl")
    else:
        save_path = save_path.replace(".jsonl", "_results.jsonl")

    if args.save_labels:
        labeled_generation = final_result.get_label_results()
        save_path_labels = save_path.replace(".jsonl", "_labels.jsonl")

        pretty_analysis_content = final_result.pretty_analysis_content()
        save_path_pac = save_path.replace(".jsonl", "_analysis.jsonl")

        write_jsonl(save_path_labels, labeled_generation)
        write_jsonl(save_path_pac, pretty_analysis_content)

    final_metrics = final_result.__dict__
    final_metrics.pop("analysis_content")

    write_jsonl(save_path, [final_metrics])


def debug_main(args):
    gen_path, prompt_path = construct_paths(args)

    # Read in the Prompts and the Generations
    prompts, gens = read_prompts_and_generations(
        prompt_path=prompt_path, gen_path=gen_path
    )

    if args.use_additional_ne:
        additonal_ne = read_jsonl(ADDITIONAL_NE)

        for prompt, entities_add in zip(prompts, additonal_ne):
            entities_add = [[ne, "place_holder"] for ne in entities_add["important_ne"]]

            new_ev_inf = prompt["evidence_info"]
            new_ev_inf.extend(entities_add)

            prompt["evidence_info"] = new_ev_inf

    # Check assigned types for each sentences in all generations
    assigned_types_list = check_assigned_types(gens, prompts)

    # Write to file
    write_jsonl(args.out_path_assigned_types, assigned_types_list)


"""

for GEN_FOLDER in standard_wiki_1.3B
do

    for PROMPT_TYPE in nonfactual
    do
        GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-gen.jsonl
        PYTHONPATH=. python src/auto_evaluation.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME} \
        --gen_dir ${GEN_FOLDER} \
        --use_additional_ne \
        --entailment_model large_mnli
    done

done

labeled_test-set 150_labeled

--use_coref_resolution \
--out_path_assigned_types ./debugging_artefacts/run_add_ne_test.jsonl \
--use_additional_ne \
--save_labels
--entailment_model vitaminc-mnli vitaminc-fever large_mnli roberta-large-mnli

PYTHONPATH=. python src/auto_evaluation.py --prompt_type factual --gen_path factual-gen.jsonl \
        --gen_dir standard_wiki_1.3B \
        --use_additional_ne \
        --dedub_generation \
        --entailment_model large_mnli


"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_type",
        type=str,
        help="name of prompt type of the testset [factual, nonfactual]",
    )

    parser.add_argument(
        "--gen_path", type=str, default=None, help="path to generations to evaluate"
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default=None,
        help="path to dir containing subfolders of gens from dif. models",
    )

    parser.add_argument(
        "--out_path_assigned_types",
        type=str,
        help="path where intermidiate results are saved for analysis",
    )
    parser.add_argument("--use_additional_ne", action="store_true")

    parser.add_argument("--dedub_generation", action="store_true")

    parser.add_argument("--use_coref_resolution", action="store_true")

    parser.add_argument("--save_labels", action="store_true")

    parser.add_argument("--entailment_model", type=str)

    # parser.add_argument('--save_gen_for_analysis', action='store_true', help='Flag for saving some lm-gens with its metric for analysis')

    args = parser.parse_args()

    if args.out_path_assigned_types != None:
        debug_main(args)

    calculate_metrics_main(args)

    print("Finished!")
