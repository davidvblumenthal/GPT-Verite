from .result import SingleFactualResult
from .result import FactualResult

from .claim_handling import obtain_important_ne


def construct_text_from_sentences(generation: dict):
    result_text = ""
    sentences = generation["single_sentences"]

    for sentence in sentences:
        if (
            sentence[1] == "true"
            or sentence[1] == "false"
            or sentence[1] == "not_enough_evidence"
        ):
            result_text = result_text + sentence[0] + " "

    return result_text


def calc_metrics_per_generation(generation: dict, calc_type="full"):
    if calc_type == "full":
        imp_text = construct_text_from_sentences(generation)
    if calc_type == "single":
        imp_text = generation["single_sentences"][0][0]

    # Calc NE Error
    named_entities = obtain_important_ne(imp_text)  # generation["text"]
    named_entities = named_entities["important_ne"]

    hallu_ne_text = " ".join(generation["HALLU_NE"])

    print(f"Named Entities from Generation: {named_entities}")
    print(f"Labeled as hallucinated: {generation['HALLU_NE']}")

    num_partial_matches = 0

    for named_entity in named_entities:
        named_entity = named_entity[0]
        # Check for an exact match
        if named_entity in generation["HALLU_NE"]:
            print(f"Exact Match: {named_entity}")
            continue

        # If not exact match check if partial match
        elif any(subword in hallu_ne_text for subword in named_entity.split(" ")):
            num_partial_matches += 1
            print(f"Partial Match: {named_entity}")

    if len(named_entities) != 0:
        correct_ratio = (
            len(named_entities) - (len(generation["HALLU_NE"]) + num_partial_matches)
        ) / len(named_entities)

        num_correct_named_entities = len(named_entities) - (
            len(generation["HALLU_NE"]) + num_partial_matches
        )

    else:
        correct_ratio = 1

        num_correct_named_entities = 0

    hallu_ne_ratio = 1 - correct_ratio

    # Calculate Entailment Ratio

    off_topic = 0
    no_fact = 0
    neutral = 0

    entail = 0

    if calc_type == "full":
        single_sentences = generation["single_sentences"]
    if calc_type == "single":
        single_sentences = [generation["single_sentences"][0]]
        print(single_sentences)

    for sent in single_sentences:
        # sent is tuple of the from (sentence, label)
        if sent[1] == "true":
            entail += 1

        if sent[1] == "off_topic" or sent[1] == "no_fact":
            off_topic += 1

        if sent[1] == "not_enough_evidence":
            neutral += 1

        if sent[1] == "false":
            no_fact += 1

    true_ratio = entail / len(single_sentences)
    false_ratio = no_fact / len(single_sentences)
    off_topic_ratio = off_topic / len(single_sentences)
    neutral_ratio = neutral / len(single_sentences)

    # return hallu_ne_ratio, (true_ratio, false_ratio, off_topic_ratio)

    return SingleFactualResult(
        id=generation["id"],
        wiki_article=generation["wiki_ne"],
        entail_ratio=true_ratio,
        hallu_ner=hallu_ne_ratio,
        false_ratio=false_ratio,
        off_topic_ratio=off_topic_ratio,
        neutral_ratio=neutral_ratio,
        num_neutral=neutral,
        num_entail=entail,
        num_false=no_fact,
        num_off_topic=off_topic,
        num_correct_mentioned_ne=num_correct_named_entities,
        num_total_mentioned_ne=len(named_entities),
        num_sentences=len(single_sentences),
    )


def calc_metrics_per_generation_list(label_objects: list, calc_type="full") -> list:
    # Calculate metrics for all generations
    list_results = []
    for label_object in label_objects:
        result = calc_metrics_per_generation(label_object, calc_type=calc_type)
        list_results.append(result)

    return list_results


def calc_metrics_all(result_objects: list):
    final_result = FactualResult()

    for result_obj in result_objects:
        # set all absolute values
        final_result.num_entail += result_obj.num_entail
        final_result.num_false += result_obj.num_false
        final_result.num_off_topic += result_obj.num_off_topic
        final_result.num_neutral += result_obj.num_neutral

        final_result.num_correct_mentioned_ne += result_obj.num_correct_mentioned_ne
        final_result.num_total_mentioned_ne += result_obj.num_total_mentioned_ne

        final_result.num_sentences += result_obj.num_sentences

    # Call Method to calculate metrics
    final_result.calculate_ratios()

    print(
        f"Hallu-ner {final_result.hallu_ner} Entail Ratio: {final_result.entail_ratio} False Ratio: {final_result.false_ratio} Neutral Ratio: {final_result.neutral_ratio}Off Topic Ratio: {final_result.off_topic_ratio}"
    )

    return final_result
