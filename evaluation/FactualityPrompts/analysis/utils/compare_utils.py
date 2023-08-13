def extract_sentence_label_pairs(label_list: list):
    all_pairs = []

    for sample in label_list:
        for sent_tuple in sample["single_sentences"]:
            # sent_tuple = (sentence, label)
            all_pairs.append(sent_tuple)

    return all_pairs


def extract_gt_truth_from_analysis(analysis_list: list):
    all_gt = []

    for sample in analysis_list:
        gt = sample["gt_truth"]

        for single_sentences in sample["single_sentences"]:
            for single_sentence in single_sentences:
                single_sentence["ground_truth"] = gt
                all_gt.append(single_sentence)

    return all_gt


def get_miss_match_analysis(
    gold_labels, auto_label_with_analysis, miss_match=("true", "false")
):
    # auto_label_with_analysis is list of dict: {"sentence": "", "label": "", "probabilities": "", "used_evidence"}
    result_list = []

    pairs_gold = extract_sentence_label_pairs(gold_labels)

    auto = extract_gt_truth_from_analysis(auto_label_with_analysis)

    for gold_sent_label, auto_sent_label in zip(pairs_gold, auto):
        if (
            gold_sent_label[1] == miss_match[0]
            and auto_sent_label["label"] == miss_match[1]
        ):
            if gold_sent_label[0] == auto_sent_label["sentence"]:
                result_list.append(
                    {"gold_pair": gold_sent_label, "auto_pair": auto_sent_label}
                )
            else:
                auto_sent = auto_sent_label["sentence"]
                print(f"Something did not match: {gold_sent_label[0]} and {auto_sent}")
                break

    return result_list


def get_miss_match(gold_labels, auto_labels, miss_match=("true", "false")):
    result_list = []

    pairs_gold = extract_sentence_label_pairs(gold_labels)
    pairs_auto = extract_sentence_label_pairs(auto_labels)

    for gold_sent_label, auto_sent_label in zip(pairs_gold, pairs_auto):
        # Both are tuples of (sentence, label)
        if gold_sent_label[1] == miss_match[0] and auto_sent_label[1] == miss_match[1]:
            result_list.append(
                {"gold_pair": gold_sent_label, "auto_pair": auto_sent_label}
            )

    return result_list
