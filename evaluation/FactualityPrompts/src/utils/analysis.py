from copy import deepcopy


def reconstruct_generation_with_labels(sentence_label_pairs, checked_generation):
    copy_checked_generation = deepcopy(checked_generation)
    single_sentences = copy_checked_generation["single_sentences"]
    # Counter for the sentence_label_pairs <-- only contain those samples initialy classified as containing "fact"
    i = 0
    for idx, sent in enumerate(single_sentences):
        # Sent is tuple of form (sentence, label)
        if sent[1] == "fact":
            assert (
                sent[0] == sentence_label_pairs[i][0]
            ), f"Sentences do not match: '{sentence_label_pairs[i][0]}' and '{sent[0]}'"

            # print(f"Debugging Labels: {sentence_label_pairs[i]}")
            list_tuple = list(sentence_label_pairs[i])  # list(single_sentences[idx])

            if sentence_label_pairs[i][1] == 2:
                list_tuple[1] = "true"
            if sentence_label_pairs[i][1] == 0:
                list_tuple[1] = "false"
            if sentence_label_pairs[i][1] == 1:
                list_tuple[1] = "not_enough_evidence"

            single_sentences[idx] = list_tuple

            i += 1

    assert i == len(sentence_label_pairs), "Not all labels successfully matched"

    copy_checked_generation["single_sentences"] = single_sentences

    return copy_checked_generation
