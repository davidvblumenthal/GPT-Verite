from dataclasses import dataclass


@dataclass
class FactualResult:
    entail_ratio: float = None
    false_ratio: float = None
    neutral_ratio: float = None
    off_topic_ratio: float = None

    hallu_ner: float = None

    num_entail: int = 0
    num_false: int = 0
    num_neutral: int = 0
    num_off_topic: int = 0

    num_correct_mentioned_ne: int = 0
    num_total_mentioned_ne: int = 0

    num_sentences: int = 0

    num_empty_generations: int = 0

    analysis_content: list = None

    def calculate_ratios(self):
        self.entail_ratio = self.num_entail / self.num_sentences
        self.false_ratio = self.num_false / self.num_sentences
        self.neutral_ratio = self.num_neutral / self.num_sentences
        self.off_topic_ratio = self.num_off_topic / self.num_sentences

        self.hallu_ner = 1 - (
            self.num_correct_mentioned_ne / self.num_total_mentioned_ne
        )

    def get_label_results(self):
        auto_labels = []

        for sample in self.analysis_content:
            sent_label = []
            for single_sentence in sample["single_sentences"]:
                sent_label.append(single_sentence[:2])

            auto_labels.append(
                {
                    "prompt": sample["prompt"],
                    "text": sample["text"],
                    "single_sentences": sent_label,
                }
            )

        return auto_labels

    def pretty_analysis_content(self):
        pretty_list = []
        """
            generation_with_labels now dict{"prompt": "", "text": "", "single_sentences": [[sentence, label, probabilities, used_evidence, evidence]], "gt_wiki_sentences": ["", ""]}
        """
        for sample in self.analysis_content:
            flatten = []

            # print(f"Debugging Pretty analysis: {sample}")

            for single_sentence in sample["single_sentences"]:
                if (
                    single_sentence[1] == "true"
                    or single_sentence[1] == "false"
                    or single_sentence[1] == "not_enough_evidence"
                ):
                    flatten.append(
                        {
                            "sentence": single_sentence[0],
                            "label": single_sentence[1],
                            "probabilities": single_sentence[2],
                            "used_evidence": single_sentence[3],
                            "evidence": single_sentence[4],
                        }
                    )
                else:
                    flatten.append(
                        {
                            "sentence": single_sentence[0],
                            "label": single_sentence[1],
                            "probabilities": "-",
                            "used_evidence": "-",
                            "evidence": "-",
                        }
                    )

            pretty_list.append(
                {
                    "prompt": sample["prompt"],
                    "text": sample["text"],
                    "single_sentences": [flatten],
                    "gt_truth": sample["gt_wiki_sentences"],
                }
            )

        return pretty_list


@dataclass
class SingleFactualResult(FactualResult):
    id: int = None
    wiki_article: str = None

    analysis_content: dict = None
