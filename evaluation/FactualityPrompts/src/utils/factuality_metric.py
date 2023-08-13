import torch
from fairseq.data.data_utils import collate_tokens
import numpy as np
import re

# ---------------------------------------------#
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

"""
NLI_MODEL = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
NLI_MODEL.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1)
NLI_MODEL.to(device)
"""
# Use of different NLI Model
"""
tokenizer = AutoTokenizer.from_pretrained("tals/albert-xlarge-vitaminc-mnli")
model = AutoModelForSequenceClassification.from_pretrained("tals/albert-xlarge-vitaminc-mnli")
entail_pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, batch_size=64, top_k=None)
"""


class NliModel:
    model = None
    tokenizer = None
    pipeline = None
    device = None
    softmax = None

    def __init__(self, model_type):
        self.model_type = model_type

        if model_type == "vitaminc-mnli":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self.model = AutoModelForSequenceClassification.from_pretrained(
                "tals/albert-xlarge-vitaminc-mnli"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "tals/albert-xlarge-vitaminc-mnli"
            )
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=64,
                top_k=None,
                device=self.device,
            )

        if model_type == "vitaminc-fever":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self.model = AutoModelForSequenceClassification.from_pretrained(
                "tals/albert-base-vitaminc-fever"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "tals/albert-base-vitaminc-fever"
            )
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=64,
                top_k=None,
                device=self.device,
            )

        if model_type == "roberta-large-mnli":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self.model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-large-mnli"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=64,
                top_k=None,
                device=self.device,
            )

        if model_type == "large_mnli":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
            self.softmax = torch.nn.Softmax(dim=1)
            self.model.to(self.device)

    def _nli_metric_batch_torch(self, batch_of_pairs):
        # batch_of_pairs = [
        #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
        #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
        #     ['potatoes are awesome.', 'I like to run.'],
        #     ['Mars is very far from earth.', 'Mars is very close.'],
        # ]

        #print(f"Debug Batch of Pairs: {batch_of_pairs}")

        encoded_tokens = [
            self.model.encode(pair[0], pair[1]) for pair in batch_of_pairs
        ]
        encoded_tokens = [
            tokens[: min(len(tokens), 512)] for tokens in encoded_tokens
        ]  # trucate any long seq
        batch = collate_tokens(encoded_tokens, pad_idx=1)

        #print(f"Debug Batch after collate: {batch}")

        logprobs = self.model.predict("mnli", batch)
        logits = self.softmax(logprobs)
        labels = logits.argmax(dim=1)  # logprobs.argmax(dim=1)

        return logits.tolist(), labels.tolist()

    def _nli_metric_batch_hug(self, batch_of_pairs):
        batched_input = preprocess(batch_of_pairs)
        # batch_size = len(batched_input)
        """
            [[{'label': 'REFUTES', 'score': 0.9936186075210571},
            {'label': 'NOT ENOUGH INFO', 'score': 0.004795404151082039},
            {'label': 'SUPPORTS', 'score': 0.0015860619023442268}],
            [{'label': 'SUPPORTS', 'score': 0.7839190363883972},
            {'label': 'NOT ENOUGH INFO', 'score': 0.1614317148923874},
        """
        # Labels are not returned in fixed order, instead sorted according prob.
        outputs = self.pipeline(batched_input)

        return postprocess(outputs, self.model_type)

    def nli_metric_batch(self, batch_of_pairs):
        if self.model_type == "large_mnli":
            probs, labels = self._nli_metric_batch_torch(batch_of_pairs)

        if (
            self.model_type == "vitaminc-mnli"
            or self.model_type == "vitaminc-fever"
            or self.model_type == "roberta-large-mnli"
        ):
            probs, labels = self._nli_metric_batch_hug(batch_of_pairs)

        #print(f"Debug NLI {probs} and labels {labels}")

        return probs, labels


def preprocess(batch_of_pairs):
    pairs = [{"text": pair[0], "text_pair": pair[1]} for pair in batch_of_pairs]

    return pairs


def postprocess(outputs, model):
    """

    Sort output order probs: [[prob, prob, prob], [prob, prob, prob]] labels: [1, 1]
    """

    labels = []
    probs = []
    for pair in outputs:
        probs_sample = [None] * 3
        # Ranking is in order of highest prob first
        labels.append(pair[0]["label"])

        if model == "vitaminc-mnli" or model == "vitaminc-fever":
            for label in pair:
                if label["label"] == "REFUTES":
                    probs_sample[0] = label["score"]
                if label["label"] == "SUPPORTS":
                    probs_sample[2] = label["score"]
                if label["label"] == "NOT ENOUGH INFO":
                    probs_sample[1] = label["score"]

        if model == "roberta-large-mnli":
            for label in pair:
                if label["label"] == "CONTRADICTION":
                    probs_sample[0] = label["score"]
                if label["label"] == "ENTAILMENT":
                    probs_sample[2] = label["score"]
                if label["label"] == "NEUTRAL":
                    probs_sample[1] = label["score"]

        probs.append(probs_sample)

    if model == "vitaminc-mnli" or model == "vitaminc-fever":
        labels = list(map(label_to_int, labels))

    if model == "roberta-large-mnli":
        labels = list(map(label_to_int_sanity, labels))

    #print(f"Postprocess Deb {probs} and labels {labels}")

    return probs, labels


def label_to_int(label):
    if label == "REFUTES":
        return 0
    elif label == "SUPPORTS":
        return 2
    elif label == "NOT ENOUGH INFO":
        return 1
    else:
        print(f"Label mapping error {label}")


def label_to_int_sanity(label):
    if label == "CONTRADICTION":
        return 0
    elif label == "ENTAILMENT":
        return 2
    elif label == "NEUTRAL":
        return 1
    else:
        print(f"Label mapping error {label}")


def nli_metric_batch_vitc(batch_of_pairs):
    batched_input = preprocess(batch_of_pairs)
    # batch_size = len(batched_input)
    """
        [[{'label': 'REFUTES', 'score': 0.9936186075210571},
        {'label': 'NOT ENOUGH INFO', 'score': 0.004795404151082039},
        {'label': 'SUPPORTS', 'score': 0.0015860619023442268}],
        [{'label': 'SUPPORTS', 'score': 0.7839190363883972},
        {'label': 'NOT ENOUGH INFO', 'score': 0.1614317148923874},
    """
    # Labels are not returned in fixed order, instead sorted according prob.
    outputs = entail_pipe(batched_input)

    return postprocess(outputs)


"""
    Returns ([[contradiction, neutral, entailment]], argmax)
"""


def nli_metric_batch(batch_of_pairs):
    # batch_of_pairs = [
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
    #     ['potatoes are awesome.', 'I like to run.'],
    #     ['Mars is very far from earth.', 'Mars is very close.'],
    # ]

    encoded_tokens = [NLI_MODEL.encode(pair[0], pair[1]) for pair in batch_of_pairs]
    encoded_tokens = [
        tokens[: min(len(tokens), 512)] for tokens in encoded_tokens
    ]  # trucate any long seq
    batch = collate_tokens(encoded_tokens, pad_idx=1)

    logprobs = NLI_MODEL.predict("mnli", batch)
    logits = softmax(logprobs)
    labels = logits.argmax(dim=1)  # logprobs.argmax(dim=1)

    return logits.tolist(), labels.tolist()


def nli_metric(premise, hypothesis):
    # Encode a pair of sentences and make a prediction
    # tokens = NLI_MODEL.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    tokens = NLI_MODEL.encode(premise, hypothesis)

    seq_len = min(len(tokens), 512)

    logits = NLI_MODEL.predict("mnli", tokens[:seq_len])
    logits = softmax(logits)
    label = logits.argmax()  # 0: contradiction

    return logits.tolist(), label.tolist()


# ('As much as', 'CARDINAL')
# ('About 20', 'CARDINAL')
# ('67', 'CARDINAL'),
# ('14,000 meters', 'QUANTITY')    vs     ('1.4 kilometers', 'QUANTITY')


def ner_metric(named_entities, prompt_wiki_candidates):
    wiki_text = " ".join(prompt_wiki_candidates).lower()

    # TODO improve the NE match here
    # hanlde DATE, TIME, etc better! appears a lot but handled poorly

    existing_correct_ne = []
    for ent in named_entities:
        ent_text = ent[0].lower()
        if "the " in ent_text:
            ent_text = ent_text.replace("the ", "")

        if ent_text in wiki_text:
            existing_correct_ne.append(ent)
        elif any(
            [
                bool(word in wiki_text)
                for word in ent_text.split(" ")
                if ent[1] == "PERSON"
            ]
        ):
            # handle shorter forms of same NE: Exists "Marcus Morgan Bentley", but NE is "Marcus Bentley" or "Bentley"
            existing_correct_ne.append(ent)
        elif ent[1] == "DATE":
            date_str = re.sub(r"[,.;@#?!&$]+\ *", " ", ent_text)
            date_str = date_str.replace("st", "")
            date_str = date_str.replace("nd", "")
            date_str = date_str.replace("th", "")
            date_str = date_str.replace("of", "")
            date_tokens = date_str.split(" ")

            if all([bool(token in wiki_text) for token in date_tokens]):
                existing_correct_ne.append(ent)

    # Check for the case should there be no named entities -> division by zero
    # //TODO Check what sentences have zero NEs -> does sentence contain a fact?
    if len(named_entities) != 0:
        correct_ratio = len(existing_correct_ne) / len(named_entities)
        num_correct_ne = len(existing_correct_ne)
    else:
        correct_ratio = 1
        num_correct_ne = 0

    return correct_ratio, num_correct_ne


def ie_metric(claims, evidences):
    return NotImplementedError


if __name__ == "__main__":
    print("Hi")
