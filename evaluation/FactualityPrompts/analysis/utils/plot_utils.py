import jsonlines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def read_jsonl(path: str) -> list:
    samples = []
    with jsonlines.open(path) as input:
        for line in input:
            samples.append(line)

    return samples


def read_auto_labels_and_gold_labels(path_auto: str, path_gold: str):
    auto = read_jsonl(path_auto)
    gold = read_jsonl(path_gold)

    return auto, gold


def extract_labels(label_list: list):
    all_labels = []
    for sample in label_list:
        for sent_tuple in sample["single_sentences"]:
            # sent_tuple = (sentence, label)
            all_labels.append(sent_tuple[1])

    return all_labels


def no_fact_to_off_topic(labels_gold, labels_auto):
    gold_new, auto_new = [], []
    for gold, auto in zip(labels_gold, labels_auto):
        if gold == "no_fact":
            gold_new.append("off_topic")
        else:
            gold_new.append(gold)
        if auto == "no_fact":
            auto_new.append("off_topic")
        else:
            auto_new.append(auto)

    return gold_new, auto_new


# hex #28AFBB


def cm_analysis_plot(y_true, y_pred, filename, labels, ymap=None, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    own_c_map = sns.light_palette(
        color="#28AFBB", reverse=False, as_cmap=True
    )  # , input='xkcd' , n_colors=6

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt="", ax=ax, cmap=own_c_map)
    plt.savefig(filename)
