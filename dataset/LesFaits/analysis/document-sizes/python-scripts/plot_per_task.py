import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from variables import MAPPING_TASK_CODE_TO_TEXT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-file-path", type=Path, required=True)
    parser.add_argument("--statistics-pickle-file", type=Path, required=True)
    args = parser.parse_args()
    
    return args


def filter_out_empty_doc(df, task):
    len_before = len(df)
    df_filtered = df.drop(df[df["bytes per document"] == 0].index)
    len_after = len(df_filtered)
    if len_before != len_after:
        df_debug = df.drop(df[df["bytes per document"] != 0].index)

        print(
            f"len_before: {len_before} | len_after: {len_after} | task: {task} | datasets: {pd.unique(df_debug['dataset'])}"
        )
    return df


def process_df_per_task(all_data_point, task):
    data_points = all_data_point[task]
    df = pd.DataFrame(
        data_points, columns=["mean", "median", "bytes per document", "dataset"]
    )
    df = (
        df.set_index(["mean", "median", "dataset"])
        .apply(lambda x: x.explode())
        .reset_index()
    )

    df = df.astype({"bytes per document": "float"})
    df["task"] = MAPPING_TASK_CODE_TO_TEXT[task]

    df_filtered = filter_out_empty_doc(df, task)
    return df_filtered[["task", "bytes per document"]]


def get_order(df_all):
    df_all_median = df_all.groupby("task").median()
    print("Median all: ", df_all_median)

    df_all_median = df_all_median.reset_index()

    df_all_median = df_all_median.sort_values(by="bytes per document", ascending=False)
    return df_all_median["task"].to_list()


def main():
    args = get_args()

    with open(args.statistics_pickle_file, "rb") as handle:
        all_data_point = pickle.load(handle)

    sub_df = []
    for _, task in enumerate(all_data_point.keys()):
        print(f"Processing {task}")
        sub_df.append(process_df_per_task(all_data_point, task))

    df_all = pd.concat(sub_df, ignore_index=True)

    order = get_order(df_all)
    print("Order: ", order)

    width_box = 0.6
    _, ax = plt.subplots(figsize=(len(order) * width_box + 2, 6))
    ax.set_yscale("log")
    plt.xticks(rotation=40, ha="right")
    
    c_palatte = sns.light_palette(
        color="#28AFBB", reverse=True) 
    
    sns.boxplot(
        x="task",
        y="bytes per document",
        palette=c_palatte,
        data=df_all,
        ax=ax,
        order=order,
        width=width_box,
    )
    ax.set_ylabel("Number of bytes per document (log scale)")
    ax.set_xlabel(f"dataset per task")

    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.savefig(args.plot_file_path, dpi=300)


"""

python plot_per_task.py --plot-file-path ../../../../plots/new \
            --statistics-pickle-file ../../../../statistics_dataset/doc_len/stats.pickle

"""

if __name__ == "__main__":
    main()
