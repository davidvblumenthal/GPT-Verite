import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterSciNotation as LogFormatter
from matplotlib.ticker import LogLocator
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from variables import MAPPING_TASK_CODE_TO_TEXT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-plot-dir", type=Path, required=True)
    parser.add_argument("--statistics-pickle-file", type=Path, required=True)
    args = parser.parse_args()
    return args


def get_matplot_lib_colors(all_data_point):
    max_median = 0
    min_median = 2000
    for data_point in all_data_point.values():
        for item in data_point:
            median = item[1]
            if median > max_median:
                max_median = median
            if median < min_median:
                min_median = median
    print("Max of median is: ", max_median)
    print("Min of median is: ", min_median)

    cmap = sns.light_palette(
        color="#28AFBB", reverse=False, as_cmap=True
    ) 
    #cmap = matplotlib.cm.get_cmap("cool")
    norm = matplotlib.colors.LogNorm(vmin=min_median, vmax=max_median)
    return cmap, norm


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


def create_and_save_boxplot(task, all_data_point, norm, cmap, width_box, save_path):
    data_points = all_data_point[task]
    df = pd.DataFrame(
        data_points, columns=["mean", "median", "bytes per document", "dataset"]
    )

    df = df.sort_values(by="median", ascending=False)
    order = df["dataset"].to_list()
    medians = df["median"].to_list()

    df = (
        df.set_index(["mean", "median", "dataset"])
        .apply(lambda x: x.explode())
        .reset_index()
    )

    df = df.astype({"bytes per document": "float"})
    df = filter_out_empty_doc(df, task)

    fig, ax = plt.subplots(figsize=(len(order) * width_box + 2, 6))
    ax.set_yscale("log")
    plt.xticks(rotation=40, ha="right")
    sns.boxplot(
        x="dataset",
        y="bytes per document",
        # hue="dataset",
        # size="weight",
        # sizes=(40, 400),
        # alpha=.5,
        # palette="rainbow",
        # height=6,
        data=df,
        ax=ax,
        order=order,
        width=width_box,
    )
    ax.set_ylabel("Number of bytes per document (log scale)")
    # Map lang code to language in natural language
    task = MAPPING_TASK_CODE_TO_TEXT[task]
    ax.set_xlabel(f"{task} datasets")
    box_patches = [
        patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch
    ]

    for artist, median in zip(box_patches, medians):
        col = cmap(norm(median))
        artist.set_facecolor(col)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def create_and_save_colorbar(norm, cmap, save_path):
    # import numpy as np

    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = np.exp(-(x**2) - y**2) * 0 + 2

    fig, ax = plt.subplots()

    im = ax.pcolormesh(z, cmap=cmap, norm=norm)
    im.set_norm(norm)

    locator = LogLocator()
    formatter = LogFormatter()

    cbar = fig.colorbar(im, ax=ax, norm=norm)
    cbar.locator = locator
    cbar.formatter = formatter
    cbar.update_normal(im)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    args = get_args()

    with open(args.statistics_pickle_file, "rb") as handle:
        all_data_point = pickle.load(handle)

    cmap, norm = get_matplot_lib_colors(all_data_point)
    width_box = 0.4

    save_path = args.save_plot_dir / f"colorbar.png"
    create_and_save_colorbar(norm, cmap, save_path)

    for task in all_data_point.keys():
        print(f"Creating plot for {task}")
        save_path = args.save_plot_dir / f"boxplot_{task}.png"
        create_and_save_boxplot(task, all_data_point, norm, cmap, width_box, save_path)
        print(f"Plot for {task} saved at {save_path}")

"""

python plot_per_ds_per_task.py --save-plot-dir ../../../../plots/new \
            --statistics-pickle-file ../../../../statistics_dataset/doc_len/stats.pickle

"""

if __name__ == "__main__":
    main()
