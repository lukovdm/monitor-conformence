from fractions import Fraction
import json
from math import ceil
import os
import traceback
from typing import Any
import matplotlib.pyplot as plt


def clean_dict(d):
    # If value is rational number, convert to float
    for k, v in d.items():
        if isinstance(v, dict):
            clean_dict(v)
        elif isinstance(v, str):
            try:
                d[k] = Fraction(v)
            except ValueError:
                pass


def clean_data(data):
    # Correctly interpret rational numebrs in data
    for d in data:
        clean_dict(d)


def add_symbol_color(data):
    symbols = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "d",
        "|",
        "_",
        ".",
        "1",
    ]
    colors = plt.get_cmap("tab20")

    experiment_names = set(data["experiment"]["name"] for data in data)
    experiment_symbols = {
        name: symbols[i % len(symbols)] for i, name in enumerate(experiment_names)
    }

    for i, exp in enumerate(data):
        exp["symbol"] = experiment_symbols[exp["experiment"]["name"]]
        exp["color"] = colors(i % colors.N)

    return symbols, colors


def load_experiment_data(path):
    json_files = [f for f in os.listdir(path + "/json") if f.endswith(".json")]

    experiment_data: list[dict] = []
    for json_file in json_files:
        with open(os.path.join(path + "/json/", json_file), "r") as f:
            try:
                data: dict = json.load(f)
            except json.JSONDecodeError:
                print(f"Error in {json_file}: JSONDecodeError")
                traceback.print_exc()
                continue
            if "verimon" in data and "error" in data["verimon"]:
                print(f"Error in {json_file}: {data['verimon']['error']}")
                continue
            experiment_data.append(data)

    print(f"Loaded {len(experiment_data)} JSON files from {path}")
    experiment_data.sort(
        key=lambda x: (x["experiment"]["name"], str(x["experiment"]["variant"]))
    )
    return experiment_data


def compare_runtimes(
    exp_data: list[dict[str, Any]],
    key1: str,
    key2: str,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: bool = True,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
):
    max_key1 = 0
    max_key2 = 0
    min_key1 = float("inf")
    min_key2 = float("inf")
    for data in exp_data:
        if key1 not in data or key2 not in data:
            continue
        time1 = data[key1]["time"]
        time2 = data[key2]["time"]
        if time1 is None or time2 is None:
            continue
        max_key1 = max(max_key1, time1)
        max_key2 = max(max_key2, time2)
        min_key1 = min(min_key1, time1)
        min_key2 = min(min_key2, time2)
        plt.plot(
            time1,
            time2,
            data["symbol"],
            color=data["color"],
            label=name_func(data),
        )

    plt.xlabel(xlabel if xlabel else f"{key1.capitalize()} run time (s log)")
    plt.ylabel(ylabel if ylabel else f"{key2.capitalize()} run time (s log)")
    plt.title(
        title
        if title
        else f"Comparison of {key1.capitalize()} and {key2.capitalize()} Run Times"
    )
    plt.plot(
        [0, max(max_key1, max_key2) * 1.5],
        [0, max(max_key1, max_key2) * 1.5],
        "k-",
        label="diagonal",
    )
    plt.plot(
        [0, max(max_key1, max_key2) * 1.5 * 5],
        [0, max(max_key1, max_key2) * 1.5],
        "k--",
        label="5x faster",
    )
    plt.plot(
        [0, max(max_key1, max_key2) * 1.5],
        [0, max(max_key1, max_key2) * 1.5 * 5],
        "k--",
    )
    plt.plot(
        [0, max(max_key1, max_key2) * 1.5 * 50],
        [0, max(max_key1, max_key2) * 1.5],
        "k:",
        label="50x faster",
    )
    plt.plot(
        [0, max(max_key1, max_key2) * 1.5],
        [0, max(max_key1, max_key2) * 1.5 * 50],
        "k:",
    )
    plt.fill_between(
        [0, max(max_key1, max_key2) * 1.5],
        [0, max(max_key1, max_key2) * 1.5],
        max(max_key1, max_key2) * 1.5,
        color="lightgreen",
        alpha=0.3,
        label=f"{key1.capitalize()} is faster",
    )
    plt.fill_between(
        [0, max(max_key1, max_key2) * 1.5],
        0,
        [0, max(max_key1, max_key2) * 1.5],
        color="lightcoral",
        alpha=0.3,
        label=f"{key2.capitalize()} is faster",
    )
    plt.xlim(min_key1 * 0.5, max_key1 * 1.5)
    plt.ylim(min_key2 * 0.5, max_key2 * 1.5)
    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    plt.show()


def compare_thresholds(
    exp_data,
    key1: str,
    key2: str,
    colors,
    threshold=0.3,
    fn_slack=0.05,
    fp_slack=0.2,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
):
    plt.axhline(y=threshold, color="gray", linestyle="--")
    plt.axhline(y=threshold + fn_slack, color="r", linestyle="--")
    plt.axvline(x=threshold, color="gray", linestyle="--")
    plt.axvline(x=threshold - fp_slack, color="r", linestyle="--")
    plt.fill_betweenx(
        [0, threshold + fn_slack],
        threshold - fp_slack,
        1,
        color="lightgreen",
        alpha=0.3,
    )

    xmin, ymin = 1, 1
    xmax, ymax = 0, 0

    for i, data in enumerate(exp_data):
        if key1 not in data or key2 not in data:
            continue
        key1_fp = data[key1]["false_positive"]
        key1_fp = 0 if key1_fp is None else key1_fp
        key1_fn = data[key1]["false_negative"]
        key1_fn = 1 if key1_fn is None else key1_fn
        key2_fp = data[key2]["false_positive"]
        key2_fp = 0 if key2_fp is None else key2_fp
        key2_fn = data[key2]["false_negative"]
        key2_fn = 1 if key2_fn is None else key2_fn

        plt.plot(
            [key1_fp],
            [key1_fn],
            data["symbol"],
            color=data["color"],
            label=name_func(data),
        )
        plt.plot([key1_fp, key2_fp], [key1_fn, key2_fn], color=colors(i % colors.N))
        xmin = min(xmin, key1_fp, key2_fp)
        ymin = min(ymin, key1_fn, key2_fn)
        xmax = max(xmax, key1_fp, key2_fp)
        ymax = max(ymax, key1_fn, key2_fn)

    plt.xlabel(
        xlabel
        if xlabel
        else "False Positives threshold\n(minimal risk for trace in monitor)"
    )
    plt.ylabel(
        ylabel
        if ylabel
        else "False Negatives threshold\n(maximal risk for trace not in monitor)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")
    plt.title(
        title
        if title
        else f"Comparison of False Positives and False Negatives thresholds in {key1.capitalize()} and {key2.capitalize()}"
    )
    plt.xlim(max(0, xmin * 0.95), min(1, xmax * 1.05))
    plt.ylim(max(0, ymin * 0.95), min(1, ymax * 1.05))
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    plt.show()


def compare_thresholds_bar(
    exp_data,
    keys: list[str],
    bottom_name: str,
    bottom_func,
    colors,
    threshold=0.3,
    fn_slack=0.0,
    fp_slack=0.0,
    fig_size=(10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):

    plt.fill_betweenx(
        [0, threshold - fp_slack], -1, len(exp_data) + 1, color="lightgreen", alpha=0.3
    )
    plt.fill_betweenx(
        [threshold + fn_slack, 1],
        -1,
        len(exp_data) + 1,
        color="lightcoral",
        alpha=0.5,
    )
    plt.fill_betweenx(
        [threshold - fp_slack, threshold + fn_slack],
        -1,
        len(exp_data) + 1,
        color="lightgrey",
        alpha=0.5,
    )

    plt.axhline(y=threshold, color="grey", linestyle="--")
    plt.axhline(y=threshold + fn_slack, color="grey", linestyle="-", linewidth=1)
    plt.axhline(y=threshold - fp_slack, color="grey", linestyle="-", linewidth=1)

    found_thresholds = [
        (
            [data[key]["false_positive"] for data in exp_data],
            [data[key]["false_negative"] for data in exp_data],
        )
        for key in keys
    ]

    # Replace None with -0.1 for plotting
    found_thresholds = [
        (
            [0 if fp is None else fp for fp in fp_thresholds],
            [1 if fn is None else fn for fn in fn_thresholds],
        )
        for fp_thresholds, fn_thresholds in found_thresholds
    ]

    # Horizon sizes for x-axis labels
    walks_per_state = [bottom_func(data) for data in exp_data]

    bar_width = 1 / (len(keys) * 2) - 0.05
    index = range(len(exp_data))

    for i, (found_thresholds, key) in enumerate(zip(found_thresholds, keys)):
        key_fp_thresholds, key_fn_thresholds = found_thresholds
        plt.bar(
            [j + i * 2 * bar_width for j in index],
            [-(1 - t) for t in key_fp_thresholds],
            bar_width,
            bottom=1,
            label=f"{key.capitalize()} (in monitor risks)",
            color=colors(i + 6 % colors.N),
        )
        plt.bar(
            [j + (i * 2 + 1) * bar_width for j in index],
            key_fn_thresholds,
            bar_width,
            label=f"{key.capitalize()} (out of monitor risks)",
            color=colors(i + 4 % colors.N),
        )

    plt.xlabel(xlabel if xlabel else bottom_name.capitalize())
    plt.ylabel(ylabel if ylabel else "threshold")
    plt.xticks(
        [i + bar_width * (len(keys) - 0.5) for i in index],
        walks_per_state,
        rotation=90,
    )
    plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
    plt.grid(axis="y")
    plt.xlim(-0.5, len(exp_data))

    fig = plt.gcf()
    fig.set_size_inches(*fig_size)
    plt.title(title if title else "Comparison of Thresholds")
    plt.show()


def runtime_by_params(
    exp_data,
    key: str,
    params: list[tuple[str, str]],
    title: str | None = None,
    figsize: tuple = (10, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: bool = True,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
):
    fig, axes = plt.subplots(nrows=ceil(len(params) / 2), ncols=2, figsize=figsize)
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        for data in exp_data:
            val = data[param[0]][param[1]]
            ax.scatter(
                val,
                data[key]["time"],
                marker=data["symbol"],
                color=data["color"],
                label=name_func(data),
            )
        ax.set_xlabel(xlabel if xlabel else param[1].replace("_", " ").capitalize())
        ax.set_ylabel(ylabel if ylabel else f"{key.capitalize()} Run Time (s log)")
        ax.set_title(
            title
            if title
            else f"{param[1].replace('_', ' ').capitalize()} to {key.capitalize()} Run Time"
        )
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="--")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0.9), loc="upper left")
    plt.tight_layout()
    plt.show()
