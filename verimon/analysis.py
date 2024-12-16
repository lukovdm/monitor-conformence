import json
from math import ceil, exp
import os
from turtle import color
from types import FunctionType
from typing import Any
import matplotlib.pyplot as plt


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
            data: dict = json.load(f)
            if "verimon" in data and "error" in data["verimon"]:
                print(f"Error in {json_file}: {data['verimon']['error']}")
                continue
            experiment_data.append(data)

    print(f"Loaded {len(experiment_data)} JSON files from {path}")
    experiment_data.sort(
        key=lambda x: (x["experiment"]["name"], str(x["experiment"]["variant"]))
    )
    return experiment_data


def compare_runtimes(exp_data: list[dict[str, Any]], key1: str, key2: str):
    max_verimon = 0
    max_trad = 0
    for data in exp_data:
        if key1 not in data or key2 not in data:
            continue
        time1 = data[key1]["time"]
        time2 = data[key2]["time"]
        if time1 is None or time2 is None:
            continue
        max_verimon = max(max_verimon, time1)
        max_trad = max(max_trad, time2)
        plt.plot(
            time1,
            time2,
            data["symbol"],
            color=data["color"],
            label=f"{data['experiment']['name']} {data['experiment']['variant']}",
        )

    plt.xlabel(f"{key1.capitalize()} Run Times (s log)")
    plt.ylabel(f"{key2.capitalize()} Run Times (s log)")
    plt.title(f"Comparison of {key1.capitalize()} and {key2.capitalize()} Run Times")
    plt.plot(
        [0, max(max_verimon, max_trad) * 1.5],
        [0, max(max_verimon, max_trad) * 1.5],
        "k-",
        label="diagonal",
    )
    plt.plot(
        [0, max(max_verimon, max_trad) * 1.5 * 5],
        [0, max(max_verimon, max_trad) * 1.5],
        "k--",
        label="5x faster",
    )
    plt.plot(
        [0, max(max_verimon, max_trad) * 1.5],
        [0, max(max_verimon, max_trad) * 1.5 * 5],
        "k--",
    )
    plt.plot(
        [0, max(max_verimon, max_trad) * 1.5 * 50],
        [0, max(max_verimon, max_trad) * 1.5],
        "k:",
        label="50x faster",
    )
    plt.plot(
        [0, max(max_verimon, max_trad) * 1.5],
        [0, max(max_verimon, max_trad) * 1.5 * 50],
        "k:",
    )
    plt.fill_between(
        [0, max(max_verimon, max_trad) * 1.5],
        [0, max(max_verimon, max_trad) * 1.5],
        max(max_verimon, max_trad) * 1.5,
        color="lightgreen",
        alpha=0.3,
        label=f"{key1.capitalize()} is faster",
    )
    plt.fill_between(
        [0, max(max_verimon, max_trad) * 1.5],
        0,
        [0, max(max_verimon, max_trad) * 1.5],
        color="lightcoral",
        alpha=0.3,
        label=f"{key2.capitalize()} is faster",
    )
    plt.xlim(1, max_verimon * 1.5)
    plt.ylim(1, max_trad * 1.5)
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")

    plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")
    plt.show()


def compare_thresholds(
    exp_data, key1: str, key2: str, colors, threshold=0.3, fn_slack=0.05, fp_slack=0.2
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
            label=f"{data['experiment']['name']} {data['experiment']['variant']}",
        )
        plt.plot([key1_fp, key2_fp], [key1_fn, key2_fn], color=colors(i % colors.N))
        xmin = min(xmin, key1_fp, key2_fp)
        ymin = min(ymin, key1_fn, key2_fn)
        xmax = max(xmax, key1_fp, key2_fp)
        ymax = max(ymax, key1_fn, key2_fn)

    plt.xlabel("False Positives threshold\n(minimal risk for trace in monitor)")
    plt.ylabel("False Negatives threshold\n(maximal risk for trace not in monitor)")
    plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")
    plt.title(
        f"Comparison of False Positives and False Negatives thresholds in {key1.capitalize()} and {key2.capitalize()}"
    )
    plt.xlim(max(0, xmin * 0.95), min(1, xmax * 1.05))
    plt.ylim(max(0, ymin * 0.95), min(1, ymax * 1.05))
    plt.grid()
    plt.show()


def compare_thresholds_bar(
    exp_data,
    keys: list[str],
    bottom_name: str,
    bottom_func,
    colors,
    threshold=0.3,
    fn_slack=None,
    fp_slack=None,
):
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
            color=colors((i * 2) % colors.N),
        )
        plt.bar(
            [j + (i * 2 + 1) * bar_width for j in index],
            key_fn_thresholds,
            bar_width,
            label=f"{key.capitalize()} (out of monitor risks)",
            color=colors((i * 2 + 1) % colors.N),
        )

    plt.xlabel(bottom_name.capitalize())
    plt.ylabel("threshold")
    plt.xticks(
        [i + bar_width * len(keys) for i in index],
        walks_per_state,
        rotation=90,
    )
    plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
    plt.grid(axis="y")
    plt.axhline(y=threshold, color="grey", linestyle="--")
    if fn_slack is not None:
        plt.axhline(y=threshold + fn_slack, color="grey", linestyle="--")
    if fp_slack is not None:
        plt.axhline(y=threshold - fp_slack, color="grey", linestyle="--")

    fig = plt.gcf()
    fig.set_size_inches(10, 5)

    plt.show()


def runtime_by_params(exp_data, key: str, params: list[tuple[str, str]]):
    fig, axes = plt.subplots(nrows=ceil(len(params) / 2), ncols=2, figsize=(10, 10))
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
                label=f"{data['experiment']['name']} {data['experiment']['variant']}",
            )
        ax.set_xlabel(param[1].replace("_", " ").capitalize())
        ax.set_ylabel(f"{key.capitalize()} Run Time (s log)")
        ax.set_title(
            f"{param[1].replace('_', ' ').capitalize()} to {key.capitalize()} Run Time"
        )
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0.9), loc="upper left")
    plt.tight_layout()
    plt.show()
