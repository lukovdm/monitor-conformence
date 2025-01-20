from fractions import Fraction
import json
from math import ceil
import math
import os
from random import seed, shuffle
import traceback
from typing import Any
import re
import matplotlib.pyplot as plt
from tabulate import SEPARATING_LINE, tabulate


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


def prep_data_for_latex(data):
    for d in data:
        d["experiment"]["name"] = d["experiment"]["name"].replace("_", "\\_")
        d["experiment"]["variant"] = d["experiment"]["variant"].replace("_", "\\_")


def add_short_names(data):
    # Short names are first letter of name and index of variant
    variant_indexes: dict[str, int] = {}
    for d in data:
        name = d["experiment"]["name"]
        if name not in variant_indexes:
            variant_indexes[name] = 0
        d["experiment"][
            "short_name"
        ] = f"{name[0].capitalize()}-{variant_indexes[name]}"
        variant_indexes[name] += 1


def add_symbol_color(data):
    symbols = [
        "o",
        "s",
        "D",
        ">",
        "^",
        "p",
        "*",
        "h",
        "H",
        "d",
    ]
    seed(42)
    shuffle(symbols)
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
    json_files: list[str] = [
        f for f in os.listdir(path + "/json") if f.endswith(".json")
    ]

    experiment_data: list[dict] = []
    for json_file in json_files:
        with open(os.path.join(path + "/json/", json_file), "r") as f:
            try:
                data: dict = json.load(f)
                data["json_path"] = os.path.join(path + "/json/", json_file)
                data["log_path"] = os.path.join(
                    path + "/logs/", json_file.replace(".json", ".log")
                )
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


def generate_experiment_table(
    data, save_figures=False, save_path="./", file_name="runtime"
):
    tab_data = [
        [
            d["experiment"]["name"],
            d["experiment"]["short_name"],
            d["experiment"]["threshold"] - d["experiment"]["fp_slack"],
            d["experiment"]["threshold"] + d["experiment"]["fn_slack"],
            d["mc"]["mc_states"],
            d["mc"]["mc_transitions"],
            d["mc"]["mc_observations"],
            d["verimon"]["time"],
            d["verimon"]["monitor_states"],
            d["verimon"]["false_positive"],
            d["verimon"]["false_negative"],
            d["sampling"]["time"],
            d["sampling"]["monitor_states"],
            d["sampling"]["false_positive"],
            d["sampling"]["false_negative"],
        ]
        for d in data
    ]
    tab_with_lines: list[Any] = [tab_data[0]]
    for l in tab_data[1:]:
        # if l[0] != tab_with_lines[-1][0]:
        #     tab_with_lines.append(SEPARATING_LINE)
        tab_with_lines.append(l)

    table = tabulate(
        tab_with_lines,
        headers=[
            "Name",
            "Short Name",
            "$\\lambda_u$",
            "$\\lambda_s$",
            "$|\\Sts|$",
            "$|\\ptrans|$",
            "$|Z|$",
            "\\alg time",
            "\\alg monitor states",
            "\\alg minimum in monitor",
            "\\alg maximum out of monitor",
            "Sampling time",
            "Sampling monitor states",
            "Sampling minimum in monitor",
            "Sampling maximum out of monitor",
        ],
        tablefmt="latex_raw" if save_figures else "github",
    )
    if save_figures:
        with open(f"{save_path}/{file_name}.tex", "w") as f:
            f.write(table)

    print(table)


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
    experiments_in_legends=True,
    save_figures=False,
    save_path="./",
    file_name="runtime",
):
    max_runtime = max(
        max(
            data[key1]["time"] if key1 in data else 0,
            data[key2]["time"] if key2 in data else 0,
        )
        for data in exp_data
    )

    max_lim = max_runtime * 1.5
    offset = 1.3
    plt.text(
        max_lim * (1 / offset),
        offset,
        "Sampling is faster",
        color="gray",
        ha="right",
        va="bottom",
        rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    )
    plt.text(
        offset,
        max_lim * (1 / offset),
        "ToVer is faster",
        color="gray",
        ha="left",
        va="top",
        rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    )

    plt.plot(
        [0, max_lim],
        [0, max_lim],
        "k-",
    )
    plt.plot(
        [0, max_lim * 10],
        [0, max_lim],
        "k--",
        label="10x faster",
    )
    plt.plot(
        [0, max_lim],
        [0, max_lim * 10],
        "k--",
    )
    plt.plot(
        [0, max_lim * 100],
        [0, max_lim],
        "k:",
        label="100x faster",
    )
    plt.plot(
        [0, max_lim],
        [0, max_lim * 100],
        "k:",
    )
    plt.fill_between(
        [0, max_lim],
        [0, max_lim],
        max_lim,
        color="lightgreen",
        alpha=0.3,
        label=f"{key1.capitalize()} is faster",
    )
    plt.fill_between(
        [0, max_lim],
        0,
        [0, max_lim],
        color="lightcoral",
        alpha=0.3,
        label=f"{key2.capitalize()} is faster",
    )

    for data in exp_data:
        if key1 not in data or key2 not in data:
            continue
        time1 = data[key1]["time"]
        time2 = data[key2]["time"]
        if time1 is None or time2 is None:
            continue
        plt.plot(
            time1,
            time2,
            data["symbol"],
            color=data["color"],
            label=name_func(data) if experiments_in_legends else None,
        )

    plt.xlabel(xlabel if xlabel else f"{key1.capitalize()} run time (s log)")
    plt.ylabel(ylabel if ylabel else f"{key2.capitalize()} run time (s log)")
    # plt.title(
    #     title
    #     if title
    #     else f"Comparison of {key1.capitalize()} and {key2.capitalize()} Run Times"
    # )
    plt.xlim(1, max_lim)
    plt.ylim(1, max_lim)
    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # plt.legend(loc="upper left")
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def compare_monitor_sizes(
    exp_data: list[dict[str, Any]],
    key1: str,
    key2: str,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: bool = True,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
    experiments_in_legends=True,
    save_figures=False,
    save_path="./",
    file_name="monitor_sizes",
):
    max_mon_states = max(
        max(
            data[key1]["monitor_states"] if key1 in data else 0,
            data[key2]["monitor_states"] if key2 in data else 0,
        )
        for data in exp_data
    )
    max_lim = max_mon_states * 1.5
    offset = 1.5
    plt.text(
        max_lim * (1 / offset),
        offset,
        "Sampling is smaller",
        color="gray",
        ha="right",
        va="bottom",
        rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    )
    plt.text(
        offset,
        max_lim * (1 / offset),
        "ToVer is smaller",
        color="gray",
        ha="left",
        va="top",
        rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    )

    plt.plot(
        [0, max_lim],
        [0, max_lim],
        "k-",
        linewidth=1,
    )
    plt.plot(
        [0, max_lim * 10],
        [0, max_lim],
        "k--",
        label="10x smaller",
    )
    plt.plot(
        [0, max_lim],
        [0, max_lim * 10],
        "k--",
    )
    plt.plot(
        [0, max_lim * 100],
        [0, max_lim],
        "k:",
        label="100x smaller",
    )
    plt.plot(
        [0, max_lim],
        [0, max_lim * 100],
        "k:",
    )
    plt.fill_between(
        [0, max_lim],
        [0, max_lim],
        max_lim,
        color="lightgreen",
        alpha=0.3,
        label=f"{key1.capitalize()} is smaller",
    )
    plt.fill_between(
        [0, max_lim],
        0,
        [0, max_lim],
        color="lightcoral",
        alpha=0.3,
        label=f"{key2.capitalize()} is smaller",
    )

    for data in exp_data:
        if key1 not in data or key2 not in data:
            continue
        monitor_states1 = data[key1]["monitor_states"]
        monitor_states2 = data[key2]["monitor_states"]
        if monitor_states1 is None or monitor_states2 is None:
            continue
        plt.plot(
            monitor_states1,
            monitor_states2,
            data["symbol"],
            color=data["color"],
            label=name_func(data) if experiments_in_legends else None,
        )

    plt.xlabel(xlabel if xlabel else f"{key1.capitalize()} nr of monitor states (log)")
    plt.ylabel(ylabel if ylabel else f"{key2.capitalize()} nr of monitor states (log)")
    # plt.title(
    #     title
    #     if title
    #     else f"Comparison of {key1.capitalize()} and {key2.capitalize()} in Monitor Sizes"
    # )
    plt.xlim(1, max_lim)
    plt.ylim(1, max_lim)
    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
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
    # plt.title(
    #     title
    #     if title
    #     else f"Comparison of False Positives and False Negatives thresholds in {key1.capitalize()} and {key2.capitalize()}"
    # )
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
    bundle=1,
    fig_size=(10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    experiments_in_legends=True,
    save_figures=False,
    save_path="./",
    file_name="thresholds",
):

    plt.fill_betweenx(
        [0, threshold - fp_slack],
        -1,
        len(exp_data) / bundle + 1,
        color="lightgreen",
        alpha=0.3,
    )
    plt.fill_betweenx(
        [threshold + fn_slack, 1],
        -1,
        len(exp_data) / bundle + 1,
        color="lightcoral",
        alpha=0.5,
    )
    plt.fill_betweenx(
        [threshold - fp_slack, threshold + fn_slack],
        -1,
        len(exp_data) / bundle + 1,
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
            [1 if fp is None else fp for fp in fp_thresholds],
            [0 if fn is None else fn for fn in fn_thresholds],
        )
        for fp_thresholds, fn_thresholds in found_thresholds
    ]

    found_bundled_thresholds = []
    if bundle == 1:
        found_bundled_thresholds = found_thresholds
        exp_names = [bottom_func(data) for data in exp_data]
    else:
        exp_names = []
        for thresh in found_thresholds:
            found_bundled_thresholds.append(([], []))
            for j in range(0, len(thresh[1]), bundle):
                max_idx = thresh[1][j : j + bundle].index(
                    max(thresh[1][j : j + bundle])
                )
                exp_names.append(bottom_func(exp_data[j : j + bundle][max_idx]))
                found_bundled_thresholds[-1][0].append(
                    thresh[0][j : j + bundle][max_idx]
                )
                found_bundled_thresholds[-1][1].append(
                    thresh[1][j : j + bundle][max_idx]
                )

    bar_width = 1 / (len(keys) * 2) - 0.05
    index = range(math.ceil(len(exp_data) / bundle))

    for i, (key_fp_thresholds, key_fn_thresholds) in enumerate(
        found_bundled_thresholds
    ):
        plt.bar(
            [j + i * 2 * bar_width for j in index],
            [-(1 - t) for t in key_fp_thresholds],
            bar_width,
            bottom=1,
            label=f"in monitor traces",
            color=colors(i + 6 % colors.N),
        )
        plt.bar(
            [j + (i * 2 + 1) * bar_width for j in index],
            key_fn_thresholds,
            bar_width,
            label=f"out of monitor traces",
            color=colors(i + 4 % colors.N),
        )

    plt.xlabel(xlabel if xlabel else bottom_name.capitalize())
    plt.ylabel(ylabel if ylabel else "risk threshold")
    plt.xticks(
        [i + bar_width * (len(keys) - 0.5) for i in index],
        exp_names,
        rotation=90,
    )
    plt.legend(loc="upper left")
    plt.grid(axis="y")
    plt.xlim(-0.5, len(exp_data) / bundle)

    fig = plt.gcf()
    fig.set_size_inches(*fig_size)
    # plt.title(title if title else "Comparison of Thresholds")
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
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
    experiments_in_legends=True,
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
    if experiments_in_legends:
        fig.legend(handles, labels, bbox_to_anchor=(1, 0.9), loc="upper left")
    plt.tight_layout()
    plt.show()


def runtime_from_logs(logpath: str):
    entries: dict[str, float] = {}
    with open(logpath, "r") as f:
        for line in f:
            time_pattern = re.compile(r"\((\d+(?:\.\d+))s\)")

            for line in f:
                if "(s)" in line:
                    continue
                match = time_pattern.search(line)
                if match:
                    try:
                        elapsed = float(match.group(1))
                        message = line.split(" - ", 2)[-1].strip()
                        if message not in entries:
                            entries[message] = 0

                        entries[message] += elapsed
                    except ValueError:
                        pass
        return entries
