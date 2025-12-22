from collections import OrderedDict
from fractions import Fraction
from math import ceil, log10
import re
import time
from typing import Any, cast
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations, product


def calculate_error_lines(data, value_func):
    max_time = 0
    for data in data:
        time_value = value_func(data)
        if (
            isinstance(time_value, float)
            or isinstance(time_value, int)
            or isinstance(time_value, Fraction)
        ):
            max_time = max(max_time, time_value)

    timeout = max_time * 1.5
    out_of_memory = timeout * 4
    incorrect = out_of_memory * 4
    unfinished = incorrect * 4
    return max_time, timeout, out_of_memory, incorrect, unfinished


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
    show_y_axis: bool = True,
    plot_kwargs={},
    min_value=1,
):
    max_time, timeout, out_of_memory, incorrect, unfinished = calculate_error_lines(
        exp_data,
        lambda d: max(
            d[key1]["time"] if key1 in d else 0,
            d[key2]["time"] if key2 in d else 0,
        ),
    )

    for data in exp_data:
        if key1 not in data or key2 not in data:
            continue
        time1 = data[key1]["time"]
        time2 = data[key2]["time"]

        if time1 == "timeout":
            time1 = timeout
        elif time1 == "out of memory":
            time1 = out_of_memory
        elif (
            data[key1]["false_positive"] is None
            or data[key1]["false_negative"] is None
            or data[key1]["false_positive"]
            < data["experiment"]["threshold"] - data["experiment"]["fp_slack"]
            or data[key1]["false_negative"]
            > data["experiment"]["threshold"] + data["experiment"]["fn_slack"]
        ):
            time1 = incorrect

        if time2 == "timeout":
            time2 = timeout
        elif time2 == "out of memory":
            time2 = out_of_memory
        elif (
            data[key2]["false_positive"] is None
            or data[key2]["false_negative"] is None
            or data[key2]["false_positive"]
            < data["experiment"]["threshold"] - data["experiment"]["fp_slack"]
            or data[key2]["false_negative"]
            > data["experiment"]["threshold"] + data["experiment"]["fn_slack"]
        ):
            time2 = incorrect

        plt.plot(
            max(time1, min_value),
            max(time2, min_value),
            data["symbol"],
            color=data["color"],
            label=name_func(data) if experiments_in_legends else None,
            **plot_kwargs,
        )

    max_lim = incorrect * 2

    plt.plot([0, max_lim], [0, max_lim], r"-", color="0.5")
    plt.plot(
        [0, max_lim * 10],
        [0, max_lim],
        "--",
        color="0.5",
        label="10x faster",
    )
    plt.plot(
        [0, max_lim],
        [0, max_lim * 10],
        "--",
        color="0.5",
    )
    plt.plot(
        [0, max_lim * 100],
        [0, max_lim],
        ":",
        color="0.5",
        label="100x faster",
    )
    plt.plot(
        [0, max_lim],
        [0, max_lim * 100],
        ":",
        color="0.5",
    )
    plt.fill_between(
        [0, max_lim],
        [0, max_lim],
        max_lim,
        color="lightgreen",
        alpha=0.2,
        label=f"{key1.capitalize()} is faster",
    )
    plt.fill_between(
        [0, max_lim],
        0,
        [0, max_lim],
        color="lightcoral",
        alpha=0.2,
        label=f"{key2.capitalize()} is faster",
    )
    plt.axline(
        (0, timeout),
        (timeout, timeout),
        color="gray",
        linestyle=r"--",
        label="ToVer timeout",
    )
    plt.axline(
        (timeout, 0),
        (timeout, timeout),
        color="gray",
        linestyle=r"--",
        label="Baseline timeout",
    )
    plt.axline(
        (0, out_of_memory),
        (out_of_memory, out_of_memory),
        color="gray",
        linestyle=r"--",
        label="ToVer out of memory",
    )
    plt.axline(
        (out_of_memory, 0),
        (out_of_memory, out_of_memory),
        color="gray",
        linestyle=r"--",
        label="Baseline out of memory",
    )
    plt.axline(
        (0, incorrect),
        (incorrect, incorrect),
        color="gray",
        linestyle=r"--",
        label="ToVer incorrect",
    )
    plt.axline(
        (incorrect, 0),
        (incorrect, incorrect),
        color="gray",
        linestyle=r"--",
        label="Baseline incorrect",
    )

    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else "ToVer (s log)")
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks(
            [10**i for i in range(-1, 5)] + [timeout, out_of_memory, incorrect],
            [],
        )
    else:
        plt.ylabel(ylabel if ylabel else "Baseline (s log)")
        plt.yticks(
            [10**i for i in range(-1, 5)] + [timeout, out_of_memory, incorrect],
            [f"$10^{{{i}}}$" for i in range(-1, 5)] + [r"$\infty$", "MO", r"$\times$"],
        )

    plt.xticks(
        [10**i for i in range(-1, 5)] + [timeout, out_of_memory, incorrect],
        [f"$10^{{{i}}}$" for i in range(-1, 5)] + [r"$\infty$", "MO", r"$\times$"],
    )
    ax = plt.gca()
    # xticks = ax.get_xticklabels()
    # for i in range(len(xticks) - 3, len(xticks)):
    #     xticks[i].set_rotation(90)
    #     xticks[i].set_ha("right")

    plt.xlim(min_value, max_lim)
    plt.ylim(min_value, max_lim)

    # Hide minor ticks between the last 3 major ticks (≥ 10^5)
    if log_scale:
        for axis in [ax.xaxis, ax.yaxis]:
            for tick in axis.get_minor_ticks():
                if tick.get_loc() >= 10**4:
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)

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
    show_y_axis: bool = True,
    plot_kwargs={},
):
    max_mon_states = max(
        max(
            (
                data[key1]["monitor_states"]
                if key1 in data and isinstance(data[key1]["monitor_states"], int)
                else 0
            ),
            (
                data[key2]["monitor_states"]
                if key2 in data and isinstance(data[key2]["monitor_states"], int)
                else 0
            ),
        )
        for data in exp_data
    )
    max_lim = max_mon_states * 1.5
    # offset = 1.5
    # plt.text(
    #     max_lim * (1 / offset),
    #     offset,
    #     "Baseline is smaller",
    #     color="gray",
    #     ha="right",
    #     va="bottom",
    #     rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    # )
    # plt.text(
    #     offset,
    #     max_lim * (1 / offset),
    #     "ToVer is smaller",
    #     color="gray",
    #     ha="left",
    #     va="top",
    #     rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    # )

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
        alpha=0.2,
        label=f"{key1.capitalize()} is smaller",
    )
    plt.fill_between(
        [0, max_lim],
        0,
        [0, max_lim],
        color="lightcoral",
        alpha=0.2,
        label=f"{key2.capitalize()} is smaller",
    )

    for data in exp_data:
        if key1 not in data or key2 not in data:
            continue

        monitor_states1 = data[key1]["monitor_states"]
        monitor_states2 = data[key2]["monitor_states"]
        if monitor_states1 is None or monitor_states2 is None:
            continue
        if isinstance(monitor_states1, str) or isinstance(monitor_states2, str):
            continue
        if (
            data[key1]["false_positive"] is None
            or data[key1]["false_negative"] is None
            or data[key1]["false_positive"]
            < data["experiment"]["threshold"] - data["experiment"]["fp_slack"]
            or data[key1]["false_negative"]
            > data["experiment"]["threshold"] + data["experiment"]["fn_slack"]
            or data[key2]["false_positive"] is None
            or data[key2]["false_negative"] is None
            or data[key2]["false_positive"]
            < data["experiment"]["threshold"] - data["experiment"]["fp_slack"]
            or data[key2]["false_negative"]
            > data["experiment"]["threshold"] + data["experiment"]["fn_slack"]
        ):
            continue

        plt.plot(
            monitor_states1,
            monitor_states2,
            data["symbol"],
            color=data["color"],
            label=name_func(data) if experiments_in_legends else None,
            **plot_kwargs,
        )

    plt.xlim(1, max_lim)
    plt.ylim(1, max_lim)
    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else r"ToVer $|\mathcal{A}|$ (log)")
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks([10**i for i in range(0, 4)], ["" for i in range(0, 4)])
    else:
        plt.ylabel(
            ylabel if ylabel else r"Baseline $|\mathcal{A}|$ (log)",
            ha="center",
            y=0.43,
        )

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
    show_y_axis: bool = True,
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
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks([])
    else:
        plt.ylabel(
            ylabel
            if ylabel
            else "False Negatives threshold\n(maximal risk for trace not in monitor)"
        )
    plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")
    if title:
        plt.title(title)
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
    show_y_axis: bool = True,
):
    colors = plt.get_cmap("tab20")
    fig, ax = plt.subplots()

    ax.fill_betweenx(
        [0, threshold - fp_slack],
        -1,
        len(exp_data) / bundle + 1,
        color="lightgreen",
        alpha=0.3,
    )
    ax.fill_betweenx(
        [threshold + fn_slack, 1],
        -1,
        len(exp_data) / bundle + 1,
        color="lightcoral",
        alpha=0.5,
    )
    ax.fill_betweenx(
        [threshold - fp_slack, threshold + fn_slack],
        -1,
        len(exp_data) / bundle + 1,
        color="lightgrey",
        alpha=0.5,
    )

    ax.axhline(y=threshold, color="grey", linestyle="--")
    ax.axhline(y=threshold + fn_slack, color="grey", linestyle=r"-", linewidth=1)
    ax.axhline(y=threshold - fp_slack, color="grey", linestyle=r"-", linewidth=1)

    found_thresholds = [
        (
            [
                (
                    data[key]["false_positive"]
                    if key in data and not isinstance(data[key]["false_positive"], str)
                    else 1
                )
                for data in exp_data
            ],
            [
                (
                    data[key]["false_negative"]
                    if key in data and not isinstance(data[key]["false_negative"], str)
                    else 0
                )
                for data in exp_data
            ],
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
                exp_names.append(
                    "/".join([bottom_func(exp_data[k]) for k in range(j, j + bundle)])
                )
                found_bundled_thresholds[-1][0].append(
                    thresh[0][j : j + bundle][max_idx]
                )
                found_bundled_thresholds[-1][1].append(
                    thresh[1][j : j + bundle][max_idx]
                )

    bar_width = 1 / (len(keys) * 2) - 0.05
    index = range(ceil(len(exp_data) / bundle))

    for i, (key_fp_thresholds, key_fn_thresholds) in enumerate(
        found_bundled_thresholds
    ):
        ax.bar(
            [j + i * 2 * bar_width for j in index],
            [-(1 - t) for t in key_fp_thresholds],
            bar_width,
            bottom=1,
            label=f"in monitor traces",
            color=colors(i + 6 % colors.N),
        )
        ax.bar(
            [j + (i * 2 + 1) * bar_width for j in index],
            key_fn_thresholds,
            bar_width,
            label=f"out of monitor traces",
            color=colors(i + 4 % colors.N),
        )

    if not show_y_axis:
        plt.ylabel("")
        ax.set_yticklabels(["" for _ in ax.get_yticks()])
    else:
        plt.ylabel(ylabel if ylabel else "risk threshold")
    plt.xticks(
        [i + bar_width * (len(keys) - 0.5) for i in index],
        exp_names,
        rotation=90,
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    plt.xlim(-0.5, len(exp_data) / bundle)
    if title:
        plt.title(title)

    fig.set_size_inches(*fig_size)

    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def any_frac_to_float(x):
    if isinstance(x, Fraction):
        return float(x)
    return x


def compare_runtime_by_params(
    exp_data,
    param_keys: list[str],
    key: str = "verimon",
    time_key: str = "time",
    figsize: tuple = (20, 10),
    experiments_in_legends=True,
    fit_all=True,
    title=None,
    plot_kwargs={},
):
    """Compare runtime by varying parameters specified in param_keys. Creates a plot of all combinations of param_keys values specified in param_values."""

    # First group data by other parameters in 'experiment' except the param_keys
    # Also find the possible values for each param_key
    param_values = {k: set() for k in param_keys}
    param_groups = {}
    for data in exp_data:
        group_key = tuple(
            (k, str(any_frac_to_float(v)))
            for k, v in data["experiment"].items()
            if k not in param_keys
            and k
            not in [
                "variant",
                "result_json_file",
                "short_name",
                "variant_hash",
                "learn_experiment",
            ]
        ) + (
            (lambda x: x.group(0) if x is not None else None)(
                re.compile(r"intermediate_monitor=(\d+\.\d+)").search(
                    data["experiment"]["variant"]
                )
            ),
        )
        if group_key not in param_groups:
            param_groups[group_key] = {}

        for k in param_keys:
            param_values[k].add(data["experiment"][k])

        param_value = tuple(data["experiment"][k] for k in param_keys)
        if param_value in param_groups[group_key]:
            raise ValueError(
                f"Duplicate data for parameter setting {param_value} in group {group_key}"
            )
        param_groups[group_key][param_value] = data

    for group_key, group_data in param_groups.items():
        if len(group_data) < 4:
            print(
                f"Missing data for {group_key}: only {len(group_data)} combinations present."
            )

    # Now create subplots for each combination of pairs of assiging values to param_keys. Ignore combinations where both param sets are the same.
    param_combinations = list(product(*[sorted(param_values[k]) for k in param_keys]))
    param_param_combinations = list(combinations(param_combinations, 2))
    num_plots = len(param_param_combinations)

    fig, axes = plt.subplots(nrows=ceil(num_plots / 3), ncols=3, figsize=figsize)
    fig.suptitle(
        (
            title
            if title
            else f"Runtime comparison by parameter combinations: {', '.join(param_keys)}"
        ),
        fontsize=12,
    )
    axes = axes.flatten()
    plot_idx = 0

    max_time, timeout, out_of_memory, incorrect, unfinished = calculate_error_lines(
        exp_data,
        lambda d: max(
            d[key][time_key] if key in d and d[key][time_key] else 0,
            d[key][time_key] if key in d and d[key][time_key] else 0,
        ),
    )

    for params1, params2 in param_param_combinations:
        ax = axes[plot_idx]
        plot_idx += 1

        max_x, max_y = 0.0, 0.0

        for group_key, group_data in param_groups.items():
            time1, time2 = None, None
            colors = []
            hashes = [None, None]
            for param in (params1, params2):
                if param in group_data:
                    data = group_data[param]

                    if time_key not in data[key]:
                        print(
                            f"Missing time_key '{time_key}' in data for key '{key}' with the following data: {data[key]}"
                        )
                        continue

                    time_value = data[key][time_key]
                    if isinstance(time_value, str) and "/" in time_value:
                        time_value = float(Fraction(time_value))
                    elif time_value is None:
                        continue

                    # Handle timeout, out-of-memory, incorrect as large values
                    if time_value == "timeout":
                        time_value = timeout
                    elif time_value == "OOM":
                        time_value = out_of_memory
                    elif time_value == "unfinished":
                        time_value = unfinished
                    elif "goal_threshold" not in data[key] and (
                        data[key]["false_positive"] is None
                        or data[key]["false_negative"] is None
                        or data[key]["false_positive"]
                        < data["experiment"]["threshold"]
                        - data["experiment"]["fp_slack"]
                        or data[key]["false_negative"]
                        > data["experiment"]["threshold"]
                        + data["experiment"]["fn_slack"]
                    ):
                        time_value = incorrect

                    if param == params1:
                        time1 = time_value
                        hashes[0] = data["experiment"]["variant_hash"]
                    if param == params2:
                        time2 = time_value
                        hashes[1] = data["experiment"]["variant_hash"]

                    colors.append(data["color"])

                elif param == params1:
                    time1 = unfinished
                elif param == params2:
                    time2 = unfinished

            # Detech it there is a large relative difference in times and print out the group_key
            if time1 is not None and time2 is not None:
                rel_diff = abs(time1 - time2) / max(time1, time2)
                if rel_diff >= 0.9 and max(time1, time2) < timeout:
                    print(
                        f"Large relative difference ({rel_diff:.4f}: times {time1} vs {time2}) for params {params1} vs {params2} in hashes {hashes}"
                    )

            max_x = max(max_x, cast(float, time1) if time1 is not None else 0)
            max_y = max(max_y, cast(float, time2) if time2 is not None else 0)

            if len(colors) == 0:
                continue

            # Now plot the point
            ax.scatter(
                time1,
                time2,
                marker=data["symbol"],
                # color=sorted(colors)[0],
                color="None",
                edgecolor=sorted(colors)[0],
                label=(f"{data['experiment']['name']} {data['experiment']['variant']}"),
                **plot_kwargs,
            )

        # Add even and 10x, 100x slower and faster lines for reference
        # Cut lines so they do not cross the "incorrect" thresholds on either axis
        ax.plot(
            [0, timeout],
            [0, timeout],
            r"-",
            color="0.5",
        )
        ax.plot(
            [0, timeout],
            [0, timeout / 10],
            "--",
            color="0.5",
            label="10x slower",
        )
        ax.plot(
            [0, timeout],
            [0, timeout / 100],
            "--",
            color="0.5",
            label="100x slower",
        )
        ax.plot(
            [0, timeout / 10],
            [0, timeout],
            "--",
            color="0.5",
        )
        ax.plot(
            [0, timeout / 100],
            [0, timeout],
            "--",
            color="0.5",
        )

        # Add non-time lines for timeouts, out-of-memory, incorrect, unfinished
        ax.axline(
            (0, timeout),
            (timeout, timeout),
            color="gray",
            linestyle="--",
            label="Param 2 timeout",
        )
        ax.axline(
            (timeout, 0),
            (timeout, timeout),
            color="gray",
            linestyle="--",
            label="Param 1 timeout",
        )
        ax.axline(
            (0, out_of_memory),
            (out_of_memory, out_of_memory),
            color="gray",
            linestyle="--",
            label="Param 2 out of memory",
        )
        ax.axline(
            (out_of_memory, 0),
            (out_of_memory, out_of_memory),
            color="gray",
            linestyle="--",
            label="Param 1 out of memory",
        )
        ax.axline(
            (0, incorrect),
            (incorrect, incorrect),
            color="gray",
            linestyle="--",
            label="Param 2 incorrect",
        )
        ax.axline(
            (incorrect, 0),
            (incorrect, incorrect),
            color="gray",
            linestyle="--",
            label="Param 1 incorrect",
        )
        ax.axline(
            (0, unfinished),
            (unfinished, unfinished),
            color="gray",
            linestyle="--",
            label="Param 2 unfinished",
        )
        ax.axline(
            (unfinished, 0),
            (unfinished, unfinished),
            color="gray",
            linestyle="--",
            label="Param 1 unfinished",
        )

        ax.loglog()

        ax.set_yticks(
            [10**i for i in range(-1, int(log10(max_time)) + 1)]
            + [timeout, out_of_memory, incorrect, unfinished],
            [f"$10^{{{i}}}$" for i in range(-1, int(log10(max_time)) + 1)]
            + [r"$\infty$", "ERR", r"$\times$", "U"],
        )

        ax.set_xticks(
            [10**i for i in range(-1, int(log10(max_time)) + 1)]
            + [timeout, out_of_memory, incorrect, unfinished],
            [f"$10^{{{i}}}$" for i in range(-1, int(log10(max_time)) + 1)]
            + [r"$\infty$", "ERR", r"$\times$", "U"],
        )

        # Set labels and title
        ax.set_xlabel(", ".join(f"{v}" for k, v in zip(param_keys, params1)))
        ax.set_ylabel(", ".join(f"{v}" for k, v in zip(param_keys, params2)))
        ax.grid(True, which="major", ls="--")
        if fit_all:
            ax.set_xlim(0.1, unfinished * 2)
            ax.set_ylim(0.1, unfinished * 2)
        else:
            ax.set_xlim(0.1, max_x * 1.2)
            ax.set_ylim(0.1, max_y * 1.2)

    return fig, axes


def runtime_by_params(
    exp_data,
    key: str,
    params: list[tuple[tuple[str, str] | list[tuple[str, str]], str]],
    time_key: str = "time",
    title: str | None = None,
    figsize: tuple = (10, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    fit_line: bool = False,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
    experiments_in_legends=True,
    show_y_axis: bool = True,
    plot_kwargs={},
):
    fig, axes = plt.subplots(nrows=ceil(len(params) / 2), ncols=2, figsize=figsize)
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        select_keys, type_plot = param

        if fit_line:
            symbol_points = {}
            for d in exp_data:
                if "fake" in d[key]:
                    continue
                val = d[select_keys[0]][select_keys[1]]
                t = d[key][time_key]
                if (
                    t is None
                    or val is None
                    or isinstance(val, str)
                    or isinstance(t, str)
                ):
                    continue

                if d["symbol"] not in symbol_points:
                    symbol_points[d["symbol"]] = []

                symbol_points[d["symbol"]].append((val, t))

            for symbol, points in symbol_points.items():
                xs, ys = zip(*points)
                # Linear fit
                # fit = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 100)
                # ax.plot(
                #     x_line,
                #     np.polyval(fit, x_line),
                #     "-",
                #     color="green",
                #     marker=symbol,
                #     markevery=0.1,  # Show marker every 20 points
                #     label="Linear fit",
                # )

                # Exponential fit using scipy's curve_fit

                def exp_func(x, a, b):
                    return a * np.exp(b * x)

                try:
                    popt, pcov = curve_fit(exp_func, xs, ys, p0=(1, 0.1))
                    ax.plot(
                        x_line,
                        exp_func(x_line, *popt),
                        "--",
                        marker=symbol,
                        markevery=0.1,
                        color="gray",
                        label=f"Exp fit: {popt[0]:.2f}*e^({popt[1]:.4f}x)",
                    )
                except RuntimeError:
                    # Curve fitting might fail in some cases
                    pass

        if type_plot == "box":
            groups = {}
            for d in exp_data:
                if "fake" in d[key]:
                    continue

                if isinstance(select_keys, list):
                    val = "\n".join(str(d[k][sk]) for k, sk in select_keys)
                else:
                    val = d[select_keys[0]][select_keys[1]]

                val = "None" if val is None else val
                if val not in groups:
                    groups[val] = []
                t = d[key][time_key]
                if t is not None:
                    groups[val].append(t)

            labels, data_list = zip(*groups.items())
            ax.boxplot(data_list, labels=labels, showmeans=True, showfliers=False)
        else:
            for data in exp_data:
                if "fake" in data[key]:
                    continue

                if isinstance(select_keys, list):
                    val = str(tuple(data[k][sk] for k, sk in select_keys))
                else:
                    val = data[select_keys[0]][select_keys[1]]

                ax.plot(
                    val,
                    data[key][time_key],
                    data["symbol"],
                    color=data["color"],
                    label=name_func(data),
                    **plot_kwargs,
                )
            if type_plot == "log":
                ax.set_yscale("log")

        if isinstance(select_keys, list):
            ax.set_xlabel(
                xlabel
                if xlabel
                else ", ".join(
                    sk.replace("_", " ").capitalize() for _, sk in select_keys
                )
            )
        else:
            ax.set_xlabel(
                xlabel if xlabel else select_keys[1].replace("_", " ").capitalize()
            )

        if not show_y_axis:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabel if ylabel else f"{key.capitalize()} Run Time (s log)")
        ax.set_title(
            title
            if title
            else f"{param[1].replace('_', ' ').capitalize()} to {key.capitalize()} Run {time_key.capitalize()}"
        )
        ax.grid(True, which="both", ls="--")

    handles, labels = ax.get_legend_handles_labels()
    if experiments_in_legends:
        fig.legend(handles, labels, bbox_to_anchor=(1, 0.9), loc="upper left")
    plt.tight_layout()
    plt.show()


def runtime_from_logs(logpath: str):
    entries: dict[str, float] = {}
    example_msg: dict[str, str] = {}
    time_pattern = re.compile(r"\((\d+(?:\.\d+))s\)")

    with open(logpath, "r") as f:
        for line in f:
            if "(s)" in line:
                continue
            match = time_pattern.search(line)
            if match:
                try:
                    elapsed = float(match.group(1))
                    _, _, loc, msg = [s.strip() for s in line.split(" - ", 4)]

                    if loc not in entries:
                        entries[loc] = 0
                    entries[loc] += elapsed

                    if loc not in example_msg:
                        example_msg[loc] = msg
                except ValueError:
                    pass
        return entries, example_msg
