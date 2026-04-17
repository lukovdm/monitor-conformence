from fractions import Fraction
from math import ceil, log10
import re
from typing import Any, cast
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations, product


# Fields used to match experiments across methods when pairing
DEFAULT_MATCH_FIELDS = [
    "name",
    "file",
    "parameters",
    "horizon",
    "threshold",
    "fp_slack",
    "fn_slack",
    "spec",
    "good_label",
]


def pair_by_benchmark(
    data1: list[dict],
    data2: list[dict],
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
) -> list[tuple[dict, dict]]:
    """Pair entries from data1 and data2 that share the same benchmark.

    Matching is done on match_fields in d["experiment"]. Unmatched entries
    are silently dropped.
    """
    pairs = []
    for d1 in data1:
        for d2 in data2:
            if all(
                d1["experiment"].get(f) == d2["experiment"].get(f) for f in match_fields
            ):
                pairs.append((d1, d2))
                break
    return pairs


def _resolve_time(
    d: dict,
    timeout: float,
    out_of_memory: float,
    incorrect: float,
) -> float:
    """Map an experiment entry to a plot-ready time value.

    Failed experiments map to sentinel values (timeout/OOM/incorrect);
    correct experiments return results["time"].
    """
    if d["results"] is None:
        return timeout if d.get("error") == "timeout" else out_of_memory
    r = d["results"]
    threshold = d["experiment"]["threshold"]
    fp_slack = d["experiment"]["fp_slack"]
    fn_slack = d["experiment"]["fn_slack"]
    fp, fn = r.get("false_positive"), r.get("false_negative")
    if (
        fp is None
        or fn is None
        or fp < threshold - fp_slack
        or fn > threshold + fn_slack
    ):
        return incorrect
    return r["time"]


def calculate_error_lines(
    data: list[dict],
    value_func,
) -> tuple[float, float, float, float, float]:
    """Compute sentinel line positions based on the maximum observed value."""
    max_time = max(
        (
            v
            for d in data
            for v in [value_func(d)]
            if isinstance(v, (int, float, Fraction))
        ),
        default=1.0,
    )
    timeout = max_time * 1.5
    out_of_memory = timeout * 4
    incorrect = out_of_memory * 4
    unfinished = incorrect * 4
    return max_time, timeout, out_of_memory, incorrect, unfinished


def compare_runtimes(
    data1: list[dict[str, Any]],
    data2: list[dict[str, Any]],
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: bool = True,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    experiments_in_legends: bool = True,
    save_figures: bool = False,
    save_path: str = "./",
    file_name: str = "runtime",
    show_y_axis: bool = True,
    plot_kwargs: dict = {},
    min_value: float = 1,
):
    pairs = pair_by_benchmark(data1, data2, match_fields)

    max_time, timeout, out_of_memory, incorrect, unfinished = calculate_error_lines(
        [d for pair in pairs for d in pair],
        lambda d: d["results"]["time"] if d["results"] is not None else 0,
    )

    for d1, d2 in pairs:
        time1 = _resolve_time(d1, timeout, out_of_memory, incorrect)
        time2 = _resolve_time(d2, timeout, out_of_memory, incorrect)
        plt.plot(
            max(time1, min_value),
            max(time2, min_value),
            d1["symbol"],
            color=d1["color"],
            label=name_func(d1, d2) if experiments_in_legends else None,
            **plot_kwargs,
        )

    max_lim = incorrect * 2

    plt.plot([0, max_lim], [0, max_lim], r"-", color="0.5")
    plt.plot([0, max_lim * 10], [0, max_lim], "--", color="0.5", label="10x faster")
    plt.plot([0, max_lim], [0, max_lim * 10], "--", color="0.5")
    plt.plot([0, max_lim * 100], [0, max_lim], ":", color="0.5", label="100x faster")
    plt.plot([0, max_lim], [0, max_lim * 100], ":", color="0.5")
    ax = plt.gca()
    ax.text(
        0.05,
        0.95,
        f"{label1} faster",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.4",
    )
    ax.text(
        0.95,
        0.05,
        f"{label2} faster",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.4",
    )
    for sentinel, label in [
        (timeout, "timeout"),
        (out_of_memory, "out of memory"),
        (incorrect, "incorrect"),
    ]:
        plt.axhline(sentinel, color="gray", linestyle="--", label=f"{label2} {label}")
        plt.axvline(sentinel, color="gray", linestyle="--", label=f"{label1} {label}")

    plt.xlim(min_value, max_lim)
    plt.ylim(min_value, max_lim)

    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else f"{label1} (s log)")

    min_pow = ceil(log10(min_value))
    max_pow = ceil(log10(max_lim))
    tick_positions = [10**i for i in range(min_pow, max_pow)] + [timeout, out_of_memory, incorrect]
    tick_labels = [f"$10^{{{i}}}$" for i in range(min_pow, max_pow)] + [r"$\infty$", "MO", r"$\times$"]
    plt.xticks(tick_positions, tick_labels)
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks(tick_positions, [])
    else:
        plt.ylabel(ylabel if ylabel else f"{label2} (s log)")
        plt.yticks(tick_positions, tick_labels)

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def compare_monitor_sizes(
    data1: list[dict[str, Any]],
    data2: list[dict[str, Any]],
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: bool = True,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    experiments_in_legends: bool = True,
    save_figures: bool = False,
    save_path: str = "./",
    file_name: str = "monitor_sizes",
    show_y_axis: bool = True,
    plot_kwargs: dict = {},
):
    pairs = pair_by_benchmark(data1, data2, match_fields)

    max_mon_states = max(
        (
            v
            for d1, d2 in pairs
            for v in [
                d1["results"]["monitor_states"] if d1["results"] is not None else 0,
                d2["results"]["monitor_states"] if d2["results"] is not None else 0,
            ]
            if isinstance(v, int)
        ),
        default=10,
    )
    max_lim = max_mon_states * 1.5

    plt.plot([0, max_lim], [0, max_lim], "k-", linewidth=1)
    plt.plot([0, max_lim * 10], [0, max_lim], "k--", label="10x smaller")
    plt.plot([0, max_lim], [0, max_lim * 10], "k--")
    plt.plot([0, max_lim * 100], [0, max_lim], "k:", label="100x smaller")
    plt.plot([0, max_lim], [0, max_lim * 100], "k:")
    plt.fill_between(
        [0, max_lim],
        [0, max_lim],
        max_lim,
        color="lightgreen",
        alpha=0.2,
        label=f"{label1} is smaller",
    )
    plt.fill_between(
        [0, max_lim],
        0,
        [0, max_lim],
        color="lightcoral",
        alpha=0.2,
        label=f"{label2} is smaller",
    )

    for d1, d2 in pairs:
        if d1["results"] is None or d2["results"] is None:
            continue
        r1, r2 = d1["results"], d2["results"]
        threshold = d1["experiment"]["threshold"]
        fp_slack = d1["experiment"]["fp_slack"]
        fn_slack = d1["experiment"]["fn_slack"]
        ms1, ms2 = r1.get("monitor_states"), r2.get("monitor_states")
        if not isinstance(ms1, int) or not isinstance(ms2, int):
            continue
        if (
            r1.get("false_positive") is None
            or r1.get("false_negative") is None
            or r1["false_positive"] < threshold - fp_slack
            or r1["false_negative"] > threshold + fn_slack
            or r2.get("false_positive") is None
            or r2.get("false_negative") is None
            or r2["false_positive"] < threshold - fp_slack
            or r2["false_negative"] > threshold + fn_slack
        ):
            continue
        plt.plot(
            ms1,
            ms2,
            d1["symbol"],
            color=d1["color"],
            label=name_func(d1, d2) if experiments_in_legends else None,
            **plot_kwargs,
        )

    plt.xlim(1, max_lim)
    plt.ylim(1, max_lim)
    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else rf"{label1} $|\mathcal{{A}}|$ (log)")
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks([10**i for i in range(0, 4)], [""] * 4)
    else:
        plt.ylabel(
            ylabel if ylabel else rf"{label2} $|\mathcal{{A}}|$ (log)",
            ha="center",
            y=0.43,
        )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def compare_thresholds(
    data1: list[dict],
    data2: list[dict],
    colors,
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    threshold: float = 0.3,
    fn_slack: float = 0.05,
    fp_slack: float = 0.2,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    show_y_axis: bool = True,
):
    pairs = pair_by_benchmark(data1, data2, match_fields)

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

    xmin, ymin = 1.0, 1.0
    xmax, ymax = 0.0, 0.0

    for i, (d1, d2) in enumerate(pairs):
        r1 = d1["results"] or {}
        r2 = d2["results"] or {}
        fp1 = float(r1.get("false_positive") or 0)
        fn1 = float(r1.get("false_negative") or 1)
        fp2 = float(r2.get("false_positive") or 0)
        fn2 = float(r2.get("false_negative") or 1)

        plt.plot([fp1], [fn1], d1["symbol"], color=d1["color"], label=name_func(d1, d2))
        plt.plot([fp1, fp2], [fn1, fn2], color=colors(i % colors.N))
        xmin = min(xmin, fp1, fp2)
        ymin = min(ymin, fn1, fn2)
        xmax = max(xmax, fp1, fp2)
        ymax = max(ymax, fn1, fn2)

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
    datasets: list[tuple[str, list[dict]]],
    bottom_func,
    threshold: float = 0.3,
    fn_slack: float = 0.0,
    fp_slack: float = 0.0,
    bundle: int = 1,
    fig_size: tuple = (10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_figures: bool = False,
    save_path: str = "./",
    file_name: str = "thresholds",
    show_y_axis: bool = True,
):
    """Bar chart of FP/FN thresholds across experiments.

    Args:
        datasets: List of (label, data_list) pairs, one per method to compare.
        bottom_func: Function mapping a data entry to its x-axis label.
    """
    tab_colors = plt.get_cmap("tab20")
    fig, ax = plt.subplots()

    # Use the first dataset's list to determine experiment count
    _, first_data = datasets[0]
    n = len(first_data)

    ax.fill_betweenx(
        [0, threshold - fp_slack], -1, n / bundle + 1, color="lightgreen", alpha=0.3
    )
    ax.fill_betweenx(
        [threshold + fn_slack, 1], -1, n / bundle + 1, color="lightcoral", alpha=0.5
    )
    ax.fill_betweenx(
        [threshold - fp_slack, threshold + fn_slack],
        -1,
        n / bundle + 1,
        color="lightgrey",
        alpha=0.5,
    )
    ax.axhline(y=threshold, color="grey", linestyle="--")
    ax.axhline(y=threshold + fn_slack, color="grey", linestyle="-", linewidth=1)
    ax.axhline(y=threshold - fp_slack, color="grey", linestyle="-", linewidth=1)

    found_thresholds = [
        (
            [
                (
                    float(d["results"]["false_positive"])
                    if d["results"] is not None
                    and not isinstance(d["results"].get("false_positive"), str)
                    and d["results"].get("false_positive") is not None
                    else 1.0
                )
                for d in data
            ],
            [
                (
                    float(d["results"]["false_negative"])
                    if d["results"] is not None
                    and not isinstance(d["results"].get("false_negative"), str)
                    and d["results"].get("false_negative") is not None
                    else 0.0
                )
                for d in data
            ],
        )
        for _, data in datasets
    ]

    if bundle == 1:
        bundled = found_thresholds
        exp_names = [bottom_func(d) for d in first_data]
    else:
        bundled = []
        exp_names = []
        for fp_vals, fn_vals in found_thresholds:
            fps_b, fns_b = [], []
            for j in range(0, len(fn_vals), bundle):
                chunk_fn = fn_vals[j : j + bundle]
                max_idx = chunk_fn.index(max(chunk_fn))
                fps_b.append(fp_vals[j : j + bundle][max_idx])
                fns_b.append(chunk_fn[max_idx])
                exp_names.append(
                    "/".join(bottom_func(first_data[k]) for k in range(j, j + bundle))
                )
            bundled.append((fps_b, fns_b))

    n_datasets = len(datasets)
    bar_width = 1 / (n_datasets * 2) - 0.05
    index = range(ceil(n / bundle))

    for i, ((key_fp, key_fn), (label, _)) in enumerate(zip(bundled, datasets)):
        ax.bar(
            [j + i * 2 * bar_width for j in index],
            [-(1 - t) for t in key_fp],
            bar_width,
            bottom=1,
            label=f"{label} — in monitor",
            color=tab_colors((i * 2 + 6) % tab_colors.N),
        )
        ax.bar(
            [j + (i * 2 + 1) * bar_width for j in index],
            key_fn,
            bar_width,
            label=f"{label} — out of monitor",
            color=tab_colors((i * 2 + 4) % tab_colors.N),
        )

    if not show_y_axis:
        plt.ylabel("")
        ax.set_yticklabels([""] * len(ax.get_yticks()))
    else:
        plt.ylabel(ylabel if ylabel else "risk threshold")
    plt.xticks(
        [i + bar_width * (n_datasets - 0.5) for i in index],
        exp_names,
        rotation=90,
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    plt.xlim(-0.5, n / bundle)
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
    exp_data: list[dict],
    param_keys: list[str],
    time_key: str = "time",
    figsize: tuple = (20, 10),
    fit_all: bool = True,
    title: str | None = None,
    plot_kwargs: dict = {},
):
    """Compare runtime across different parameter settings.

    Creates a scatter subplot for every pair of param_keys value combinations,
    showing how runtime changes between two parameter configurations.
    """
    param_values = {k: set() for k in param_keys}
    param_groups: dict = {}

    for data in exp_data:
        if data["results"] is None:
            continue
        group_key = tuple(
            (k, str(any_frac_to_float(v)))
            for k, v in data["experiment"].items()
            if k not in param_keys
            and k not in ["variant", "result_json_file", "short_name", "variant_hash"]
        ) + (
            (lambda m: m.group(0) if m else None)(
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
            print(f"Missing data for {group_key}: only {len(group_data)} combinations.")

    param_combinations = list(product(*[sorted(param_values[k]) for k in param_keys]))
    param_param_combinations = list(combinations(param_combinations, 2))
    num_plots = len(param_param_combinations)

    fig, axes = plt.subplots(nrows=ceil(num_plots / 3), ncols=3, figsize=figsize)
    fig.suptitle(
        title
        or f"Runtime comparison by parameter combinations: {', '.join(param_keys)}",
        fontsize=12,
    )
    axes = axes.flatten()
    plot_idx = 0

    max_time, timeout, out_of_memory, incorrect, unfinished = calculate_error_lines(
        exp_data,
        lambda d: d["results"][time_key] if d["results"] is not None else 0,
    )

    for params1, params2 in param_param_combinations:
        ax = axes[plot_idx]
        plot_idx += 1
        max_x, max_y = 0.0, 0.0

        for group_key, group_data in param_groups.items():
            time1: float | None = None
            time2: float | None = None
            colors = []
            hashes = [None, None]

            for param in (params1, params2):
                if param in group_data:
                    data = group_data[param]
                    if data["results"] is None:
                        tv = (
                            timeout if data.get("error") == "timeout" else out_of_memory
                        )
                    else:
                        tv = data["results"].get(time_key)
                        if tv is None:
                            continue
                        if isinstance(tv, str) and "/" in tv:
                            tv = float(Fraction(tv))
                        fp = data["results"].get("false_positive")
                        fn = data["results"].get("false_negative")
                        thresh = data["experiment"]["threshold"]
                        if (
                            fp is None
                            or fn is None
                            or fp < thresh - data["experiment"]["fp_slack"]
                            or fn > thresh + data["experiment"]["fn_slack"]
                        ):
                            tv = incorrect

                    if param == params1:
                        time1 = tv
                        hashes[0] = data["experiment"].get("variant_hash")
                    else:
                        time2 = tv
                        hashes[1] = data["experiment"].get("variant_hash")
                    colors.append(data["color"])
                elif param == params1:
                    time1 = unfinished
                else:
                    time2 = unfinished

            if time1 is not None and time2 is not None:
                rel_diff = abs(time1 - time2) / max(time1, time2)
                if rel_diff >= 0.9 and max(time1, time2) < timeout:
                    print(
                        f"Large relative difference ({rel_diff:.4f}: {time1} vs {time2}) "
                        f"for params {params1} vs {params2} in hashes {hashes}"
                    )

            max_x = max(max_x, cast(float, time1) if time1 is not None else 0)
            max_y = max(max_y, cast(float, time2) if time2 is not None else 0)

            if not colors:
                continue
            ax.scatter(
                time1,
                time2,
                marker=data["symbol"],
                color="None",
                edgecolor=sorted(colors)[0],
                label=f"{data['experiment']['name']} {data['experiment']['variant']}",
                **plot_kwargs,
            )

        ax.plot([0, timeout], [0, timeout], r"-", color="0.5")
        ax.plot([0, timeout], [0, timeout / 10], "--", color="0.5", label="10x slower")
        ax.plot([0, timeout], [0, timeout / 100], ":", color="0.5", label="100x slower")
        ax.plot([0, timeout / 10], [0, timeout], "--", color="0.5")
        ax.plot([0, timeout / 100], [0, timeout], ":", color="0.5")
        for sentinel, label in [
            (timeout, "timeout"),
            (out_of_memory, "OOM"),
            (incorrect, "incorrect"),
            (unfinished, "unfinished"),
        ]:
            ax.axline(
                (0, sentinel),
                (sentinel, sentinel),
                color="gray",
                linestyle="--",
                label=f"Param 2 {label}",
            )
            ax.axline(
                (sentinel, 0),
                (sentinel, sentinel),
                color="gray",
                linestyle="--",
                label=f"Param 1 {label}",
            )

        ax.loglog()
        ticks = [10**i for i in range(-1, int(log10(max_time)) + 1)] + [
            timeout,
            out_of_memory,
            incorrect,
            unfinished,
        ]
        tick_labels = [f"$10^{{{i}}}$" for i in range(-1, int(log10(max_time)) + 1)] + [
            r"$\infty$",
            "ERR",
            r"$\times$",
            "U",
        ]
        ax.set_yticks(ticks, tick_labels)
        ax.set_xticks(ticks, tick_labels)
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
    exp_data: list[dict],
    params: list[tuple[tuple[str, str] | list[tuple[str, str]], str]],
    time_key: str = "time",
    title: str | None = None,
    figsize: tuple = (10, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    fit_line: bool = False,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
    experiments_in_legends: bool = True,
    show_y_axis: bool = True,
    plot_kwargs: dict = {},
):
    fig, axes = plt.subplots(nrows=ceil(len(params) / 2), ncols=2, figsize=figsize)
    axes = axes.flatten()

    for i, (select_keys, type_plot) in enumerate(params):
        ax = axes[i]

        if fit_line:
            symbol_points: dict = {}
            for d in exp_data:
                if d["results"] is None:
                    continue
                val = d[select_keys[0]][select_keys[1]]
                t = d["results"][time_key]
                if (
                    t is None
                    or val is None
                    or isinstance(val, str)
                    or isinstance(t, str)
                ):
                    continue
                symbol_points.setdefault(d["symbol"], []).append((val, t))

            for symbol, points in symbol_points.items():
                xs, ys = zip(*points)
                x_line = np.linspace(min(xs), max(xs), 100)

                def exp_func(x, a, b):
                    return a * np.exp(b * x)

                try:
                    popt, _ = curve_fit(exp_func, xs, ys, p0=(1, 0.1))
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
                    pass

        if type_plot == "box":
            groups: dict = {}
            for d in exp_data:
                if d["results"] is None:
                    continue
                if isinstance(select_keys, list):
                    val = "\n".join(str(d[k][sk]) for k, sk in select_keys)
                else:
                    val = d[select_keys[0]][select_keys[1]]
                val = "None" if val is None else val
                t = d["results"][time_key]
                if t is not None:
                    groups.setdefault(str(val), []).append(t)
            labels, data_list = zip(*groups.items())
            ax.boxplot(data_list, labels=labels, showmeans=True, showfliers=False)
        else:
            for data in exp_data:
                if data["results"] is None:
                    continue
                if isinstance(select_keys, list):
                    val = str(tuple(data[k][sk] for k, sk in select_keys))
                else:
                    val = data[select_keys[0]][select_keys[1]]
                ax.plot(
                    val,
                    data["results"][time_key],
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
                or ", ".join(sk.replace("_", " ").capitalize() for _, sk in select_keys)
            )
        else:
            ax.set_xlabel(xlabel or select_keys[1].replace("_", " ").capitalize())

        if not show_y_axis:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabel or f"Run Time (s log)")
        ax.set_title(
            title or f"{type_plot.replace('_', ' ').capitalize()} run {time_key}"
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
                    entries[loc] = entries.get(loc, 0) + elapsed
                    example_msg.setdefault(loc, msg)
                except ValueError:
                    pass
    return entries, example_msg
