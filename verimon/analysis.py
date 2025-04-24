from collections import Counter
from fractions import Fraction
import json
from math import ceil
import math
import os
from random import seed
import time
import traceback
from typing import Any
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from verimon.verify import false_positive


VERIFY_EXPERIMENTS = 336
LEARN_EXPERIMENTS = 275


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


def combine_sampling_and_verimon(data, equal_fields):
    for d in data:
        if "sampling" in d["experiment"]["learning_algs"]:
            continue
        for d2 in data:
            if d == d2 or "sampling" not in d2["experiment"]["learning_algs"]:
                continue
            if all(
                d["experiment"][field] == d2["experiment"][field]
                for field in equal_fields
            ):
                if "sampling" in d2:
                    d["sampling"] = d2["sampling"]
                if "verimon" in d2:
                    d["verimon"] = d2["verimon"]
                d["experiment"]["learning_algs"] += d2["experiment"]["learning_algs"]
                d2["ignore"] = True
                break

    return [d for d in data if "ignore" not in d]


def add_family_size(data):
    # Use regex on log file to find family size
    for d in data:
        if "family_size" in d:
            continue
        with open(d["log_path"], "r") as f:
            log = f.read()
        match = re.search(r"family size: (\d+e?\d*),", log)
        if match:
            d["family_size"] = float(match.group(1))
        else:
            d["family_size"] = None


def add_learning_rounds(data):
    # Use regex on log file to find learning rounds
    for d in data:
        if "verimon" in d:
            continue
        with open(d["log_path"], "r") as f:
            log = f.read()
        match = re.search(r"Learning Rounds:  (\d+)", log)
        if match:
            d["sampling"]["learning_rounds"] = int(match.group(1))
        else:
            d["sampling"]["learning_rounds"] = None


def prep_data_for_latex(data):
    for d in data:
        d["experiment"]["name"] = d["experiment"]["name"].replace("_", "\\_")
        d["experiment"]["variant"] = d["experiment"]["variant"].replace("_", "\\_")


def add_short_names(data, verify=False):
    # Short names are first letter of name and index of variant
    variant_indexes: dict[str, int] = {}
    for d in data:
        if "learn_experiment" in d["experiment"]:
            name = d["experiment"]["learn_experiment"]["name"]
        else:
            name = d["experiment"]["name"]

        if name not in variant_indexes:
            variant_indexes[name] = 0
        d["experiment"]["short_name"] = (
            f"\\textsc{{{name[0].capitalize()}-{variant_indexes[name]}}}"
        )
        variant_indexes[name] += 1


def add_symbol_color(data, verify=False, color_map="hsv"):
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
    if color_map is None:
        colors = lambda x: "black"
    else:
        colors = plt.get_cmap(color_map)

    if verify:
        experiment_names = set(
            data["experiment"]["learn_experiment"]["name"] for data in data
        )
        experiment_name_counter = Counter(
            [data["experiment"]["learn_experiment"]["name"] for data in data]
        )
        experiment_symbols = {
            name: symbols[i % len(symbols)]
            for i, name in enumerate(sorted(experiment_names))
        }

        for i, exp in enumerate(data):
            exp["symbol"] = experiment_symbols[
                exp["experiment"]["learn_experiment"]["name"]
            ]
            exp["color"] = colors(
                (
                    i
                    % (
                        experiment_name_counter[
                            exp["experiment"]["learn_experiment"]["name"]
                        ]
                    )
                )
                % 20
            )
    else:
        experiment_names = set(data["experiment"]["name"] for data in data).difference(
            ["compare-trad"]
        )
        experiment_name_counter = Counter([data["experiment"]["name"] for data in data])
        experiment_symbols = {
            name: symbols[i % len(symbols)]
            for i, name in enumerate(sorted(experiment_names))
        }

        for i, exp in enumerate(data):
            exp["symbol"] = experiment_symbols.get(exp["experiment"]["name"], "+")
            exp["color"] = colors(
                (i % (experiment_name_counter[exp["experiment"]["name"]])) % 20
            )

    return symbols, colors


def load_experiment_data(path: str):
    json_files: list[str] = [
        f for f in os.listdir(path + "/json") if f.endswith(".json")
    ]

    experiment_data: list[dict] = []
    unfished_count = 0
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

            data["json_path"] = os.path.join(path + "/json/", json_file)
            data["log_path"] = os.path.join(path + "/logs/", json_file[:-5] + ".log")

            if not data["finished"]:
                unfished_count += 1
                if "time" not in data:
                    data["time"] = {"total": 0}
                if (
                    "learning_algs" in data["experiment"]
                ):  # Fill in gaps of Learning experiment
                    with open(data["log_path"], "r") as f:
                        log = f.read()
                        timed_out = "timed out after 12 hours" in log
                        val = "timeout" if timed_out else "out of memory"
                    if (
                        "verimon" in data["experiment"]["learning_algs"]
                        and "verimon" not in data
                    ):
                        data["verimon"] = {
                            "fake": True,
                            "time": val,
                            "monitor_states": val,
                            "false_positive": val,
                            "false_negative": val,
                        }
                    if (
                        "sampling" in data["experiment"]["learning_algs"]
                        and "sampling" not in data
                    ):
                        data["sampling"] = {
                            "fake": True,
                            "time": val,
                            "monitor_states": val,
                            "false_positive": val,
                            "false_negative": val,
                        }
                elif "learn_experiment" in data["experiment"]:
                    if "result" not in data:
                        data["result"] = {
                            "fake": True,
                            "goal_threshold": -1,
                            "pomdp_states": 0,
                            "time": 10**4,
                            "product_time": 0,
                            "paynt_time": 0,
                            "double_check_time": 0,
                        }

            if "result" in data and "error" in data["result"]:
                if data["result"]["error"] == "No result":
                    data["result"]["goal_threshold"] = None
                else:
                    data["result"] = {
                        "fake": True,
                        "goal_threshold": -1,
                        "pomdp_states": 0,
                        "time": 10**4,
                        "product_time": 0,
                        "paynt_time": 0,
                        "double_check_time": 0,
                    }
            experiment_data.append(data)

    print(
        f"Loaded {len(experiment_data) - unfished_count}/{len(experiment_data)}/{(VERIFY_EXPERIMENTS if 'verify' in path else LEARN_EXPERIMENTS) - len(experiment_data)} ({(len(experiment_data) - unfished_count) / (VERIFY_EXPERIMENTS if 'verify' in path else LEARN_EXPERIMENTS) * 100:.2f}%) JSON files from {path}"
    )
    experiment_data.sort(
        key=lambda x: (x["experiment"]["name"], str(x["experiment"]["variant"]))
    )
    return experiment_data


def generate_verify_table(data, save_figures=False, save_path="./", file_name="verify"):
    preamble = r"""% Auto generated table
\begin{longtable}[c]{@{}llrrrrrrrrrrrrrr@{}}
\caption{Table of all verification experiments.}
\label{tab:fullverifyres}\\                                                                                                                                                                                                                                                                                                        \\
\toprule
 & & \multicolumn{8}{c}{Benchmark} & \multicolumn{5}{c}{\alg}                                                                                                                                                       \\
\cmidrule(lr){3-11}\cmidrule(lr){12-16}
 & & $\lambda_l$ & $h$ & MA/FA & $|\Sts^\mc|$ & $|\ptrans^\mc|$ & $|Z|$ & $|\Sts^\dfa|$ & $|\ptrans^\dfa|$ & $|\lang^{\leq h}|$ & Time (s) & Trans (s) & PAYNT (s) & $|\mdp_{\gtrdot h}|$ & $\lambda^{found}$  \\
\midrule
\endhead"""

    name_map = {
        "airport": r"\textsc{Airport}",
        "evade": r"\textsc{Evade}",
        "refuel": r"\textsc{Refuel}",
        "icy-driving": r"\textsc{Icy-Driving}",
        "hidden_incentive": r"\textsc{Hidden-Incen.}",
        "snakes_ladders": r"\textsc{SnL}",
    }
    file_map = {
        "tests/premise/airportA-3.nm": r"\textsc{AirportA-3}",
        "tests/premise/airportA-7.nm": r"\textsc{AirportA-7}",
        "tests/premise/airportB-3.nm": r"\textsc{AirportB-3}",
        "tests/premise/airportB-7.nm": r"\textsc{AirportB-7}",
        "tests/premise/refuel.nm": r"\textsc{Refuel}",
        "tests/premise/refuelB.nm": r"\textsc{RefuelB}",
    }

    tab_data = [
        [
            file_map[d["experiment"]["mc"]]
            if d["experiment"]["mc"] in file_map
            else name_map[d["experiment"]["learn_experiment"]["name"]],
            d["experiment"]["short_name"],
            d["experiment"]["threshold"]
            if d["experiment"]["threshold"] is not None
            else r"\checkmark",
            d["experiment"]["horizon"],
            "MA" if d["experiment"]["search"] == "fn" else "FA",
            d["mc"]["mc_states"],
            d["mc"]["mc_transitions"],
            d["mc"]["mc_observations"],
            d["monitor"]["monitor_states"],
            d["monitor"]["monitor_transitions"],
            f"$10^{{{int(math.log10(d['family_size']))}}}$"
            if d["family_size"] is not None
            else r"-",
            (round(d["result"]["time"]) if d["result"]["time"] >= 1 else r"$\leq 1s$")
            if "fake" not in d["result"]
            else r"-",
            (
                round(d["result"]["product_time"])
                if d["result"]["product_time"] >= 1
                else r"$\leq 1s$"
            )
            if "fake" not in d["result"]
            else r"-",
            (
                round(d["result"]["paynt_time"])
                if d["result"]["paynt_time"] >= 1
                else r"$\leq 1s$"
            )
            if "fake" not in d["result"]
            else r"-",
            d["result"]["pomdp_states"]
            if "fake" not in d["result"] and d["result"]["pomdp_states"] is not None
            else r"-",
            float(d["result"]["goal_threshold"])
            if d["result"]["goal_threshold"] is not None and "fake" not in d["result"]
            else (r"\checkmark" if "fake" not in d["result"] else r"-"),
        ]
        for d in data
    ]
    tab_with_lines: list[Any] = [tab_data[0]]
    for line in tab_data[1:]:
        if line[0] != tab_with_lines[-1][0]:
            tab_with_lines.append("SEPERATING LINE")
        tab_with_lines.append(line)

    generate_table(preamble, tab_with_lines, save_path, file_name)


def generate_learn_table(data, save_figures=False, save_path="./", file_name="runtime"):
    preamble = r"""% Auto generated table
\begin{longtable}[c]{@{}llrrrrrrrrrrrrrrrr@{}}
\caption{Table of all learn experiments.}
\label{tab:fulllearnexp}\\                                                                                                                                                                                                                                                                                                    \\
\toprule
 & & \multicolumn{6}{c}{Benchmark} & \multicolumn{5}{c}{\alg} & \multicolumn{5}{c}{Baseline}                                                                                                                                                       \\
\cmidrule(lr){3-8}\cmidrule(lr){9-13}\cmidrule(lr){14-18}
 & & $\lambda_u$ & $\lambda_s$ & $h$ & $|\Sts|$ & $|\ptrans|$ & $|Z|$ & Time (s) & Rounds & $|\dfa|$ & $\lambda_u^{\min}$ & $\lambda_s^{\max}$ & Time (s) & $|\dfa|$ & Rounds & $\lambda_u^{\min}$ & $\lambda_s^{\max}$ \\
\midrule
\endhead"""

    name_map = {
        "airport": r"\textsc{Airport}",
        "evade": r"\textsc{Evade}",
        "refuel": r"\textsc{Refuel}",
        "icy-driving": r"\textsc{Icy-Driving}",
        "hidden_incentive": r"\textsc{Hidden-Incen.}",
        "snakes_ladders": r"\textsc{SnL}",
    }
    tab_data = [
        [
            name_map[d["experiment"]["name"]],
            d["experiment"]["short_name"],
            d["experiment"]["threshold"] - d["experiment"]["fp_slack"],
            d["experiment"]["threshold"] + d["experiment"]["fn_slack"],
            d["experiment"]["horizon"],
            d["mc"]["mc_states"],
            d["mc"]["mc_transitions"],
            d["mc"]["mc_observations"],
            (round(d["verimon"]["time"]) if d["verimon"]["time"] >= 1 else r"$\leq 1s$")
            if "fake" not in d["verimon"]
            else r"-",
            len(d["verimon"]["monitors"]) if "fake" not in d["verimon"] else r"-",
            d["verimon"]["monitor_states"] if "fake" not in d["verimon"] else r"-",
            (
                float(d["verimon"]["false_positive"]),
                d["verimon"]["false_positive"]
                < d["experiment"]["threshold"] - d["experiment"]["fp_slack"],
            )
            if "fake" not in d["verimon"]
            else r"-",
            (
                float(d["verimon"]["false_negative"]),
                d["verimon"]["false_negative"]
                > d["experiment"]["threshold"] + d["experiment"]["fn_slack"],
            )
            if "fake" not in d["verimon"]
            else r"-",
            (
                round(d["sampling"]["time"])
                if d["sampling"]["time"] >= 1
                else r"$\leq 1s$"
            )
            if "fake" not in d["sampling"]
            else r"-",
            d["sampling"]["monitor_states"] if "fake" not in d["sampling"] else r"-",
            d["sampling"]["learning_rounds"] if "fake" not in d["sampling"] else r"-",
            (
                float(d["sampling"]["false_positive"]),
                d["sampling"]["false_positive"] < d["experiment"]["threshold"],
            )
            if "fake" not in d["sampling"]
            and d["sampling"]["false_positive"] is not None
            else r"-",
            (
                float(d["sampling"]["false_negative"]),
                d["sampling"]["false_negative"] > d["experiment"]["threshold"],
            )
            if "fake" not in d["sampling"]
            and d["sampling"]["false_negative"] is not None
            else r"-",
        ]
        for d in data
        if "sampling" in d
    ]
    tab_with_lines: list[Any] = [tab_data[0]]
    for line in tab_data[1:]:
        if line[0] != tab_with_lines[-1][0]:
            tab_with_lines.append("SEPERATING LINE")
        tab_with_lines.append(line)

    generate_table(preamble, tab_with_lines, save_path, file_name)


def generate_table(preamble, data, save_path="./", file_name="runtime"):
    with open(f"{save_path}/{file_name}.tex", "w") as f:
        f.write(preamble)
        for line in data:
            if line == "SEPERATING LINE":
                f.write(r"\midrule" + "\n")
            else:
                str_line = []
                for e in line:
                    prefix = ""
                    postfix = ""
                    if isinstance(e, tuple):
                        if e[1]:
                            prefix = r"{\color{red} "
                            postfix = r"}"
                        e = e[0]

                    if isinstance(e, float):
                        str_line.append(f"{prefix}{e:.2f}{postfix}")
                    elif isinstance(e, Fraction):
                        str_line.append(
                            f"{prefix}\\sfrac{{{e.numerator}}}{{{e.denominator}}}{postfix}"
                        )
                    else:
                        str_line.append(str(e))
                f.write(" & ".join(e for e in str_line) + r"\\" + "\n")
        f.write(
            r"""
\bottomrule
\end{longtable}
            """
        )


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
    timeout=10**5,
    out_of_memory=10**5 * 4,
    incorrect=10**5 * 16,
    min_value=1,
):
    max_lim = incorrect * 2
    # offset = 1.3
    # plt.text(
    #     timeout * (1 / offset),
    #     offset * min_value,
    #     "Baseline is faster",
    #     color="gray",
    #     ha="right",
    #     va="bottom",
    #     rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    # )
    # plt.text(
    #     offset * min_value,
    #     timeout * (1 / offset),
    #     "ToVer is faster",
    #     color="gray",
    #     ha="left",
    #     va="top",
    #     rotation=math.degrees(math.atan(figsize[1] / figsize[0])),
    # )

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

    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else "ToVer (s log)")
    if not show_y_axis:
        plt.ylabel("")
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
            data[key1]["monitor_states"]
            if key1 in data and isinstance(data[key1]["monitor_states"], int)
            else 0,
            data[key2]["monitor_states"]
            if key2 in data and isinstance(data[key2]["monitor_states"], int)
            else 0,
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
                data[key]["false_positive"]
                if key in data and not isinstance(data[key]["false_positive"], str)
                else 1
                for data in exp_data
            ],
            [
                data[key]["false_negative"]
                if key in data and not isinstance(data[key]["false_negative"], str)
                else 0
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
    index = range(math.ceil(len(exp_data) / bundle))

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

    fig.set_size_inches(*fig_size)

    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def runtime_by_params(
    exp_data,
    key: str,
    params: list[tuple[str, str, str]],
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

        if fit_line:
            symbol_points = {}
            for d in exp_data:
                if "fake" in d[key]:
                    continue
                val = d[param[0]][param[1]]
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

        if param[2] == "box":
            groups = {}
            for d in exp_data:
                if "fake" in d[key]:
                    continue
                val = d[param[0]][param[1]]
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
                val = data[param[0]][param[1]]
                ax.plot(
                    val,
                    data[key][time_key],
                    data["symbol"],
                    color=data["color"],
                    label=name_func(data),
                    **plot_kwargs,
                )
            if param[2] == "log":
                ax.set_yscale("log")

        ax.set_xlabel(xlabel if xlabel else param[1].replace("_", " ").capitalize())
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
