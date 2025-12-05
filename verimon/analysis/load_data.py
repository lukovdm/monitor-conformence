from collections import Counter
from fractions import Fraction
import json
import os
from random import seed
import re
import traceback
from typing import Any, Callable

from matplotlib import pyplot as plt


VERIFY_EXPERIMENTS = 720
LEARN_EXPERIMENTS = 64


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
        if "sampling" not in d:
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
        d["experiment"][
            "short_name"
        ] = f"\\textsc{{{name[0].capitalize()}-{variant_indexes[name]}}}"
        variant_indexes[name] += 1


def add_symbol_color(
    data,
    verify=False,
    color_map: str | None = "tab20",
    col_func: Callable[[Any], str | None] | None = None,
):
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

        if col_func is not None:
            for i, exp in enumerate(data):
                color = col_func(exp)
                if color is not None:
                    exp["color"] = color

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
                            "time": 10**5 * 64,
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
                        "time": 10**5 * 4,
                        "product_time": 0,
                        "paynt_time": 0,
                        "double_check_time": 0,
                    }
            experiment_data.append(data)

    print(
        f"Loaded {len(experiment_data) - unfished_count}/{unfished_count}/{(VERIFY_EXPERIMENTS if 'verify' in path else LEARN_EXPERIMENTS) - len(experiment_data)} ({(len(experiment_data) - unfished_count) / (VERIFY_EXPERIMENTS if 'verify' in path else LEARN_EXPERIMENTS) * 100:.2f}%) JSON files from {path}"
    )
    experiment_data.sort(
        key=lambda x: (x["experiment"]["name"], str(x["experiment"]["variant"]))
    )
    return experiment_data
