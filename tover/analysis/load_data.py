from collections import Counter
from fractions import Fraction
import json
import os
import re
from random import seed
import traceback
from typing import Any, Callable, cast

from matplotlib import pyplot as plt


def clean_dict(d: dict) -> None:
    """Convert rational number strings to Fraction objects in-place."""
    for k, v in d.items():
        if isinstance(v, dict):
            clean_dict(v)
        elif isinstance(v, str):
            try:
                d[k] = Fraction(v)
            except ValueError:
                pass


def clean_data(data: list[dict]) -> None:
    """Convert rational number strings to Fraction objects across all data entries."""
    for d in data:
        clean_dict(d)


def _timeout_or_oom(log_path: str) -> str:
    """Detect whether a log file indicates a timeout or OOM."""
    try:
        with open(log_path, "r") as f:
            log = f.read()
        if "timed out" in log:
            return "timeout"
    except FileNotFoundError:
        pass
    return "OOM"


def add_family_size(data: list[dict]) -> None:
    """Parse log files to find the PAYNT family size for each experiment."""
    for d in data:
        if d["results"] is None:
            continue
        with open(d["log_path"], "r") as f:
            log = f.read()
        match = re.search(r"family size: (\d+e?\d*),", log)
        d["family_size"] = float(match.group(1)) if match else None


def add_learning_rounds(data: list[dict]) -> None:
    """Parse log files to find the number of L* learning rounds for each experiment."""
    for d in data:
        if d["results"] is None:
            continue
        with open(d["log_path"], "r") as f:
            log = f.read()
        match = re.search(r"Learning Rounds:\s+(\d+)", log)
        d["results"]["learning_rounds"] = int(match.group(1)) if match else None


def add_short_names(data: list[dict]) -> None:
    """Assign a short LaTeX name (e.g. \\textsc{A-0}) to each experiment entry."""
    variant_indexes: dict[str, int] = {}
    for d in data:
        name = d["experiment"]["name"]
        if name not in variant_indexes:
            variant_indexes[name] = 0
        d["experiment"][
            "short_name"
        ] = f"\\textsc{{{name[0].capitalize()}-{variant_indexes[name]}}}"
        variant_indexes[name] += 1


def add_symbol_color(
    data: list[dict],
    color_map: str | None = "tab20",
    col_func: Callable[[Any], str | None] | None = None,
) -> tuple:
    """Assign a matplotlib marker symbol and color to each experiment entry."""
    symbols = ["o", "s", "D", ">", "^", "p", "*", "h", "H", "d"]
    seed(42)
    colors = (lambda x: "black") if color_map is None else plt.get_cmap(color_map)

    experiment_names = set(d["experiment"]["name"] for d in data)
    experiment_name_counter = Counter(d["experiment"]["name"] for d in data)
    experiment_symbols = {
        name: symbols[i % len(symbols)]
        for i, name in enumerate(sorted(experiment_names))
    }
    for i, exp in enumerate(data):
        name = exp["experiment"]["name"]
        exp["symbol"] = experiment_symbols.get(name, "+")
        exp["color"] = colors((i % experiment_name_counter[name]) % 20)

    if col_func is not None:
        for exp in data:
            color = col_func(exp)
            if color is not None:
                exp["color"] = color

    return symbols, colors


def load_experiment_data(path: str, expected_total: int | None = None) -> list[dict]:
    """Load all experiment JSON files from path/json/.

    If experiment_metadata.json exists in path, experiments that never produced
    a JSON file are included as not-started placeholders. All entries have a
    ``started`` flag (True/False).

    Results remain under data["results"]. Unfinished/not-started entries get a
    fake placeholder result under data["results"].

    Args:
        path: Directory containing json/ and logs/ subdirectories.
        expected_total: If given, reported in the summary line.
    """
    json_dir = os.path.join(path, "json")
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    experiment_data: list[dict] = []
    unfinished_count = 0
    started_keys: set[tuple[str, str]] = set()

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        log_path = os.path.join(path, "logs", json_file[:-5] + ".log")

        try:
            with open(json_path, "r") as f:
                data: dict = json.load(f)
        except json.JSONDecodeError:
            print(f"Error in {json_file}: JSONDecodeError")
            traceback.print_exc()
            continue

        data["json_path"] = json_path
        data["log_path"] = log_path
        data["error"] = None

        if not data["finished"] and "results" not in data:
            val = _timeout_or_oom(log_path)
            data["results"] = None
            data["error"] = val

        if data["results"] and "error" in cast(dict, data["results"]):
            data["error"] = data["results"]["error"] + data["results"]["msg"]
            data["results"] = None

        if not data["finished"]:
            unfinished_count += 1

        started_keys.add((data["experiment"]["name"], data["experiment"]["variant"]))
        experiment_data.append(data)

    # Fill in experiments from metadata that never produced a JSON file
    metadata_path = os.path.join(path, "experiment_metadata.json")
    not_started_count = 0
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for exp in metadata["experiments"]:
            key = (exp["name"], exp["variant"])
            if key not in started_keys:
                experiment_data.append(
                    {
                        "experiment": exp,
                        "results": None,
                        "error": "not_started",
                    }
                )
                not_started_count += 1

    finished_count = len(experiment_data) - unfinished_count - not_started_count
    total = len(experiment_data)
    if expected_total is not None:
        pct = finished_count / expected_total * 100
        print(
            f"Loaded {finished_count}/{unfinished_count}/{not_started_count} "
            f"(finished/unfinished/not_started, {pct:.2f}%) from {path}"
        )
    else:
        print(
            f"Loaded {finished_count} finished, {unfinished_count} unfinished, "
            f"{not_started_count} not started ({total} total) from {path}"
        )

    experiment_data.sort(
        key=lambda d: (d["experiment"]["name"], str(d["experiment"]["variant"]))
    )

    add_family_size(experiment_data)
    add_learning_rounds(experiment_data)
    add_short_names(experiment_data)
    return experiment_data
