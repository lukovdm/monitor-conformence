"""CLI entry point for batch experiment execution."""
import argparse
import os
from datetime import datetime

import yaml

from tover.experiments.config import ObjectGroup
from tover.experiments.runner import LearningExperiment, VerifyExperiment
from tover.experiments.scheduler import run_experiments


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(description="Run ToVer experiments.")
    parser.add_argument(
        "files", type=argparse.FileType("r"), nargs="+",
        help="Path to the experiment YAML config file(s).",
    )
    parser.add_argument(
        "-e", "--experiment", type=str,
        help="Name of the specific experiment to run (default: all).",
    )
    parser.add_argument("-b", "--base-dir", type=str, default="")
    parser.add_argument(
        "-l", "--list-experiments", action="store_true", help="List all available experiments."
    )
    parser.add_argument("-p", "--print", action="store_true", help="Print experiment configs.")
    parser.add_argument(
        "-c", "--concurrent", action="store_true", help="Run experiments concurrently."
    )
    parser.add_argument("--cores", type=int, default=0)
    parser.add_argument(
        "-t", "--timeout", type=int, default=43200,
        help="Per-experiment timeout in seconds (default: 43200 = 12h).",
    )
    parser.add_argument("--debug", action="store_true", help="Pause before running (attach debugger).")
    args = parser.parse_args()

    if args.debug:
        input("Press Enter to continue... " + str(os.getpid()))

    data = [exp for f in args.files for exp in yaml.load(f, Loader=yaml.FullLoader)]

    experiment_type_map = {
        "LearningExperiment": LearningExperiment,
        "VerifyExperiment": VerifyExperiment,
    }

    experiments: list[ObjectGroup] = []
    for group in data:
        exp_type_name = group.pop("type")
        if exp_type_name not in experiment_type_map:
            raise ValueError(f"Unknown experiment type: {exp_type_name}")
        experiments.append(ObjectGroup(experiment_type_map[exp_type_name], **group))

    if args.list_experiments:
        total = 0
        print("Available experiments:")
        for group in experiments:
            objects = list(group.get_objects())
            total += len(objects)
            print(f"- {group.kwargss['name'][0]} ({len(objects)} variants):")
            for exp in objects:
                print(f"\t- {exp.name} {exp.variant}")
        print(f"Total experiments: {total}")
        return

    if args.print:
        for group in experiments:
            for exp in group.get_objects():
                print(f"{exp.name} ({exp.variant}) {str(exp.__dict__)}")
        return

    if args.experiment:
        group = next(
            (g for g in experiments if g.kwargss["name"][0] == args.experiment),
            None,
        )
        if group:
            experiments = [group]
        else:
            print(f"Experiment {args.experiment} not found.")
            return

    if args.base_dir == "":
        filenames = [f.name.split("/")[-1].split(".")[0] for f in args.files]
        base_dir = f"out/exp-{timestamp}-{'-'.join(filenames)}"
    else:
        base_dir = args.base_dir

    os.makedirs(os.path.dirname(base_dir) or ".", exist_ok=True)

    run_experiments(
        experiments,
        timestamp=timestamp,
        base_dir=base_dir,
        concurrent=args.concurrent,
        cores=args.cores,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
