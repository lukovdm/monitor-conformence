"""CLI entry point for batch experiment execution."""

import os
from datetime import datetime
from typing import override

import yaml
from tap import Tap

from tover.experiments.config import ObjectGroup
from tover.experiments.runner import LearningExperiment, VerifyExperiment
from tover.experiments.scheduler import run_experiments


class ExperimentArgs(Tap):
    files: list[str]  # Path(s) to experiment YAML config file(s)

    # Filtering
    experiment: str | None = None  # Run only the named experiment (default: all)
    base_dir: str = ""  # Output base directory (default: auto-generated)

    # Actions
    list: bool = False  # List all available experiments and exit
    print: bool = False  # Print experiment configs and exit

    # Execution
    concurrent: bool = False  # Run experiments concurrently
    cores: int = 0  # Number of cores to use (0 = all available)
    timeout: int = 43200  # Per-experiment timeout in seconds (default: 12h)
    debug: bool = False  # Pause before running (for attaching a debugger)

    @override
    def process_args(self) -> None:
        for path in self.files:
            if not os.path.exists(path):
                self.error(f"File not found: {path}")


def main():
    args = ExperimentArgs().parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.debug:
        input("Press Enter to continue... " + str(os.getpid()))

    data = []
    for path in args.files:
        with open(path) as f:
            data.extend(yaml.load(f, Loader=yaml.FullLoader))

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

    if args.list:
        total = 0
        print("Available experiments:")
        for group in experiments:
            objects = list(group.get_objects())
            total += len(objects)
            print(f"- {group.kwargss['name'][0]} ({len(objects)} variants):")
            for exp in objects:
                print(f"\t- {exp}")
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
        filenames = [p.split("/")[-1].split(".")[0] for p in args.files]
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
