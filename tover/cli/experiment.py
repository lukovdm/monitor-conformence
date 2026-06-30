"""CLI entry point for experiment batches.

The default action expands the YAML experiment grid into one self-contained
command per variant (written to ``<base_dir>/commands.txt``) for execution with
GNU ``parallel``. Each command runs ``tover.cli.parallel`` on a base64-pickled
experiment object. Nothing is executed here -- see ``tover/cli/parallel.py``.
"""

import base64
import json
import os
import pickle
from datetime import datetime
from typing import override

import yaml
from tap import Tap

from tover.experiments.config import ObjectGroup
from tover.experiments.runner import LearningExperiment, VerifyExperiment


class ExperimentArgs(Tap):
    files: list[str]  # Path(s) to experiment YAML config file(s)

    # Filtering
    experiment: str | None = None  # Only the named experiment (default: all)
    base_dir: str = ""  # Output base directory (default: auto-generated)
    timestamp: str = ""  # Run timestamp (default: now); shared by all commands

    # Actions
    list: bool = False  # List all available experiments and exit
    print: bool = False  # Print experiment configs and exit

    # Values used only to fill in the printed `parallel` recipe
    jobs: int = 8  # parallel --jobs
    timeout: int = 60*60*6  # parallel --timeout (seconds)

    debug: bool = False  # Pause before running (for attaching a debugger)

    @override
    def process_args(self) -> None:
        for path in self.files:
            if not os.path.exists(path):
                self.error(f"File not found: {path}")


def main():
    args = ExperimentArgs().parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

    generate_commands(experiments, timestamp, base_dir, args.jobs, args.timeout)


def generate_commands(
    experiments: list[ObjectGroup],
    timestamp: str,
    base_dir: str,
    jobs: int,
    timeout: int,
) -> None:
    """Write one `tover.cli.parallel` command per expanded variant, plus the
    metadata/run-info the executor and report step need, and print a ready-to-run
    GNU `parallel` recipe."""
    # Deterministic expansion (no shuffle) so the metadata matches the commands.
    all_experiments = [exp for group in experiments for exp in group.get_objects()]

    os.makedirs(base_dir, exist_ok=True)

    with open(os.path.join(base_dir, "experiment_metadata.json"), "w") as f:
        json.dump(
            {"experiments": [exp.__dict__ for exp in all_experiments]},
            f,
            indent=4,
            default=str,
        )
    with open(os.path.join(base_dir, "run_info.json"), "w") as f:
        json.dump({"timestamp": timestamp, "base_dir": base_dir}, f, indent=4)

    commands_path = os.path.join(base_dir, "commands.txt")
    with open(commands_path, "w") as f:
        for exp in all_experiments:
            blob = base64.urlsafe_b64encode(pickle.dumps(exp)).decode()
            f.write(
                f"python -m tover.cli.parallel "
                f"--base_dir {base_dir} --timestamp {timestamp} --pickle {blob}\n"
            )

    joblog = os.path.join(base_dir, "joblog.txt")
    print(f"Wrote {len(all_experiments)} commands to {commands_path}")
    print(f"Output directory: {base_dir}\n")
    print("Run with GNU parallel:")
    print(
        f"  parallel --shuf --bar --jobs {jobs} --timeout {timeout} --memfree 15G "
        f"--joblog {joblog} < {commands_path}"
    )
    print("\nThen label any timed-out runs:")
    print(f"  python -m tover.cli.parallel --report {joblog} --base_dir {base_dir}")


if __name__ == "__main__":
    main()
