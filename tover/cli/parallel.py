"""Execute / report on experiments that are run via GNU ``parallel``.

`tover.cli.experiment` (the default action) writes one command per expanded
experiment variant to ``<base_dir>/commands.txt``; each command is

    python -m tover.cli.parallel --base_dir D --timestamp T --pickle <blob>

where ``<blob>`` is a base64-encoded pickle of the (pre-run) experiment object.
GNU ``parallel`` then executes that file and owns concurrency + timeouts. This
module provides:

* run mode (``--pickle``): decode one experiment and run it, reusing
  ``Experiment.run`` (which writes the analysis json/log and sets the memory
  limit), so the output is identical to what the old scheduler produced.
* report mode (``--report <joblog>``): read ``parallel``'s ``--joblog`` and mark
  the jobs it killed on timeout, so the analysis loader labels them correctly.
"""

import base64
import csv
import json
import os
import pickle
from hashlib import md5
from typing import override

from tap import Tap

from tover.experiments.runner import Experiment
from tover.utils.logger import logger


class ParallelArgs(Tap):
    base_dir: str  # Shared experiment output directory

    # Exactly one of these selects the mode.
    pickle: str | None = None  # base64(pickle(experiment)) -> run that variant
    report: str | None = None  # path to parallel's --joblog -> label timeouts

    timestamp: str = ""  # Shared run timestamp (run mode); read from run_info.json otherwise

    @override
    def process_args(self) -> None:
        if (self.pickle is None) == (self.report is None):
            self.error("Specify exactly one of --pickle (run) or --report (report).")
        if self.pickle is not None and self.timestamp == "":
            self.error("--timestamp is required in run mode.")


def _decode(blob: str):
    return pickle.loads(base64.urlsafe_b64decode(blob.encode()))


def _run(args: ParallelArgs) -> None:
    exp: Experiment = _decode(args.pickle)
    # Seeding is driven by the experiment itself (Experiment.run uses exp.seed,
    # default 0) so it stays consistent across entry points and is recorded.
    exp.run(args.timestamp, args.base_dir, output_to_stdout=False)


def _log_path(base_dir: str, timestamp: str, exp: Experiment) -> str:
    variant_hash = md5(str(exp.variant).encode()).hexdigest()
    return os.path.join(base_dir, "logs", f"{timestamp}_{exp.name}_{variant_hash}.log")


def _pickle_token(command: str) -> str | None:
    parts = command.split()
    if "--pickle" in parts:
        i = parts.index("--pickle")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _report(args: ParallelArgs) -> None:
    info_path = os.path.join(args.base_dir, "run_info.json")
    timestamp = args.timestamp
    if not timestamp and os.path.exists(info_path):
        with open(info_path) as f:
            timestamp = json.load(f)["timestamp"]

    finished = timed_out = other = 0
    with open(args.report) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            command = row.get("Command", "")
            token = _pickle_token(command)
            if token is None:
                continue
            exitval = int(row.get("Exitval") or 0)
            signal = int(row.get("Signal") or 0)

            if exitval == 0 and signal == 0:
                finished += 1
                continue

            # parallel --timeout kills with SIGTERM (signal 15); coreutils
            # `timeout` exits 124. Either means the job was timed out.
            if signal == 15 or exitval == 124:
                timed_out += 1
                exp = _decode(token)
                log_path = _log_path(args.base_dir, timestamp, exp)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                # The analysis loader greps the log for "timed out"; make sure the
                # marker is present so the run is labelled timeout (not OOM).
                with open(log_path, "a") as lf:
                    lf.write("\ntimed out\n")
            else:
                other += 1  # nonzero exit / SIGKILL -> error or OOM (loader default)

    logger.info(
        f"parallel report: {finished} finished, {timed_out} timed out, "
        f"{other} failed (error/OOM)"
    )
    print(
        f"{finished} finished, {timed_out} timed out, {other} failed (error/OOM)"
    )


def main():
    args = ParallelArgs().parse_args()
    if args.pickle is not None:
        _run(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
