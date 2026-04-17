"""Parallel experiment execution with timeout handling."""

import json
import logging
import os
import signal
from random import seed, shuffle
from time import time
from typing import cast

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_experiment_with_timeout(arg):
    """Worker function: runs a single experiment with a SIGALRM timeout."""
    exp, timestamp, base_dir, timeout = arg

    logger.info(
        f"Starting experiment {exp.name} ({exp.variant}) with timeout {timeout}s"
    )

    def timeout_handler(signum, frame):
        logger.warning(
            f"Experiment {exp.name} ({exp.variant}) timed out after {timeout} seconds"
        )
        os._exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    seed(0)
    try:
        exp.run(timestamp, base_dir)
    finally:
        signal.alarm(0)


def run_experiments(
    experiments,
    timestamp: str,
    base_dir: str,
    concurrent: bool = False,
    cores: int = 0,
    timeout: int = 43200,
):
    """Run a list of ObjectGroup experiment collections, sequentially or in parallel."""
    all_experiments = []
    for group in experiments:
        for exp in group.get_objects():
            all_experiments.append(exp)

    if not concurrent:
        for exp in all_experiments:
            exp.run(timestamp, base_dir)
        return

    from multiprocessing import Pool, set_start_method

    set_start_method("forkserver")

    if cores == 0:
        cores = cast(int, os.cpu_count()) - 1
    elif cores < 0:
        cores = cast(int, os.cpu_count()) + cores

    os.makedirs(base_dir, exist_ok=True)
    json.dump(
        {"experiments": [exp.__dict__ for exp in all_experiments]},
        open(os.path.join(base_dir, "experiment_metadata.json"), "w"),
        indent=4,
        default=str,
    )

    shuffle(all_experiments)

    args = [(exp, timestamp, base_dir, timeout) for exp in all_experiments]
    with Pool(cores) as pool:
        try:
            for _ in tqdm(
                pool.imap_unordered(run_experiment_with_timeout, args),
                total=len(all_experiments),
            ):
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            logger.warning("Terminating all processes")
            pool.join()
