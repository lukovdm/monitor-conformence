"""CLI entry point for a single ToVer monitor learning run."""

import os
from datetime import datetime
from typing import Literal, override

from stormpy import export_to_drn
from stormpy.utility import sharpen
from tap import Tap

from tover.core.learning import LearningMethod, run_tover
from tover.core.synthesis import ConditionalMethod
from tover.models.automata import aalpy_dfa_to_stormpy
from tover.models.pomdp import pomdp_to_stormpy_mc
from tover.models.snakes import (
    SNAKES_OBSERVATION_LABELS,
    SNL_MC_PATH,
    load_snl_stormpy,
    random_snl_board,
)
from tover.utils.logger import clear_logging, logger, setup_logging


class RunArgs(Tap):
    # Model
    file: str | None = None  # Path to the model file
    loader: Literal[
        "pomdp", "snakes_ladders", "snakes_ladders_random", "snakes_ladders_real"
    ]  # Loader type for the model
    exact: bool = False  # Use exact probabilities
    double: bool = False  # Use floating-point probabilities
    constants: str | None = None  # Constants for the POMDP model (e.g. "DMAX=3,PMAX=3")
    n: int | None = None  # Board size for Snakes and Ladders
    ladders: str | None = None  # Ladders in the format 'start1:end1,start2:end2'
    snakes: str | None = None  # Snakes in the format 'start1:end1,start2:end2'

    # Specification
    spec: str  # Property specification (e.g. 'Pmax=? [F<=4 "crash"]')
    good_label: str | None = None  # Label for good/target states

    # Learning
    dont_care: bool = True  # Whether to learn with don't cares
    refrence_language: bool = True  # Whether to use the reference language
    learning_method: LearningMethod = LearningMethod.LSHARP

    # Filtering
    threshold: float = 0.3
    horizon: int = 10
    relative_error: float = 0.01
    fp_slack: float = 0.2
    fn_slack: float = 0.05
    horizon_in_filtering: bool = True

    # Equivalence Oracle
    random_eq: bool = True
    min_length: int = 1
    expected_length: int = 5
    max_seqs: int = 5000
    conditional_method: ConditionalMethod = ConditionalMethod.BISECTION_ADVANCED_PT

    # Timeouts
    solver_timeout: int = 200  # Timeout for the solver in seconds

    # Output
    base_dir: str = f"out/tover-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    export_benchmarks: bool = False

    @override
    def process_args(self) -> None:
        if not self.exact and not self.double:
            self.error("Either --exact or --float must be specified.")
        if self.loader == "snakes_ladders" and (
            self.n is None or self.ladders is None or self.snakes is None
        ):
            self.error(
                "--n, --ladders, and --snakes are required for the snakes_ladders loader."
            )


def make_exact(value: float, exact: bool = True):
    if exact:
        return sharpen(5, value)

    return value


def main():
    args = RunArgs().parse_args()
    exact = args.exact or not args.double

    log_file = f"{args.base_dir}/logfile.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    clear_logging()
    setup_logging(path=log_file)

    if args.loader == "pomdp":
        assert (
            args.constants is not None
        ), "Constants must be provided for the POMDP loader."
        assert (
            args.file is not None
        ), "Model file must be provided for the POMDP loader."
        assert (
            args.spec is not None
        ), "Specification must be provided for the POMDP loader."
        assert (
            args.good_label is not None
        ), "Good label must be provided for the POMDP loader."

        logger.info(
            f"Loading POMDP model from {args.file} with constants {args.constants}"
        )

        initial_observation, observations, mc, expr_manager = pomdp_to_stormpy_mc(
            args.file, args.constants, exact
        )
        alphabet = list(observations)
        good_label = args.good_label
    else:
        if args.loader == "snakes_ladders_random":
            logger.info(f"Loading random Snakes and Ladders model with n={args.n}")
            assert (
                args.n is not None
            ), "Board size n must be provided for random Snakes and Ladders."
            n, ladders, snakes = random_snl_board(args.n)
        elif args.loader == "snakes_ladders_real":
            logger.info("Loading real Snakes and Ladders model")
            n = 100
            ladders = {
                1: 38,
                4: 14,
                9: 31,
                28: 64,
                40: 42,
                36: 44,
                51: 67,
                71: 91,
                80: 100,
            }
            snakes = {
                98: 76,
                95: 75,
                93: 73,
                87: 24,
                64: 60,
                62: 19,
                55: 53,
                49: 11,
                47: 26,
                16: 6,
            }
        else:
            assert (
                args.n is not None
                and args.ladders is not None
                and args.snakes is not None
            )
            logger.info(
                f"Loading Snakes and Ladders model with n={args.n}, ladders={args.ladders}, snakes={args.snakes}"
            )

            ladders = {
                int(k): int(v)
                for k, v in (item.split(":") for item in args.ladders.split(","))
            }
            snakes = {
                int(k): int(v)
                for k, v in (item.split(":") for item in args.snakes.split(","))
            }
            n = args.n

        good_label = "good"
        mc, expr_manager = load_snl_stormpy(SNL_MC_PATH, n, ladders, snakes, exact)
        alphabet = SNAKES_OBSERVATION_LABELS
        initial_observation = "init"

    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, "models"), exist_ok=True)

    (learned_monitor, lstar_stats), stats = run_tover(
        mc=mc,
        alphabet=alphabet,
        initial_observation=initial_observation,
        expression_manager=expr_manager,
        spec=args.spec,
        good_label=good_label,
        threshold=make_exact(args.threshold, exact),
        horizon=args.horizon,
        fp_slack=make_exact(args.fp_slack, exact),
        fn_slack=make_exact(args.fn_slack, exact),
        relative_error=make_exact(args.relative_error, exact),
        use_risk=True,
        use_dont_care=args.dont_care,
        use_reference_language=args.refrence_language,
        use_horizon_in_filtering=args.horizon_in_filtering,
        conditional_method=args.conditional_method,
        learning_method=args.learning_method,
        random_eq_method=(
            {
                "min_length": args.min_length,
                "expected_length": args.expected_length,
                "max_seqs": args.max_seqs,
            }
            if args.random_eq
            else None
        ),
        solver_timeout=args.solver_timeout,
        base_dir=args.base_dir,
        export_benchmarks=args.export_benchmarks,
    )

    monitor_path_base = os.path.join(args.base_dir, "models/monitor_tover")
    learned_monitor.visualize(path=f"{monitor_path_base}.dot", file_type="dot")
    mon = aalpy_dfa_to_stormpy(learned_monitor, exact)
    export_to_drn(mon, f"{monitor_path_base}.drn")

    print("Learning completed.")
    print(
        f"Learned monitor saved to: {monitor_path_base}.dot and {monitor_path_base}.drn"
    )
    print(f"Statistics: {stats}")
    print(f"L* statistics: {lstar_stats}")


if __name__ == "__main__":
    main()
