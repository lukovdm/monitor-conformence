import argparse
import os
from datetime import datetime

from matplotlib.pylab import f
from verimon.loaders import pomdp_to_stormpy_mc, load_snl_stormpy
from verimon.MonitorLearning import run_verimon
from verimon.loaders import aalpy_dfa_to_stormvogel
from stormvogel.mapping import stormvogel_to_stormpy
from stormpy import export_to_drn
from verimon.logger import clear_logging, setup_logging
from stormpy.utility import sharpen


def main():
    parser = argparse.ArgumentParser(description="Run Verimon learning experiments.")

    # Model-related arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the model file (e.g., POMDP or Snakes and Ladders).",
    )
    model_group.add_argument(
        "--loader",
        type=str,
        choices=["pomdp", "snakes_ladders"],
        required=True,
        help="Loader type for the model.",
    )
    model_group.add_argument(
        "--constants",
        type=str,
        help="Constants for the POMDP model (e.g., key=value pairs).",
    )
    model_group.add_argument(
        "--n",
        type=int,
        help="Number of states for Snakes and Ladders.",
    )
    model_group.add_argument(
        "--ladders",
        type=str,
        help="Ladders for Snakes and Ladders in the format 'start1:end1,start2:end2'.",
    )
    model_group.add_argument(
        "--snakes",
        type=str,
        help="Snakes for Snakes and Ladders in the format 'start1:end1,start2:end2'.",
    )

    # Specification-related arguments
    spec_group = parser.add_argument_group("Specification")
    spec_group.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Specification for the model (e.g., a property in Storm format).",
    )
    spec_group.add_argument(
        "--good_label",
        type=str,
        required=True,
        help="Label for good states in the model.",
    )

    # Filtering-related arguments
    filtering_group = parser.add_argument_group("Filtering")
    filtering_group.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold probability for trace inclusion in the monitor (default: 0.3).",
    )
    filtering_group.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Maximum steps for filtering (default: 10).",
    )
    filtering_group.add_argument(
        "--relative_error",
        type=float,
        default=0.01,
        help="Relative error for Paynt (default: 0.01).",
    )
    filtering_group.add_argument(
        "--fp_slack",
        type=float,
        default=0.2,
        help="False positive slack (default: 0.2).",
    )
    filtering_group.add_argument(
        "--fn_slack",
        type=float,
        default=0.05,
        help="False negative slack (default: 0.05).",
    )
    filtering_group.add_argument(
        "--no_horizon_in_filtering",
        action="store_false",
        default=True,
        dest="use_horizon_in_filtering",
        help="Do not use horizon in filtering (default: Use it).",
    )

    # Equivalence oracle-related arguments
    oracle_group = parser.add_argument_group("Equivalence Oracle")
    oracle_group.add_argument(
        "--no_random_eq",
        action="store_false",
        default=True,
        dest="use_random_eq",
        help="Do not use random equivalence oracle (default: Use it).",
    )
    oracle_group.add_argument(
        "--walks_per_state",
        type=int,
        default=100,
        help="Number of walks per state for random equivalence oracle (default: 100).",
    )
    oracle_group.add_argument(
        "--walk_len",
        type=int,
        default=11,
        help="Length of walks for random equivalence oracle (default: 11).",
    )

    # Output-related arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--base_dir",
        type=str,
        default=f"stats/verimon-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Base directory for storing results (default: auto-generated timestamped directory).",
    )
    output_group.add_argument(
        "--export_benchmarks",
        action="store_true",
        default=False,
        help="Export benchmark models during verification (default: False).",
    )

    args = parser.parse_args()

    log_file = f"{args.base_dir}/logfile.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    clear_logging()
    setup_logging(path=log_file)

    # Load model based on loader type
    if args.loader == "pomdp":
        # Pass constants directly as a string
        (
            initial_observation,
            observations,
            mc,
            expr_manager,
        ) = pomdp_to_stormpy_mc(args.file, args.constants, True)
        alphabet = list(observations)
    elif args.loader == "snakes_ladders":
        if args.n is None or args.ladders is None or args.snakes is None:
            raise ValueError(
                "For Snakes and Ladders, --n, --ladders, and --snakes must be specified."
            )
        # Parse ladders and snakes into dictionaries
        ladders = {
            int(k): int(v)
            for k, v in (item.split(":") for item in args.ladders.split(","))
        }
        snakes = {
            int(k): int(v)
            for k, v in (item.split(":") for item in args.snakes.split(","))
        }
        mc_sl_u_nxn = "tests/snake_ladder/mc_u_nxn.pm"
        mc, expr_manager = load_snl_stormpy(
            mc_sl_u_nxn,
            args.n,
            ladders,
            snakes,
            True,
        )
        alphabet = ["init", "normal", "snake", "ladder"]
        initial_observation = "init"
    else:
        raise ValueError("Unknown loader type.")

    # Ensure base directory exists
    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, "models"), exist_ok=True)

    # Run Verimon
    (learned_monitor, lstar_stats), stats = run_verimon(
        mc=mc,
        alphabet=alphabet,
        initial_observation=initial_observation,
        spec=args.spec,
        good_label=args.good_label,
        threshold=sharpen(5, args.threshold),
        horizon=args.horizon,
        relative_error=sharpen(5, args.relative_error),
        use_risk=True,
        fp_slack=sharpen(5, args.fp_slack),
        fn_slack=sharpen(5, args.fn_slack),
        expression_manager=expr_manager,
        use_random_eq=args.use_random_eq,
        walks_per_state=args.walks_per_state,
        walk_len=args.walk_len,
        use_horizon_in_filtering=args.use_horizon_in_filtering,
        base_dir=args.base_dir,
        export_benchmarks=args.export_benchmarks,
    )

    # Store and plot the learned monitor
    monitor_path_base = os.path.join(args.base_dir, "models/monitor_verimon")
    learned_monitor.visualize(path=f"{monitor_path_base}.dot", file_type="dot")
    mon = aalpy_dfa_to_stormvogel(learned_monitor)
    mon_storm = stormvogel_to_stormpy(mon)
    export_to_drn(mon_storm, f"{monitor_path_base}.drn")

    # Output results
    print("Learning completed.")
    print(
        f"Learned monitor saved to: {monitor_path_base}.dot and {monitor_path_base}.drn"
    )
    print(f"Statistics: {stats}")
    print(f"L* statistics: {lstar_stats}")


if __name__ == "__main__":
    main()
