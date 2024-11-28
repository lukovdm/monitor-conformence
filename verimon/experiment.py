import logging
import os
import argparse
from datetime import datetime
from time import time
from verimon.logger import clear_logging, setup_logging, logger
from verimon import loaders
from verimon.MonitorLearning import run_verimon
from verimon.verify import false_positive, false_negative
from verimon.loaders import aalpy_dfa_to_stormvogel

# Define experiments
experiments = [
    {
        "name": "snakes_ladders",
        "n": 100,
        "ladders": {
            1: 38,
            4: 14,
            9: 31,
            28: 64,
            40: 42,
            36: 44,
            51: 67,
            71: 91,
            80: 100,
        },
        "snakes": {
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
        },
        "spec": 'P>0.5 [F<3 "good"]',
        "good_label": "good",
        "horizon": 14,
        "threshold": 0.4,
        "fp_slack": 0.3,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "snakes_ladders",
    },
    {
        "name": "premise_refuel",
        "file": "../tests/premise/refuel.nm",
        "constants": "N=3,ENERGY=3",
        "spec": 'P>0.5 [F<=5 "empty"]',
        "good_label": "empty",
        "horizon": 10,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    {
        "name": "premise_a3",
        "file": "../tests/premise/airportA-3.nm",
        "constants": "DMAX=3,PMAX=3",
        "spec": 'P>0.5 [F<2 "crash" ]',
        "good_label": "crash",
        "horizon": 8,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    {
        "name": "premise_evade",
        "file": "../tests/premise/evade-monitoring.nm",
        "constants": "N=4,RADIUS=2",
        "spec": 'P>0.5 [F<=4 "crash"]',
        "good_label": "crash",
        "horizon": 10,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    {
        "name": "airportb_3",
        "file": "../tests/premise/airportB-3.nm",
        "constants": "DMAX=3,PMAX=3",
        "spec": 'P>0.5 [F<2 "crash" ]',
        "good_label": "crash",
        "horizon": 8,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    {
        "name": "airportb_7",
        "file": "../tests/premise/airportB-7.nm",
        "constants": "DMAX=3,PMAX=3",
        "spec": 'P>0.5 [F<2 "crash" ]',
        "good_label": "crash",
        "horizon": 8,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    {
        "name": "airporta_7",
        "file": "../tests/premise/airportA-7.nm",
        "constants": "DMAX=39,PMAX=3",
        "spec": 'P>0.5 [F<2 "crash" ]',
        "good_label": "crash",
        "horizon": 8,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    {
        "name": "hidden_incentive",
        "file": "../tests/premise/hidden-incentive.nm",
        "constants": "N=3",
        "spec": 'P>0.5 [F<=4 "crash"]',
        "good_label": "crash",
        "horizon": 10,
        "threshold": 0.3,
        "fp_slack": 0.5,
        "fn_slack": 0.05,
        "relative_error": 0.1,
        "use_risk": False,
        "loader": "pomdp",
    },
    # Add more experiments here
]


# Function to run a single experiment
def run_experiment(exp: dict, results_dir: str, logs_dir: str):
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{logs_dir}/{exp['name']}-{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    clear_logging()
    setup_logging(path=log_file)

    logger.info(f"Running experiment: {exp['name']}")
    start_time = time()

    # Make sure models directory exists
    os.makedirs("models", exist_ok=True)

    # Load model
    if exp["loader"] == "pomdp":
        initial_observation, observations, mc, expr_manager = loaders.pomdp_to_mc(
            exp["file"], exp["constants"]
        )
        alphabet = list(observations)
    elif exp["loader"] == "snakes_ladders":
        mc_sl_u_nxn = "../tests/snake_ladder/mc_u_nxn.pm"
        mc, expr_manager = loaders.load_snl_stormpy(
            mc_sl_u_nxn, exp["n"], exp["ladders"], exp["snakes"]
        )
        alphabet = ["init", "normal", "snake", "ladder"]

    # Run Verimon
    learned_monitor = run_verimon(
        mc,
        alphabet,
        exp["spec"],
        exp["good_label"],
        exp["threshold"],
        exp["horizon"],
        exp["relative_error"],
        exp["use_risk"],
        exp["fp_slack"],
        exp["fn_slack"],
        expr_manager,
    )

    # Convert learned monitor to stormvogel model
    mon_cycl = aalpy_dfa_to_stormvogel(learned_monitor)  # type: ignore

    # Verify the model
    result_goal_fp, trace_fp, assignment_fp, product_fp = false_positive(
        mc,
        mon_cycl,
        exp["horizon"],
        expr_manager,
        options={
            "good_spec": exp["spec"],
            "good_label": exp["good_label"],
            "use_risk": exp["use_risk"],
        },
    )  # type: ignore
    result_goal_fn, trace_fn, assignment_fn, product_fn = false_negative(
        mc,
        mon_cycl,
        exp["horizon"],
        expr_manager,
        options={
            "good_spec": exp["spec"],
            "good_label": exp["good_label"],
            "use_risk": exp["use_risk"],
        },
    )  # type: ignore

    # Store results
    result_file = f"{results_dir}/{exp['name']}_{timestamp}.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        f.write(f"Experiment: {exp['name']}\n")
        f.write(f"Paramters: {exp}\n\n")
        f.write(f"Time: {time() - start_time:.2f} s\n\n")
        f.write(f"False Positive Result: {result_goal_fp}\n")
        f.write(f"False Positive Trace: {trace_fp}\n")
        f.write(f"False Positive Assignment: {assignment_fp}\n")
        f.write(f"False Negative Result: {result_goal_fn}\n")
        f.write(f"False Negative Trace: {trace_fn}\n")
        f.write(f"False Negative Assignment: {assignment_fn}\n\n")
        f.write(f"Markov chain: {mc}\n")
        f.write(f"Monitor: {learned_monitor}\n")


# Run experiments based on command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Verimon experiments.")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Name of the experiment to run (default: run all experiments)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store results (default: results)",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory to store logs (default: logs)",
    )
    parser.add_argument(
        "-l",
        "--list_experiments",
        action="store_true",
        help="List all available experiments",
    )
    args = parser.parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for exp in experiments:
            print(f"- {exp['name']}")
    elif args.experiment:
        exp = next((exp for exp in experiments if exp["name"] == args.experiment), None)
        if exp:
            run_experiment(exp, args.results_dir, args.logs_dir)
        else:
            print(f"Experiment {args.experiment} not found.")
    else:
        for exp in experiments:
            run_experiment(exp, args.results_dir, args.logs_dir)
