from abc import ABC, abstractmethod
from email.mime import base
from enum import verify
import json
from operator import le
import os
import argparse
from datetime import datetime
from time import time
from typing import Any, Literal
from setproctitle import getproctitle, setproctitle
import yaml

from verimon.generator import Verifier
from verimon import loaders
from verimon.MonitorLearning import (
    FilteringSUL,
    run_sampling_learning,
    run_trad_learning,
    run_verimon,
)
from verimon.logger import OutputLogger, clear_logging, setup_logging, logger
from verimon.utils import ObjectGroup
from verimon.verify import false_positive, false_negative, true_negative, true_positive
from verimon.loaders import aalpy_dfa_to_stormvogel

from stormvogel.mapping import stormvogel_to_stormpy
from stormpy import export_to_drn

from paynt.family.family import Family


class Experiment(ABC):
    def __init__(
        self,
        name: str | None = None,
        variant: str | None = None,
    ):
        if name is None:
            raise ValueError("Missing required arguments")

        self.name = name
        self.variant = variant

    def __str__(self):
        return f"(Experiment {self.name} ({self.variant}))"

    @abstractmethod
    def run(self, timestamp: str, base_dir: str):
        proc_title = getproctitle()
        proc_title = proc_title.split("<")[0]
        setproctitle(f"{proc_title} <{self.name} ({self.variant})>")
        # Setup logging
        log_file = f"{base_dir}/logs/{timestamp}-{self.name}-{self.variant}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        clear_logging()
        setup_logging(path=log_file)

        # Make sure models directory exists
        os.makedirs(f"{base_dir}/models/", exist_ok=True)


class LearningExperiment(Experiment):
    def __init__(
        self,
        name: str | None = None,
        file: str | None = None,
        spec: str | None = None,
        good_label: str | None = None,
        loader: Literal["pomdp"] | Literal["snakes_ladders"] | None = None,
        parameters: dict[str, Any] | None = None,
        horizon: int = 10,
        threshold: float = 0.3,
        fp_slack: float = 0.2,
        fn_slack: float = 0.05,
        relative_error: float = 0.01,
        use_risk: bool = True,
        use_random_eq: bool = True,
        verimon_walks_per_state: int = 100,
        verimon_walk_length: int = 11,
        learning_algs: list[str] = ["verimon", "sampling"],
        use_horizon_in_filtering: bool = True,
        old_walks_per_state: int = 100000,
        old_walk_length: int | None = None,
        variant: Any | None = None,
    ):
        old_walk_length = (
            old_walk_length if old_walk_length is not None else horizon + 1
        )
        if (
            name is None
            or file is None
            or spec is None
            or good_label is None
            or loader is None
        ):
            raise ValueError("Missing required arguments")

        super().__init__(name, variant)
        self.file = file
        self.parameters = parameters
        self.spec = spec
        self.good_label = good_label
        self.horizon = horizon
        self.threshold = threshold
        self.fp_slack = fp_slack
        self.fn_slack = fn_slack
        self.relative_error = relative_error
        self.use_risk = use_risk
        self.use_random_eq = use_random_eq
        self.verimon_walks_per_state = verimon_walks_per_state
        self.verimon_walk_length = verimon_walk_length
        self.learning_algs = learning_algs
        self.use_horizon_in_filtering = use_horizon_in_filtering
        self.old_walks_per_state = old_walks_per_state
        self.old_walk_length = old_walk_length
        self.loader = loader

    def _run_verimon(self, base_dir, mc, alphabet, initial_observation, expr_manager):
        try:
            logger.info("Running Verimon")
            verimon_start_time = time()
            (verimon_learned_monitor, verimon_info), stats = run_verimon(
                mc,
                alphabet,
                initial_observation,
                self.spec,
                self.good_label,
                self.threshold,
                self.horizon,
                self.relative_error,
                self.use_risk,
                self.fp_slack,
                self.fn_slack,
                expr_manager,
                self.use_random_eq,
                self.verimon_walks_per_state,
                self.verimon_walk_length,
                self.use_horizon_in_filtering,
                base_dir,
            )
            verimon_time = time() - verimon_start_time

            logger.info(
                f"-----------------------------------\n"
                f"Learning Finished.\n"
                f'Learning Rounds:  {verimon_info["learning_rounds"]}\n'
                f'Number of states: {verimon_info["automaton_size"]}\n'
                f"Time (in seconds)\n"
                f'  Total                  : {verimon_info["total_time"]}\n'
                f'  Learning algorithm     : {verimon_info["learning_time"]}\n'
                f'  Conformance checking   : {verimon_info["eq_oracle_time"]}\n'
                f"Learning Algorithm\n"
                f' # Membership Queries    : {verimon_info["queries_learning"]}\n'
                f' # Steps                 : {verimon_info["steps_learning"]}\n'
                f"Equivalence Query\n"
                f' # Membership Queries    : {verimon_info["queries_eq_oracle"]}\n'
                f' # Steps                 : {verimon_info["steps_eq_oracle"]}\n'
                f' # EQ oracle used        : {stats["eq_used"]}\n'
                f' # False positives found : {stats["fp_found"]}\n'
                f' # False negatives found : {stats["fn_found"]}\n'
                f"-----------------------------------"
            )
            verimon_learned_monitor.visualize(
                path=f"{base_dir}/models/monitor_{self.name}_{self.variant}_verimon",
                file_type="dot",
            )
            mon = aalpy_dfa_to_stormvogel(verimon_learned_monitor)
            mon_storm = stormvogel_to_stormpy(mon)
            export_to_drn(
                mon_storm,
                f"{base_dir}/models/monitor_{self.name}_{self.variant}_verimon.drn",
            )
            logger.info("Verifying the verimon learned models")
            fp_verimon_result, fn_verimon_result = self._run_verify(
                mc,
                verimon_learned_monitor,
                expr_manager,
                initial_observation,
                alphabet,
            )

            return {
                "time": verimon_time,
                "false_positive": (
                    fp_verimon_result[0] if fp_verimon_result is not None else None
                ),
                "false_positive_trace": (
                    fp_verimon_result[1] if fp_verimon_result is not None else None
                ),
                "false_negative": (
                    fn_verimon_result[0] if fn_verimon_result is not None else None
                ),
                "false_negative_trace": (
                    fn_verimon_result[1] if fn_verimon_result is not None else None
                ),
                "monitor_states": (len(verimon_learned_monitor.states)),
                "eq_used": stats["eq_used"],
                "fp_found": stats["fp_found"],
                "fn_found": stats["fn_found"],
                "fp_bounds": stats["fp_bounds"],
                "fn_bounds": stats["fn_bounds"],
                "product_time": stats["product_time"],
                "paynt_time": stats["paynt_time"],
                "eq_time": stats["eq_time"],
                "counterexample_time": verimon_info["eq_oracle_time"],
                "lstar_time": verimon_info["learning_time"],
                "dot_file": f"{base_dir}/models/monitor_{self.name}_{self.variant}_verimon.dot",
                "drn_file": f"{base_dir}/models/monitor_{self.name}_{self.variant}_verimon.drn",
            }
        except Exception as e:
            logger.error(f"Error in Verimon: {e}", exc_info=e)
            return {"error": str(e), "msg": e.__repr__()}

    def _run_trad(self, alg, base_dir, mc, alphabet, initial_observation, expr_manager):
        logger.info(f"Running {alg} learning")
        trad_start_time = time()
        if alg == "wrandom":
            trad_learned_monitor, trad_info = run_trad_learning(
                mc,
                alphabet,
                initial_observation,
                self.spec,
                self.threshold,
                self.horizon,
                self.old_walks_per_state,
                self.old_walk_length,
                self.use_risk,
                self.use_horizon_in_filtering,
            )
        elif alg == "sampling":
            trad_learned_monitor, trad_info = run_sampling_learning(
                mc,
                alphabet,
                initial_observation,
                self.spec,
                self.threshold,
                self.horizon,
                self.old_walks_per_state,
                self.old_walk_length,
                self.use_risk,
                self.use_horizon_in_filtering,
            )
        else:
            raise ValueError(f"Unknown learning algorithm: {alg}")

        trad_time = time() - trad_start_time
        logger.info(
            f"-----------------------------------\n"
            f"Learning Finished.\n"
            f'Learning Rounds:  {trad_info["learning_rounds"]}\n'
            f'Number of states: {trad_info["automaton_size"]}\n'
            f"Time (in seconds)\n"
            f'  Total                : {trad_info["total_time"]}\n'
            f'  Learning algorithm   : {trad_info["learning_time"]}\n'
            f'  Conformance checking : {trad_info["eq_oracle_time"]}\n'
            f"Learning Algorithm\n"
            f' # Membership Queries  : {trad_info["queries_learning"]}\n'
            f' # Steps               : {trad_info["steps_learning"]}\n'
            f"Equivalence Query\n"
            f' # Membership Queries  : {trad_info["queries_eq_oracle"]}\n'
            f' # Steps               : {trad_info["steps_eq_oracle"]}\n'
            f"-----------------------------------"
        )
        trad_learned_monitor.visualize(
            path=f"{base_dir}/models/monitor_{self.name}_{self.variant}_{alg}",
            file_type="dot",
        )
        mon = aalpy_dfa_to_stormvogel(trad_learned_monitor)
        mon_storm = stormvogel_to_stormpy(mon)
        export_to_drn(
            mon_storm,
            f"{base_dir}/models/monitor_{self.name}_{self.variant}_{alg}.drn",
        )
        logger.info(f"Verifying the {alg} learned models")
        fp_trad_result, fn_trad_result = self._run_verify(
            mc,
            trad_learned_monitor,
            expr_manager,
            initial_observation,
            alphabet,
        )
        return {
            "time": trad_info["total_time"],
            "lstar_time": trad_info["learning_time"],
            "eq_time": trad_info["eq_oracle_time"],
            "false_positive": fp_trad_result[0] if fp_trad_result else None,
            "false_positive_trace": fp_trad_result[1] if fp_trad_result else None,
            "false_negative": fn_trad_result[0] if fn_trad_result else None,
            "false_negative_trace": fn_trad_result[1] if fn_trad_result else None,
            "monitor_states": len(trad_learned_monitor.states),
            "dot_file": f"{base_dir}/models/monitor_{self.name}_{self.variant}_{alg}.dot",
            "drn_file": f"{base_dir}/models/monitor_{self.name}_{self.variant}_{alg}.drn",
        }

    def _run_verify(
        self, mc, learned_monitor, expr_manager, initial_observation, alphabet
    ) -> tuple[
        tuple[
            float | None, list[str] | None, Family | None, Verifier | None, dict | None
        ],
        tuple[
            float | None, list[str] | None, Family | None, Verifier | None, dict | None
        ],
    ]:
        sul = FilteringSUL(
            mc,
            initial_observation,
            alphabet,
            self.spec,
            self.threshold,
            self.horizon if self.use_horizon_in_filtering else None,
            self.use_risk,
        )

        mon = aalpy_dfa_to_stormvogel(learned_monitor)
        try:
            fp_result = false_positive(
                mc,
                mon,
                self.horizon,
                expr_manager,
                options={
                    "good_spec": self.spec,
                    "good_label": self.good_label,
                    "relative_error": self.relative_error,
                    "use_risk": self.use_risk,
                    "filtering": sul,
                    "model_path": base_dir + "/debug-models/",
                },
            )

        except Exception as e:
            logger.error(f"Exception for fp: {e}")
            fp_result = None, None, None, None, None

        try:
            fn_result = false_negative(
                mc,
                mon,
                self.horizon,
                expr_manager,
                options={
                    "good_spec": self.spec,
                    "good_label": self.good_label,
                    "relative_error": self.relative_error,
                    "use_risk": self.use_risk,
                    "filtering": sul,
                    "model_path": base_dir + "/debug-models/",
                },
            )

        except Exception as e:
            logger.error(f"Exception for fn: {e}")
            fn_result = None, None, None, None, None
        return fp_result, fn_result

    # Function to run a single experiment
    def run(self, timestamp: str, base_dir: str):
        super().run(timestamp, base_dir)

        start_time = time()

        logger.info(
            f"Running learning experiment: {self.name} ({self.variant}) {self.__dict__}"
        )

        # Make sure models directory exists
        os.makedirs(f"{base_dir}/models/", exist_ok=True)

        # Load model
        with OutputLogger():
            if self.loader == "pomdp" and self.parameters:
                (
                    initial_observation,
                    observations,
                    mc,
                    expr_manager,
                ) = loaders.pomdp_to_mc(self.file, self.parameters["constants"])
                alphabet = list(observations)
            elif self.loader == "snakes_ladders" and self.parameters:
                mc_sl_u_nxn = "tests/snake_ladder/mc_u_nxn.pm"
                mc, expr_manager = loaders.load_snl_stormpy(
                    mc_sl_u_nxn,
                    self.parameters["n"],
                    self.parameters["ladders"],
                    self.parameters["snakes"],
                )
                alphabet = ["init", "normal", "snake", "ladder"]
                initial_observation = "init"
            else:
                raise ValueError("Unknown loader or missing parameters")

            results = {}
            for alg in self.learning_algs:
                if alg == "verimon":
                    results["verimon"] = self._run_verimon(
                        base_dir,
                        mc,
                        alphabet,
                        initial_observation,
                        expr_manager,
                    )
                elif alg in ["sampling", "wrandom"]:
                    results[alg] = self._run_trad(
                        alg, base_dir, mc, alphabet, initial_observation, expr_manager
                    )

            total_time = time() - start_time

        # Write results as json
        result_json = {
            "experiment": self.__dict__,
            "time": {
                "total": total_time,
            },
            "mc": {
                "mc_states": len(mc.states),
                "mc_transitions": mc.nr_transitions,
                "mc_observations": len(alphabet),
            },
        } | results
        result_json_file = (
            f"{base_dir}/json/{timestamp}_{self.name}_{self.variant}.json"
        )
        os.makedirs(os.path.dirname(result_json_file), exist_ok=True)
        with open(result_json_file, "w") as f:
            json.dump(result_json, f, indent=4)


class VerifyExperiment(Experiment):
    def __init__(
        self,
        name: str | None = None,
        variant: str | None = None,
        results_file: str | None = None,
        monitor_from: str | None = None,
        monitor: str | None = None,
        mc: str | None = None,
        spec: str | None = None,
        good_label: str | None = None,
        search: (
            Literal["fp"] | Literal["fn"] | Literal["tp"] | Literal["tn"] | None
        ) = None,
        loader: Literal["pomdp"] | Literal["snakes_ladders"] | None = None,
        paynt_strategy: str = "ar",
        parameters: dict[str, Any] | None = None,
        horizon: int = 10,
        relative_error: float = 0.01,
        use_risk: bool = True,
        threshold: float | None = None,
    ):
        super().__init__(name, variant)

        if results_file and monitor_from:
            exp = json.load(open(results_file, "r"))
            self.results_file = results_file
            self.monitor = exp[monitor_from]["drn_file"]
            self.mc = exp["experiment"]["file"]
            self.spec = exp["experiment"]["spec"]
            self.good_label = exp["experiment"]["good_label"]
            self.horizon = exp["experiment"]["horizon"]
            self.relative_error = exp["experiment"]["relative_error"]
            self.use_risk = exp["experiment"]["use_risk"]
            self.threshold = exp["experiment"]["threshold"]
            self.loader = exp["experiment"]["loader"]
            self.parameters = exp["experiment"]["parameters"]
        else:
            if (
                mc is None
                or spec is None
                or good_label is None
                or search is None
                or loader is None
                or parameters is None
            ):
                raise ValueError("Missing required arguments")

            self.monitor = monitor
            self.mc = mc
            self.spec = spec
            self.good_label = good_label
            self.horizon = horizon
            self.relative_error = relative_error
            self.use_risk = use_risk
            self.threshold = threshold
            self.loader = loader
            self.parameters = parameters
            self.results_file = None

        self.threshold = threshold
        self.search = search
        self.paynt_strategy = paynt_strategy

    def run(self, timestamp: str, base_dir: str):
        super().run(timestamp, base_dir)

        start_time = time()

        logger.info(
            f"Running verification experiment: {self.name} ({self.variant}) {self.__dict__}"
        )

        with OutputLogger():
            # Load mc
            if self.loader == "pomdp" and self.parameters:
                (
                    _,
                    alphabet,
                    mc,
                    expr_manager,
                ) = loaders.pomdp_to_mc(self.mc, self.parameters["constants"])
                alphabet = list(alphabet)
            elif self.loader == "snakes_ladders" and self.parameters:
                mc_sl_u_nxn = "tests/snake_ladder/mc_u_nxn.pm"
                mc, expr_manager = loaders.load_snl_stormpy(
                    mc_sl_u_nxn,
                    self.parameters["n"],
                    self.parameters["ladders"],
                    self.parameters["snakes"],
                )
            else:
                raise ValueError("Unknown loader or missing parameters")

            # Load monitor
            if self.monitor is None:
                monitor = loaders.gen_monitor(alphabet, self.horizon)
            elif self.monitor.endswith(".nm") or self.monitor.endswith(".pm"):
                monitor = loaders.load_dfa(self.monitor)
            elif self.monitor.endswith(".drn"):
                monitor = loaders.load_dfa_drn(self.monitor)
            else:
                raise ValueError("Unknown monitor format")

            # Set verification function
            if self.search == "fp":
                verify_func = false_positive
            elif self.search == "fn":
                verify_func = false_negative
            elif self.search == "tp":
                verify_func = true_positive
            elif self.search == "tn":
                verify_func = true_negative
            else:
                raise ValueError(f"Unknown search type: {self.search}")

            try:
                verify_start = time()
                result_goal, trace, assignment, model, stats = verify_func(
                    mc,
                    monitor,
                    self.horizon,
                    expr_manager,
                    threshold=self.threshold,
                    options={
                        "good_spec": self.spec,
                        "good_label": self.good_label,
                        "relative_error": self.relative_error,
                        "use_risk": self.use_risk,
                        "paynt_strategy": self.paynt_strategy,
                        "model_path": base_dir + "/debug-models/",
                    },
                )
                verify_time = time() - verify_start

                if result_goal is not None:
                    if assignment:
                        logger.info(f"Assignment: {assignment}")
                        logger.info(f"PAYNT result: {assignment.analysis_result}")
                    verify_json = {
                        "goal_threshold": result_goal,
                        "trace": trace,
                        "time": verify_time,
                    }
                    if model:
                        with open(
                            f"{base_dir}/models/pomdp_{self.name}_{self.variant}.dot",
                            "w",
                        ) as f:
                            f.write(model.pomdp.to_dot())  # type: ignore
                        export_to_drn(
                            model.pomdp,
                            f"{base_dir}/models/pomdp_{self.name}_{self.variant}.drn",
                        )
                else:
                    verify_json = {
                        "error": "No result",
                        "product_time": stats["product_time"] if stats else None,
                        "paynt_time": stats["paynt_time"] if stats else None,
                    }
            except Exception as e:
                logger.error(f"Error in verification: {e}", exc_info=e)
                verify_json = {"error": str(e)}

            total_time = time() - start_time
            result_json = {
                "experiment": self.__dict__,
                "time": {
                    "total": total_time,
                },
                "mc": {
                    "mc_states": len(mc.states),
                    "mc_transitions": mc.nr_transitions,
                },
                "monitor": {
                    "monitor_states": len(monitor.states),
                },
                "result": verify_json,
            }
            result_json_file = (
                f"{base_dir}/json/{timestamp}_{self.name}_{self.variant}.json"
            )
            os.makedirs(os.path.dirname(result_json_file), exist_ok=True)
            with open(result_json_file, "w") as f:
                json.dump(result_json, f, indent=4)

            logger.info(
                f"Finished verification experiment: {self.name} ({self.variant})"
            )


# Run experiments based on command line arguments
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(description="Run Verimon experiments.")
    parser.add_argument(
        "files",
        type=argparse.FileType("r"),
        nargs="+",
        help="Path to the experiment group file(s)",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Name of the experiment to run (default: run all experiments)",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=str,
        default=f"",
    )
    parser.add_argument(
        "-l",
        "--list_experiments",
        action="store_true",
        help="List all available experiments",
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print the experiment",
    )
    parser.add_argument(
        "-c",
        "--concurrent",
        action="store_true",
        help="Run experiments concurrently",
    )
    args = parser.parse_args()

    data = [exp for f in args.files for exp in yaml.load(f, Loader=yaml.FullLoader)]
    experiments = []
    for group in data:
        if group["type"] == "LearningExperiment":
            exp_type = LearningExperiment
        elif group["type"] == "VerifyExperiment":
            exp_type = VerifyExperiment
        else:
            raise ValueError(f"Unknown experiment type: {group['type']}")

        del group["type"]

        experiments.append(ObjectGroup(exp_type, **group))

    if args.list_experiments:
        print("Available experiments:")
        for group in experiments:
            for exp in group.get_objects():
                print(f"- {exp.name} ({exp.variant})")
        exit()

    if args.print:
        for group in experiments:
            for exp in group.get_objects():
                print(f"{exp.name} ({exp.variant}) {yaml.dump(exp.__dict__)}")
        exit()

    if args.experiment:
        group = next(
            (
                group
                for group in experiments
                if group.kwargss["name"][0] == args.experiment
            ),
            None,
        )
        if group:
            experiments = [group]
        else:
            print(f"Experiment {args.experiment} not found.")

    if args.base_dir == "":
        filenames = [f.name.split("/")[-1].split(".")[0] for f in args.files]
        base_dir = f"stats/exp-{timestamp}-{'-'.join(filenames)}"
    else:
        base_dir = args.base_dir

    os.makedirs(os.path.dirname(base_dir), exist_ok=True)

    if args.concurrent:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(group.prod_class.run, exp, timestamp, base_dir)
                for group in experiments
                for exp in group.get_objects()
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for group in experiments:
            for exp in group.get_objects():
                exp.run(timestamp, base_dir)
