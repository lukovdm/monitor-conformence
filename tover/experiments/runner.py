import json
import os
import resource
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import md5
from time import time
from typing import Any, Literal, final, override

from aalpy import Dfa
from setproctitle import getproctitle, setproctitle
from stormpy import (
    ExpressionManager,
    SparseDtmc,
    SparseExactDtmc,
    SparseExactMdp,
    SparseMdp,
    export_to_drn,
)
from stormpy.utility import sharpen
from stormvogel.mapping import stormvogel_to_stormpy

from tover.core.learning import (
    LearningMethod,
    run_sampling_learning,
    run_tover,
    run_trad_learning,
)
from tover.core.oracles import OracleStats
from tover.core.sul import FilteringSUL
from tover.core.synthesis import ConditionalMethod
from tover.core.verification import (
    false_negative,
    false_positive,
    true_negative,
    true_positive,
)
from tover.models import pomdp as pomdp_loader
from tover.models import snakes as snl_loader
from tover.models.automata import aalpy_dfa_to_stormpy, aalpy_dfa_to_stormvogel
from tover.utils.helpers import str_to_float
from tover.utils.logger import OutputLogger, clear_logging, logger, setup_logging


class Experiment(ABC):
    def __init__(
        self,
        name: str,
        variant: str | None = None,
    ):
        self.name: str = name
        self.variant: str | None = variant
        self.result_json_file: str = ""

    @override
    def __str__(self):
        return f"(Experiment {self.name} ({self.variant}))"

    @abstractmethod
    def run(self, timestamp: str, base_dir: str):
        resource.setrlimit(
            resource.RLIMIT_AS,
            (1024 * 1024 * 1024 * 15, resource.RLIM_INFINITY),  # 15 GiB limit
        )
        proc_title = getproctitle().split("<")[0]
        setproctitle(f"{proc_title} <{self.name} {self.variant}>")

        self.variant_hash: str = md5(str(self.variant).encode()).hexdigest()
        log_file = f"{base_dir}/logs/{timestamp}_{self.name}_{self.variant_hash}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        clear_logging()
        setup_logging(path=log_file, output_to_stdout=False)

        os.makedirs(f"{base_dir}/models/", exist_ok=True)

        self.result_json_file = (
            f"{base_dir}/json/{timestamp}_{self.name}_{self.variant_hash}.json"
        )
        self.write_results(finished=False)

    def write_results(
        self,
        finished: bool,
        total_time: float | None = None,
        results: dict[str, Any] | None = None,
        mc: SparseDtmc | SparseExactDtmc | None = None,
        alphabet: list[str] | None = None,
        monitor: SparseMdp | SparseExactMdp | None = None,
    ):
        result_json: dict[str, Any] = {
            "experiment": self.__dict__,
            "finished": finished,
        }
        if total_time is not None:
            result_json["time"] = {"total": total_time}
        if mc is not None and alphabet is not None:
            result_json["mc"] = {
                "mc_states": len(mc.states),
                "mc_transitions": mc.nr_transitions,
                "mc_observations": len(alphabet),
            }
        if monitor is not None:
            result_json["monitor"] = {
                "monitor_states": len(monitor.states),
                "monitor_transitions": monitor.nr_transitions,
            }
        if results is not None:
            result_json.update(results)

        os.makedirs(name=os.path.dirname(self.result_json_file), exist_ok=True)
        with open(self.result_json_file, "w") as f:
            json.dump(result_json, f, indent=4, default=str)


@final
class LearningExperiment(Experiment):
    def __init__(
        self,
        # Required
        name: str,
        file: str,
        spec: str,
        good_label: str,
        loader: Literal["pomdp", "snakes_ladders"],
        # Optional model
        parameters: dict[str, Any] | None = None,
        # Monitor parameters
        horizon: int = 10,
        threshold: float = 0.3,
        slack: tuple[float, float] = (0.2, 0.05),
        relative_error: float = 0.01,
        # Behavior flags
        use_risk: bool = True,
        use_horizon_in_filtering: bool = True,
        conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
        use_exact: bool = False,
        # Learning
        learning_algs: list[str] | None = None,
        learning_method: LearningMethod = LearningMethod.LSHARP,
        random_eq_method: dict[str, int] | None = None,
        # Baseline comparison
        old_walks_per_state: int = 100000,
        old_walk_length: int | None = None,
        # Variant
        variant: str | None = None,
    ):
        super().__init__(name, variant)
        self.file = file
        self.parameters = parameters
        self.spec = spec
        self.good_label = good_label
        self.horizon = horizon
        self.threshold = threshold
        self.fp_slack = slack[0]
        self.fn_slack = slack[1]
        self.relative_error = relative_error
        self.use_risk = use_risk
        self.use_horizon_in_filtering = use_horizon_in_filtering
        self.conditional_method = conditional_method
        self.loader = loader
        self.use_exact = use_exact
        self.learning_algs = learning_algs if learning_algs is not None else ["verimon"]
        self.learning_method = learning_method
        self.random_eq_method = random_eq_method
        self.old_walks_per_state = old_walks_per_state
        self.old_walk_length = (
            old_walk_length if old_walk_length is not None else horizon + 1
        )

    def _load_model(self):
        if self.loader == "pomdp" and self.parameters:
            return pomdp_loader.pomdp_to_stormpy_mc(
                self.file, self.parameters["constants"], self.use_exact
            )
        elif self.loader == "snakes_ladders" and self.parameters:
            mc, expr_manager = snl_loader.load_snl_stormpy(
                snl_loader.SNL_MC_PATH,
                self.parameters["n"],
                self.parameters["ladders"],
                self.parameters["snakes"],
                self.use_exact,
            )
            alphabet = snl_loader.SNAKES_OBSERVATION_LABELS
            initial_observation = "init"
            return initial_observation, set(alphabet), mc, expr_manager
        else:
            raise ValueError("Unknown loader or missing parameters")

    def _export_monitor(self, monitor: Dfa[str], base_dir: str, alg_suffix: str) -> str:
        """Visualize and export a learned monitor to dot and drn formats."""
        path_base = (
            f"{base_dir}/models/monitor_{self.name}_{self.variant_hash}_{alg_suffix}"
        )
        monitor.visualize(path=path_base, file_type="dot")
        mon_sv = aalpy_dfa_to_stormvogel(monitor)
        mon_storm = stormvogel_to_stormpy(mon_sv)
        export_to_drn(mon_storm, f"{path_base}.drn")
        return path_base

    def _log_learning_summary(
        self, info: dict[str, Any], extra: OracleStats | None = None
    ) -> None:
        extra_lines = ""
        if extra:
            extra_lines = (
                f" # EQ oracle used        : {extra.eq_used}\n"
                f" # False positives found : {extra.fp_found}\n"
                f" # False negatives found : {extra.fn_found}\n"
            )
        logger.info(
            f"-----------------------------------\n"
            f"Learning Finished.\n"
            f"Learning Rounds:  {info['learning_rounds']}\n"
            f"Number of states: {info['automaton_size']}\n"
            f"Time (in seconds)\n"
            f"  Total                  : {info['total_time']}\n"
            f"  Learning algorithm     : {info['learning_time']}\n"
            f"  Conformance checking   : {info['eq_oracle_time']}\n"
            f"Learning Algorithm\n"
            f" # Membership Queries    : {info['queries_learning']}\n"
            f" # Steps                 : {info['steps_learning']}\n"
            f"Equivalence Query\n"
            f" # Membership Queries    : {info['queries_eq_oracle']}\n"
            f" # Steps                 : {info['steps_eq_oracle']}\n"
            + extra_lines
            + f"-----------------------------------"
        )

    def _run_verify(
        self,
        mc: SparseDtmc | SparseExactDtmc,
        learned_monitor: Dfa[str],
        expr_manager: ExpressionManager,
        initial_observation: str,
        alphabet: list[str],
        base_dir: str,
    ) -> tuple[Any, Any]:
        sul = FilteringSUL(
            mc,
            initial_observation,
            alphabet,
            self.spec,
            self.threshold,
            self.horizon if self.use_horizon_in_filtering else None,
            self.use_risk,
        )
        mon = aalpy_dfa_to_stormpy(learned_monitor, mc.is_exact)
        verify_opts = {
            "good_spec": self.spec,
            "good_label": self.good_label,
            "relative_error": self.relative_error,
            "use_risk": self.use_risk,
            "filtering": sul,
            "model_path": base_dir + "/debug-models/",
            "export_benchmarks": False,
            "conditional_method": self.conditional_method,
        }

        try:
            fp_result = false_positive(
                mc, mon, self.horizon, expr_manager, options=verify_opts
            )
        except Exception:
            logger.error(f"Exception for fp: {traceback.format_exc()}")
            fp_result = None, None, {}

        try:
            fn_result = false_negative(
                mc, mon, self.horizon, expr_manager, options=verify_opts
            )
        except Exception:
            logger.error(f"Exception for fn: {traceback.format_exc()}")
            fn_result = None, None, {}

        return fp_result, fn_result

    def _build_verify_result_dict(self, fp_result, fn_result) -> dict:
        fp_cex, fn_cex = fp_result[0], fn_result[0]
        return {
            "false_positive": fp_cex[0] if fp_cex is not None else None,
            "false_positive_trace": fp_cex[1] if fp_cex is not None else None,
            "false_negative": fn_cex[0] if fn_cex is not None else None,
            "false_negative_trace": fn_cex[1] if fn_cex is not None else None,
        }

    def _run_tover(
        self, base_dir, mc, alphabet, initial_observation, expr_manager
    ) -> dict:
        try:
            logger.info("Running ToVer")
            start_time = time()
            (learned_monitor, lstar_info), stats = run_tover(
                mc=mc,
                alphabet=alphabet,
                initial_observation=initial_observation,
                expression_manager=expr_manager,
                spec=self.spec,
                good_label=self.good_label,
                threshold=self.threshold,
                horizon=self.horizon,
                fp_slack=self.fp_slack,
                fn_slack=self.fn_slack,
                relative_error=self.relative_error,
                use_risk=self.use_risk,
                use_horizon_in_filtering=self.use_horizon_in_filtering,
                conditional_method=self.conditional_method,
                learning_method=self.learning_method,
                random_eq_method=self.random_eq_method,
                base_dir=base_dir,
            )
            elapsed = time() - start_time

            self._log_learning_summary(lstar_info, stats)
            path_base = self._export_monitor(learned_monitor, base_dir, "tover")

            logger.info("Verifying the ToVer learned monitor")
            fp_result, fn_result = self._run_verify(
                mc,
                learned_monitor,
                expr_manager,
                initial_observation,
                alphabet,
                base_dir,
            )

            return {
                "time": elapsed,
                "monitor_states": len(learned_monitor.states),
                "eq_used": stats.eq_used,
                "fp_found": stats.fp_found,
                "fn_found": stats.fn_found,
                "fp_bounds": stats.fp_bounds,
                "fn_bounds": stats.fn_bounds,
                "monitors": stats.monitors,
                "product_time": stats.product_time,
                "paynt_time": stats.paynt_time,
                "eq_time": stats.eq_time,
                "counterexample_time": lstar_info["eq_oracle_time"],
                "lstar_time": lstar_info["learning_time"],
                "dot_file": f"{path_base}.dot",
                "drn_file": f"{path_base}.drn",
                **self._build_verify_result_dict(fp_result, fn_result),
            }
        except Exception as e:
            logger.error(f"Error in ToVer: {traceback.format_exc()}")
            return {"error": str(e), "msg": e.__repr__()}

    def _run_baseline(
        self,
        alg: str,
        base_dir: str,
        mc: SparseDtmc | SparseExactDtmc,
        alphabet: list[str],
        initial_observation: str,
        expr_manager: ExpressionManager,
    ) -> dict[str, Any]:
        logger.info(f"Running {alg} learning")
        if alg == "wrandom":
            learned_monitor, lstar_info = run_trad_learning(
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
            learned_monitor, lstar_info = run_sampling_learning(
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

        self._log_learning_summary(lstar_info)
        path_base = self._export_monitor(learned_monitor, base_dir, alg)

        logger.info(f"Verifying the {alg} learned monitor")
        fp_result, fn_result = self._run_verify(
            mc, learned_monitor, expr_manager, initial_observation, alphabet, base_dir
        )

        return {
            "time": lstar_info["total_time"],
            "lstar_time": lstar_info["learning_time"],
            "eq_time": lstar_info["eq_oracle_time"],
            "monitor_states": len(learned_monitor.states),
            "dot_file": f"{path_base}.dot",
            "drn_file": f"{path_base}.drn",
            **self._build_verify_result_dict(fp_result, fn_result),
        }

    @override
    def run(self, timestamp: str, base_dir: str):
        if self.use_exact:
            self.threshold = sharpen(3, self.threshold)
            self.fp_slack = sharpen(5, self.fp_slack)
            self.fn_slack = sharpen(5, self.fn_slack)
            self.relative_error = sharpen(5, self.relative_error)

        super().run(timestamp, base_dir)
        start_time = time()
        logger.info(
            f"Running learning experiment: {self.name} ({self.variant}) {self.__dict__}"
        )
        os.makedirs(f"{base_dir}/models/", exist_ok=True)

        with OutputLogger():
            initial_observation, observations, mc, expr_manager = self._load_model()
            alphabet = list(observations)
            self.write_results(finished=False, mc=mc, alphabet=alphabet)

            results = {}
            for alg in self.learning_algs:
                if alg == "verimon":
                    results["verimon"] = self._run_tover(
                        base_dir, mc, alphabet, initial_observation, expr_manager
                    )
                elif alg in ["sampling", "wrandom"]:
                    results[alg] = self._run_baseline(
                        alg, base_dir, mc, alphabet, initial_observation, expr_manager
                    )
                self.write_results(
                    finished=False, results=results, mc=mc, alphabet=alphabet
                )

            total_time = time() - start_time
            self.write_results(
                finished=True,
                total_time=total_time,
                results=results,
                mc=mc,
                alphabet=alphabet,
            )
            logger.info(f"Finished learning experiment {self.name} ({self.variant})")


@final
class VerifyExperiment(Experiment):
    def __init__(
        self,
        # Required
        name: str,
        search: Literal["fp", "fn", "tp", "tn"],
        # Required when results_file is not given
        mc: str | None = None,
        spec: str | None = None,
        good_label: str | None = None,
        loader: Literal["pomdp", "snakes_ladders"] | None = None,
        parameters: dict[str, Any] | None = None,
        # Load from prior learning experiment result
        results_file: str | None = None,
        intermediate_monitor: float | None = None,
        monitor_from: str | None = None,
        monitor: str | None = None,
        # Monitor parameters
        horizon: int = 10,
        threshold: float | None = None,
        relative_error: float = 0.01,
        # Behavior flags
        use_risk: bool = True,
        use_exact: bool = False,
        paynt_strategy: str = "ar",
        conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
        # Variant
        variant: str | None = None,
    ):
        super().__init__(name, variant)

        self.stop = False
        if results_file:
            exp = json.load(open(results_file, "r"))
            self.results_file = results_file
            if not exp["finished"] or "monitors" not in exp["verimon"]:
                logger.warning(f"Experiment not finished: {results_file}")
                self.stop = True
                return

            if intermediate_monitor is not None:
                mon_idx = int(
                    (len(exp["verimon"]["monitors"]) - 1) * intermediate_monitor
                )
                for i, mon in enumerate(exp["verimon"]["monitors"][mon_idx:]):
                    if mon is not None:
                        self.mon_percent = (i + mon_idx) / len(
                            exp["verimon"]["monitors"]
                        )
                        self.monitor = mon
                        break
            else:
                self.monitor = exp[monitor_from]["drn_file"]
                self.mon_percent = 1.0

            self.intermediate_monitor = intermediate_monitor
            self.mc = exp["experiment"]["file"]
            self.spec = exp["experiment"]["spec"]
            self.good_label = exp["experiment"]["good_label"]
            self.horizon = exp["experiment"]["horizon"]
            self.relative_error = str_to_float(exp["experiment"]["relative_error"])
            self.use_risk = exp["experiment"]["use_risk"]
            self.loader = exp["experiment"]["loader"]
            self.parameters = exp["experiment"]["parameters"]
            self.use_exact = exp["experiment"]["use_exact"]
            self.learn_experiment = exp["experiment"]
        else:
            if (
                mc is None
                or spec is None
                or good_label is None
                or loader is None
                or parameters is None
            ):
                raise ValueError(
                    f"mc, spec, good_label, loader, and parameters are required when results_file is not given"
                )

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
            self.use_exact = use_exact

        self.threshold = threshold
        self.search = search
        self.paynt_strategy = paynt_strategy
        self.use_exact = use_exact
        self.conditional_method = conditional_method

    def _load_model(self):
        if self.loader == "pomdp" and self.parameters:
            return pomdp_loader.pomdp_to_stormpy_mc(
                self.mc, self.parameters["constants"], self.use_exact
            )
        elif self.loader == "snakes_ladders" and self.parameters:
            mc, expr_manager = snl_loader.load_snl_stormpy(
                snl_loader.SNL_MC_PATH,
                self.parameters["n"],
                self.parameters["ladders"],
                self.parameters["snakes"],
                self.use_exact,
            )
            alphabet = snl_loader.SNAKES_OBSERVATION_LABELS
            initial_observation = "init"
            return initial_observation, set(alphabet), mc, expr_manager
        else:
            raise ValueError("Unknown loader or missing parameters")

    def _load_monitor(self, use_exact: bool):
        if self.monitor is None:
            raise Exception("No monitor specified")
        if self.monitor.endswith(".nm") or self.monitor.endswith(".pm"):
            from tover.models.automata import load_dfa_stormpy, load_dfa_stormpy_exact

            return (
                load_dfa_stormpy_exact(self.monitor)
                if use_exact
                else load_dfa_stormpy(self.monitor)
            )
        elif self.monitor.endswith(".drn"):
            from tover.models.automata import load_dfa_drn

            return load_dfa_drn(self.monitor, use_exact)
        else:
            raise ValueError("Unknown monitor format")

    @override
    def run(self, timestamp: str, base_dir: str):
        if self.stop:
            return

        super().run(timestamp, base_dir)

        if self.threshold is not None:
            self.threshold = (
                sharpen(3, self.threshold) if self.use_exact else self.threshold
            )
        self.relative_error = (
            sharpen(5, self.relative_error) if self.use_exact else self.relative_error
        )

        start_time = time()
        logger.info(
            f"Running verification experiment: {self.name} ({self.variant}) {self.__dict__}"
        )

        verify_func_map = {
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
            "tn": true_negative,
        }
        if self.search not in verify_func_map:
            raise ValueError(f"Unknown search type: {self.search}")
        verify_func = verify_func_map[self.search]

        with OutputLogger():
            initial_observation, observations, mc, expr_manager = self._load_model()
            alphabet = list(observations)
            self.write_results(finished=False, mc=mc, alphabet=alphabet)

            monitor = self._load_monitor(self.use_exact)
            self.write_results(
                finished=False, monitor=monitor, mc=mc, alphabet=alphabet
            )

            try:
                sul = None
                if self.threshold is not None:
                    sul = FilteringSUL(
                        mc,
                        initial_observation,
                        alphabet,
                        self.spec,
                        self.threshold,
                        self.horizon,
                        self.use_risk,
                        use_dont_care=False,
                    )

                verify_start = time()
                cex, model, stats = verify_func(  # type: ignore[call-arg]
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
                        "export_benchmarks": True,
                        "conditional_method": self.conditional_method,
                        "hash": self.variant_hash,
                        **({"filtering": sul} if sul is not None else {}),
                    },
                )
                verify_time = time() - verify_start
                assert model is not None

                if cex is not None:
                    result_goal, trace, assignment = cex
                    if assignment:
                        logger.info(f"Assignment: {assignment}")
                        logger.info(f"PAYNT result: {assignment.analysis_result}")

                    path = (
                        f"{base_dir}/models/monitor_{self.name}_{self.variant_hash}"
                        f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                    )
                    verify_json = {
                        "goal_threshold": result_goal,
                        "trace": trace,
                        "time": verify_time,
                        "product_time": stats.product_time,
                        "paynt_time": stats.paynt_time,
                        "double_check_time": stats.double_check_time,
                        "iterations": stats.iterations,
                        "pomdp_states": len(model.pomdp.states),
                    }
                    if model:
                        with open(f"{path}.dot", "w") as f:
                            f.write(model.pomdp.to_dot())  # type: ignore
                        export_to_drn(model.pomdp, f"{path}.drn")
                else:
                    verify_json = {
                        "error": "No result",
                        "time": verify_time,
                        "product_time": stats.product_time,
                        "paynt_time": stats.paynt_time,
                        "iterations": stats.iterations,
                        "double_check_time": None,
                        "pomdp_states": len(model.pomdp.states),
                    }

            except Exception as e:
                logger.error(f"Error in verification: {traceback.format_exc()}")
                verify_json = {"error": str(e)}

            total_time = time() - start_time
            self.write_results(
                finished=True,
                results={"result": verify_json},
                mc=mc,
                alphabet=alphabet,
                monitor=monitor,
                total_time=total_time,
            )
            logger.info(
                f"Finished verification experiment: {self.name} ({self.variant})"
            )
