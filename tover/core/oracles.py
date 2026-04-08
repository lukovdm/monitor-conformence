import logging
import os
from datetime import datetime
from time import time

from aalpy import Dfa, Oracle, SUL
from stormpy import SparseDtmc, ExpressionManager, export_to_drn
from stormpy.simulator import create_simulator, SparseSimulator

from tover.models.automata import aalpy_dfa_to_stormpy
from tover.utils.logger import logger
from tover.core.verification import false_positive, false_negative


class SamplingEqOracle(Oracle):
    """Equivalence oracle that walks random paths through the MC and checks the hypothesis."""

    def __init__(
        self, alphabet: list[str], sul: SUL, mc: SparseDtmc, num_walks: int, walk_len: int
    ):
        super().__init__(alphabet, sul)
        self.mc = mc
        self.num_walks = num_walks
        self.walk_len = walk_len

    def find_cex(self, hypothesis: Dfa):
        simulator: SparseSimulator = create_simulator(self.mc)  # type: ignore
        for _ in range(self.num_walks):
            simulator.restart()
            self.reset_hyp_and_sul(hypothesis)
            trace = []

            for _ in range(self.walk_len):
                _, _, labels = simulator.step()
                label = next(l for l in labels if l in self.alphabet)
                trace.append(label)
                sul_out = self.sul.step(label)
                hyp_out = hypothesis.step(label)
                self.num_steps += 1
                if sul_out != hyp_out:
                    return trace

        return None


class ToVerEqOracle(Oracle):
    """Equivalence oracle that uses PAYNT synthesis to find false positives/negatives.

    Optionally combines a fast sampling pre-check (SamplingEqOracle) with the
    exact PAYNT-based check to reduce expensive synthesis calls.
    """

    def __init__(
        self,
        alphabet: list[str],
        sul,  # FilteringSUL
        mc: SparseDtmc,
        threshold: float,
        fp_slack: float,
        fn_slack: float,
        horizon: int,
        spec: str,
        good_label: str,
        relative_error: float,
        use_risk: bool,
        expression_manager: ExpressionManager,
        use_random_eq: bool = False,
        walks_per_state: int = 100,
        walk_len: int = 100,
        base_dir: str | None = None,
        export_benchmarks: bool = False,
        conditional_method: str = "rejection",
    ):
        super().__init__(alphabet, sul)
        self.filter_sul = sul
        self.alphabet = alphabet
        self.mc = mc
        self.threshold = threshold
        self.fp_slack = fp_slack
        self.fn_slack = fn_slack
        self.horizon = horizon
        self.spec = spec
        self.good_label = good_label
        self.relative_error = relative_error
        self.use_risk = use_risk
        self.expression_manager = expression_manager
        self.base_dir = base_dir
        self.export_benchmarks = export_benchmarks
        self.conditional_method = conditional_method

        self.stats = {
            "num_rounds": 0,
            "eq_used": 0,
            "fp_found": 0,
            "fn_found": 0,
            "monitors": [],
            "fp_bounds": [],
            "fn_bounds": [],
            "paynt_time": 0.0,
            "product_time": 0.0,
            "eq_time": 0.0,
        }

        self.sampling_oracle = (
            SamplingEqOracle(alphabet, sul, mc, walks_per_state, walk_len)
            if use_random_eq
            else None
        )

    def find_cex(self, hypothesis: Dfa):
        self.stats["num_rounds"] += 1
        logger.info(
            f"Finding counterexample for hypothesis with {len(hypothesis.states)} states"
        )

        if self.sampling_oracle is not None:
            cex = self._try_sampling_cex(hypothesis)
            if cex is not None:
                return cex

        logger.debug("Finding false negative probability")
        mon_cycl = aalpy_dfa_to_stormpy(hypothesis, self.mc.is_exact)
        self._maybe_export_monitor(mon_cycl, hypothesis)

        result, trace, _, _, stats = false_negative(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            self.threshold + self.fn_slack,
            self._verify_options(),
        )
        self._accumulate_paynt_stats(stats)

        if result is not None:
            self._assert_is_false_negative(hypothesis, trace)
            self.stats["fn_found"] += 1
            self.stats["fn_bounds"].append(result)
            self.stats["fp_bounds"].append(None)
            return trace

        logger.debug("Finding false positive probability")
        mon_cycl = aalpy_dfa_to_stormpy(hypothesis, self.mc.is_exact)
        result, trace, _, _, stats = false_positive(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            self.threshold - self.fp_slack,
            self._verify_options(),
        )
        self._accumulate_paynt_stats(stats)

        if result is not None:
            self._assert_is_false_positive(hypothesis, trace)
            self.stats["fp_found"] += 1
            self.stats["fp_bounds"].append(result)
            self.stats["fn_bounds"].append(None)
            return trace

        return None

    def _try_sampling_cex(self, hypothesis: Dfa):
        start_eq_time = time()
        logger.debug("Trying sampling oracle")

        # Check for false negative via sampling
        logger.debug(f"Finding fn using sampling oracle, threshold: {self.threshold + self.fn_slack}")
        self.filter_sul.threshold = self.threshold + self.fn_slack
        cex = self.sampling_oracle.find_cex(hypothesis)
        if cex is None or self._check_hyp_on_trace(hypothesis, cex):
            # No fn found, try false positive
            logger.debug(f"Finding fp using sampling oracle, threshold: {self.threshold - self.fp_slack}")
            self.filter_sul.threshold = self.threshold - self.fp_slack
            cex = self.sampling_oracle.find_cex(hypothesis)
            if cex is None or not self._check_hyp_on_trace(hypothesis, cex):
                logger.debug("No counter example found using sampling oracle")
                cex = None

        self.filter_sul.threshold = self.threshold
        self.num_steps = self.sampling_oracle.num_steps
        self.num_queries = self.sampling_oracle.num_queries
        self.stats["eq_time"] += time() - start_eq_time

        if cex is not None:
            self.stats["eq_used"] += 1
            self.stats["fn_bounds"].append(None)
            self.stats["fp_bounds"].append(None)
            self.stats["monitors"].append(None)
            logger.debug("Found counterexample using sampling oracle")
            return cex
        return None

    def _verify_options(self) -> dict:
        opts = {
            "good_spec": self.spec,
            "good_label": self.good_label,
            "relative_error": self.relative_error,
            "use_risk": self.use_risk,
            "filtering": self.filter_sul,
            "export_benchmarks": self.export_benchmarks,
            "conditional_method": self.conditional_method,
        }
        if self.base_dir is not None:
            opts["model_path"] = self.base_dir + "/debug-models"
        return opts

    def _maybe_export_monitor(self, mon_cycl, hypothesis: Dfa):
        if self.base_dir is None:
            return
        os.makedirs(self.base_dir + "/inter-mons", exist_ok=True)
        path = f"{self.base_dir}/inter-mons/mon-{datetime.now()}-{len(hypothesis.states)}.drn"
        self.stats["monitors"].append(path)
        export_to_drn(mon_cycl, path)

    def _accumulate_paynt_stats(self, stats: dict):
        self.stats["paynt_time"] += stats["paynt_time"]
        self.stats["product_time"] += stats["product_time"]

    def _assert_is_false_negative(self, hypothesis: Dfa, trace: list[str]):
        in_hyp = self._check_hyp_on_trace(hypothesis, trace)
        logger.log(
            logging.WARN if in_hyp else logging.INFO,
            f"Trace should not be in hyp: {in_hyp}",
        )
        in_sul = self._check_sul_on_trace(trace)
        logger.log(
            logging.INFO if in_sul else logging.WARN,
            f"Trace should be in SUL: {in_sul}",
        )
        if in_hyp or not in_sul:
            raise Exception("false negative found is not a false negative")

    def _assert_is_false_positive(self, hypothesis: Dfa, trace: list[str]):
        in_hyp = self._check_hyp_on_trace(hypothesis, trace)
        logger.log(
            logging.INFO if in_hyp else logging.WARN,
            f"Trace should be in hyp: {in_hyp}",
        )
        in_sul = self._check_sul_on_trace(trace)
        logger.log(
            logging.WARN if in_sul else logging.INFO,
            f"Trace should not be in SUL: {in_sul}",
        )
        if not in_hyp or in_sul:
            raise Exception("false positive found is not a false positive")

    def _check_sul_on_trace(self, trace: list[str]) -> bool:
        self.filter_sul.set_logging(True)
        self.filter_sul.pre()
        res = False
        for t in trace:
            res = self.filter_sul.step(t)
        self.filter_sul.post()
        self.filter_sul.set_logging(False)
        return res

    @staticmethod
    def _check_hyp_on_trace(hypothesis: Dfa, trace: list[str]) -> bool:
        return hypothesis.compute_output_seq(hypothesis.initial_state, trace)[-1]
