import os
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Literal, cast, final

from aalpy import Dfa, Oracle
from stormpy import ExpressionManager, SparseDtmc, SparseMdp, export_to_drn
from stormpy.simulator import SparseSimulator

from tover.core.sul import FilteringSUL
from tover.core.synthesis import ConditionalMethod
from tover.core.verification import VerifyStats, false_negative, false_positive
from tover.lsharp.box import box_compare
from tover.models.automata import aalpy_dfa_to_stormpy
from tover.utils.logger import logger


@dataclass
class OracleStats:
    num_rounds: int = 0
    eq_used: int = 0
    fp_found: int = 0
    fn_found: int = 0
    monitors: list = field(default_factory=list)
    fp_bounds: list = field(default_factory=list)
    fn_bounds: list = field(default_factory=list)
    paynt_time: float = 0.0
    product_time: float = 0.0
    eq_time: float = 0.0

    def update_from(self, stats: VerifyStats) -> None:
        """Accumulate timing from a verification sub-call."""
        self.paynt_time += stats.paynt_time
        self.product_time += stats.product_time

    def __iadd__(self, other: "OracleStats") -> "OracleStats":
        self.num_rounds += other.num_rounds
        self.eq_used += other.eq_used
        self.fp_found += other.fp_found
        self.fn_found += other.fn_found
        self.monitors += other.monitors
        self.fp_bounds += other.fp_bounds
        self.fn_bounds += other.fn_bounds
        self.paynt_time += other.paynt_time
        self.product_time += other.product_time
        self.eq_time += other.eq_time
        return self


@final
class ToVerEqOracle(Oracle):
    """Equivalence oracle that uses PAYNT synthesis to find false positives/negatives.

    Optionally combines a fast sampling pre-check (SamplingEqOracle) with the
    exact PAYNT-based check to reduce expensive synthesis calls.
    """

    def __init__(
        self,
        alphabet: list[str],
        sul: FilteringSUL,
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
        random_eq_oracle: Oracle | None = None,
        base_dir: str | None = None,
        export_benchmarks: bool = False,
        conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
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

        self.stats = OracleStats()

        self.random_eq_oracle = random_eq_oracle

    def find_cex(self, hypothesis: Dfa[str]):
        self.stats.num_rounds += 1
        logger.info(
            f"Finding counterexample for hypothesis with {len(hypothesis.states)} states"
        )

        if self.random_eq_oracle is not None:
            cex = self._try_sampling_cex(hypothesis)
            if cex is not None:
                return cex

        logger.debug("Finding false negative probability")
        mon_cycl = aalpy_dfa_to_stormpy(hypothesis, self.mc.is_exact)
        # self._maybe_export_monitor(mon_cycl, hypothesis)

        cex, _, stats = false_negative(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            self.threshold + self.fn_slack,
            self._verify_options(),
        )
        self.stats.update_from(stats)

        if cex is not None:
            result, trace, _ = cex
            self._assert_is_false_negative(hypothesis, trace)
            self.stats.fn_found += 1
            self.stats.fn_bounds.append(result)
            self.stats.fp_bounds.append(None)
            return trace

        logger.debug("Finding false positive probability")
        mon_cycl = aalpy_dfa_to_stormpy(hypothesis, self.mc.is_exact)
        cex, _, stats = false_positive(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            self.threshold - self.fp_slack,
            self._verify_options(),
        )
        self.stats.update_from(stats)

        if cex is not None:
            result, trace, _ = cex
            self._assert_is_false_positive(hypothesis, trace)
            self.stats.fp_found += 1
            self.stats.fp_bounds.append(result)
            self.stats.fn_bounds.append(None)
            return trace

        return None

    def _try_sampling_cex(self, hypothesis: Dfa[str]):
        assert self.random_eq_oracle is not None

        start_eq_time = time()
        logger.debug("Trying sampling oracle")

        # Check for false negative via sampling
        logger.debug(
            f"Finding fn using sampling oracle, threshold: {self.threshold + self.fn_slack}"
        )
        self.filter_sul.threshold = self.threshold + self.fn_slack
        cex = self.random_eq_oracle.find_cex(hypothesis)
        if cex is None or self._check_hyp_on_trace(hypothesis, cex):
            # No fn found, try false positive
            logger.debug(
                f"Finding fp using sampling oracle, threshold: {self.threshold - self.fp_slack}"
            )
            self.filter_sul.threshold = self.threshold - self.fp_slack
            cex = self.random_eq_oracle.find_cex(hypothesis)
            if cex is None or not self._check_hyp_on_trace(hypothesis, cex):
                logger.debug("No counter example found using sampling oracle")
                cex = None

        self.filter_sul.threshold = self.threshold
        self.num_steps = self.random_eq_oracle.num_steps
        self.num_queries = self.random_eq_oracle.num_queries
        self.stats.eq_time += time() - start_eq_time

        if cex is not None:
            self.stats.eq_used += 1
            self.stats.fn_bounds.append(None)
            self.stats.fp_bounds.append(None)
            self.stats.monitors.append(None)
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

    def _maybe_export_monitor(self, mon_cycl: SparseMdp, hypothesis: Dfa[str]):
        if self.base_dir is None:
            return
        os.makedirs(self.base_dir + "/inter-mons", exist_ok=True)
        path = f"{self.base_dir}/inter-mons/mon-{datetime.now()}-{len(hypothesis.states)}.drn"
        self.stats.monitors.append(path)
        export_to_drn(mon_cycl, path)

    def _assert_is_false_negative(self, hypothesis: Dfa[str], trace: list[str]):
        in_hyp = self._check_hyp_on_trace(hypothesis, trace)
        logger.debug(f"Trace should not be in hyp: {in_hyp}")
        in_sul = self._check_sul_on_trace(trace)
        logger.debug(f"Trace should be in SUL: {in_sul}")
        if in_hyp or not in_sul:
            raise Exception("false negative found is not a false negative")

    def _assert_is_false_positive(self, hypothesis: Dfa[str], trace: list[str]):
        in_hyp = self._check_hyp_on_trace(hypothesis, trace)
        logger.debug(f"Trace should be in hyp: {in_hyp}")
        in_sul = self._check_sul_on_trace(trace)
        logger.debug(f"Trace should not be in SUL: {in_sul}")
        if not in_hyp or in_sul:
            raise Exception("false positive found is not a false positive")

    def _check_sul_on_trace(self, trace: list[str]) -> bool | Literal["unknown"]:
        # self.filter_sul.set_logging(True)
        self.filter_sul.pre()
        res = False
        for t in trace:
            res = self.filter_sul.step(t)
        self.filter_sul.post()
        # self.filter_sul.set_logging(False)
        return res

    @staticmethod
    def _check_hyp_on_trace(hypothesis: Dfa[str], trace: list[str]) -> bool:
        return cast(
            list[bool], hypothesis.compute_output_seq(hypothesis.initial_state, trace)
        )[-1]


@final
class SamplingEqOracle(Oracle):
    """Equivalence oracle that walks random paths through the MC and checks the hypothesis."""

    def __init__(
        self,
        alphabet: list[str],
        sul: FilteringSUL,
        mc: SparseDtmc,
        num_walks: int,
        walk_len: int,
    ):
        super().__init__(alphabet, sul)
        self.sul = sul
        self.mc = mc
        self.num_walks = num_walks
        self.walk_len = walk_len

    def find_cex(self, hypothesis: Dfa[str]) -> list[str] | None:
        simulator: SparseSimulator = cast(SparseSimulator, (self.mc))
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
                if box_compare(sul_out, hyp_out):
                    return trace

        return None
