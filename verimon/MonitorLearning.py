from datetime import datetime
import logging
import os
from time import time

from aalpy import SUL, run_Lstar, Dfa, RandomWMethodEqOracle, Oracle
from stormpy import (
    SparseDtmc,
    SparsePomdp,
    SparseModelComponents,
    SparseExactPomdp,
    SparseExactModelComponents,
    export_to_drn,
    parse_properties,
    model_checking,
    ExpressionManager,
    Rational,
)
from stormpy.pomdp import (
    create_nondeterminstic_belief_tracker,
    NondeterministicBeliefTrackerDoubleSparse,
    NondeterministicBeliefTrackerExactSparse,
)
from stormpy.simulator import create_simulator, SparseSimulator

from verimon.loaders import aalpy_dfa_to_stormpy
from verimon.logger import logger
from verimon.verify import false_positive, false_negative


class FilteringSUL(SUL):
    def __init__(
        self,
        mc: SparseDtmc,
        initial_observation: str,
        observation_classes: list[str],
        spec: str,
        threshold: float | Rational,
        horizon: int | None,
        use_risk: bool,
    ):
        super().__init__()
        self.observation_classes = observation_classes
        self.initial_observation = initial_observation
        self.threshold = threshold
        self.spec = spec
        self.mc = mc
        self.horizon = horizon
        self.observation_length = 0
        self.do_logging = False
        self.last_risk = 0

        if mc.is_exact:
            components = SparseExactModelComponents(mc.transition_matrix, mc.labeling)
            try:
                components.choice_labeling = mc.choice_labeling
            except RuntimeError:
                pass
            try:
                components.state_valuations = mc.state_valuations
            except RuntimeError:
                pass
            components.observability_classes = FilteringSUL._labels_to_observations(
                mc,
                observation_classes,
            )

            self.pomdp = SparseExactPomdp(components)

            self.tracker: NondeterministicBeliefTrackerExactSparse = (
                create_nondeterminstic_belief_tracker(self.pomdp, 10000, 10000)
            )
        else:
            components = SparseModelComponents(mc.transition_matrix, mc.labeling)
            try:
                components.choice_labeling = mc.choice_labeling
            except RuntimeError:
                pass
            try:
                components.state_valuations = mc.state_valuations
            except RuntimeError:
                pass
            components.observability_classes = FilteringSUL._labels_to_observations(
                mc,
                observation_classes,
            )

            self.pomdp = SparsePomdp(components)

            self.tracker: NondeterministicBeliefTrackerDoubleSparse = (
                create_nondeterminstic_belief_tracker(self.pomdp, 10000, 10000)
            )

        prop = parse_properties(spec)
        result = model_checking(mc, prop[0])
        if use_risk:
            logger.debug(
                f"Filtering SUL is using the following risk function: {result.get_values()}"
            )
            self.tracker.set_risk(result.get_values())
        else:
            logger.debug(
                f"Filtering SUL is using the following risk function: {result.get_truth_values()}"
            )
            self.tracker.set_risk(result.get_truth_values())

    def set_logging(self, log: bool):
        self.do_logging = log

    def pre(self):
        self.tracker.reset(self.observation_classes.index(self.initial_observation))
        self.last_risk = self.tracker.obtain_current_risk(max=False)
        if self.do_logging:
            logger.debug(f"reset tracker, {self.last_risk}")
        self.observation_length = 0

    def post(self):
        pass

    def step(self, observation: str):
        if self.tracker.size() == 0:
            if self.do_logging:
                logger.debug(
                    f"Risk collapsed to 0 after observing {observation} ({self.observation_length})",
                )
            return False

        if self.horizon is not None and self.observation_length >= self.horizon:
            if self.do_logging:
                logger.debug(
                    f"Risk collapsed to 0 after observing past the horizon ({self.observation_length})",
                )
            return False

        if observation is not None:
            obs = self.observation_classes.index(observation)
            res = self.tracker.track(obs)
            self.observation_length += 1
            if not res:
                if self.do_logging:
                    logger.debug(
                        f"Observing {observation} resulted in collapse, {res}",
                    )
                return False

        risk = self.tracker.obtain_current_risk(max=False)
        if self.do_logging:
            logger.debug(
                f"Risk after observing {observation} ({self.observation_length}): {risk} | {self.threshold} [{[str(b) for b in self.tracker.obtain_beliefs()]}]",
            )

        self.last_risk = risk
        return risk >= self.threshold

    def steps(self, trace):
        self.set_logging(True)
        self.pre()
        res = False
        for t in trace:
            self.step(t)
        self.post()
        self.set_logging(False)
        return self.last_risk

    @staticmethod
    def _labels_to_observations(mc: SparseDtmc, observation_classes: list[str]):
        observations = []
        for state in mc.states:
            for label in mc.labeling.get_labels_of_state(state):
                if label in observation_classes:
                    observations.append(observation_classes.index(label))
                    break

        return observations


class SamplingEqOracle(Oracle):
    def __init__(
        self, alphabet, sul: SUL, mc: SparseDtmc, num_walks: int, walk_len: int
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


class VerimonEqOracle(Oracle):
    def __init__(
        self,
        alphabet,
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
        use_random_eq: bool = False,
        walks_per_state: int = 100,
        walk_len: int = 100,
        base_dir: str | None = None,
    ):
        """

        :param alphabet: The observations in the MC
        :param sul: the filtering SUL
        :param mc: the MC we are learning
        :param gb: the specification model
        :param threshold: The threshold probability of a trace to be included in the monitor
        :param slack: How far over the false negative and false positive rates can be during eq checking
        :param horizon: the max steps we are taking
        :param spec: the specification of good states
        :param good_label: the label of good states
        :param relative_error: the relative error for Paynt
        """
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

        if use_random_eq:
            self.eq_orcale = SamplingEqOracle(
                alphabet, sul, mc, walks_per_state, walk_len
            )
        else:
            self.eq_orcale = None

    def find_cex(self, hypothesis: Dfa):
        self.stats["num_rounds"] += 1
        logger.info(
            f"Finding counterexample for hypothesis with {len(hypothesis.states)} states"
        )
        # hypothesis.visualize(
        #     path=f"{self.base_dir}/models/monitor{len(hypothesis.states)}",
        #     file_type="dot",
        # )

        if self.eq_orcale is not None:
            start_eq_time = time()
            logger.debug("Trying eq oracle")
            logger.debug(
                f"Finding fn using eq oracle, threshold: {self.threshold + self.fn_slack}"
            )
            self.filter_sul.threshold = self.threshold + self.fn_slack
            cex = self.eq_orcale.find_cex(hypothesis)
            if (
                cex is None or self._check_hyp_on_trace(hypothesis, cex)
            ):  # We found a counter example but it is not a false negative, thus we ignore it
                logger.debug(
                    f"Finding fp using eq oracle, threshold: {self.threshold - self.fp_slack}"
                )
                self.filter_sul.threshold = self.threshold - self.fp_slack
                cex = self.eq_orcale.find_cex(hypothesis)
                if (
                    cex is None or not self._check_hyp_on_trace(hypothesis, cex)
                ):  # We found a counter example but it is not a false positive, thus we ignore it
                    logger.debug("No counter example found using eq oracle")
                    cex = None

            self.filter_sul.threshold = self.threshold
            self.num_steps = self.eq_orcale.num_steps
            self.num_queries = self.eq_orcale.num_queries
            self.stats["eq_time"] += time() - start_eq_time
            if cex is not None:
                self.stats["eq_used"] += 1
                self.stats["fn_bounds"].append(None)
                self.stats["fp_bounds"].append(None)
                self.stats["monitors"].append(None)
                logger.debug("Found counterexample using eq oracle")
                return cex

        logger.debug("Finding false negative probability")
        mon_cycl = aalpy_dfa_to_stormpy(hypothesis, self.mc.is_exact)

        if self.base_dir is not None:
            os.makedirs(self.base_dir + "/inter-mons", exist_ok=True)
            path = f"{self.base_dir}/inter-mons/mon-{datetime.now()}-{len(hypothesis.states)}.drn"
            self.stats["monitors"].append(path)
            export_to_drn(
                mon_cycl,
                path,
            )

        result, trace, _, _, stats = false_negative(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            self.threshold + self.fn_slack,
            {
                "good_spec": self.spec,
                "good_label": self.good_label,
                "relative_error": self.relative_error,
                "use_risk": self.use_risk,
                "filtering": self.filter_sul,
            }
            | (
                {"model_path": self.base_dir + "/debug-models"}
                if self.base_dir is not None
                else {}
            ),
        )

        self.stats["paynt_time"] += stats["paynt_time"]
        self.stats["product_time"] += stats["product_time"]

        if result is not None:
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
            {
                "good_spec": self.spec,
                "good_label": self.good_label,
                "relative_error": self.relative_error,
                "use_risk": self.use_risk,
                "filtering": self.filter_sul,
            }
            | (
                {"model_path": self.base_dir + "/debug-models"}
                if self.base_dir is not None
                else {}
            ),
        )

        self.stats["paynt_time"] += stats["paynt_time"]
        self.stats["product_time"] += stats["product_time"]

        if result is not None:
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

            self.stats["fp_found"] += 1
            self.stats["fp_bounds"].append(result)
            self.stats["fn_bounds"].append(None)
            return trace

        return None

    def _check_sul_on_trace(self, trace):
        self.filter_sul.set_logging(True)
        self.filter_sul.pre()
        res = False
        for t in trace:
            res = self.filter_sul.step(t)
        self.filter_sul.post()
        self.filter_sul.set_logging(False)
        return res

    @staticmethod
    def _check_hyp_on_trace(hypothesis: Dfa, trace):
        return hypothesis.compute_output_seq(hypothesis.initial_state, trace)[-1]


def run_verimon(
    mc: SparseDtmc,
    alphabet: list[str],
    initial_observation: str,
    spec: str,
    good_label: str,
    threshold: float,
    horizon: int,
    relative_error: float,
    use_risk: bool,
    fp_slack: float,
    fn_slack: float,
    expression_manager: ExpressionManager,
    use_random_eq: bool,
    walks_per_state: int,
    walk_len: int,
    use_horizon_in_filtering: bool,
    base_dir: str | None = None,
) -> tuple[tuple[Dfa, dict], dict]:
    sul = FilteringSUL(
        mc,
        initial_observation,
        alphabet,
        spec,
        threshold,
        horizon if use_horizon_in_filtering else None,
        use_risk,
    )

    eq_oracle = VerimonEqOracle(
        alphabet,
        sul,
        mc,
        threshold,
        fp_slack,
        fn_slack,
        horizon,
        spec,
        good_label,
        relative_error,
        use_risk,
        expression_manager,
        use_random_eq,
        walks_per_state,
        walk_len,
        base_dir,
    )

    return (
        run_Lstar(
            alphabet,
            sul,
            eq_oracle,
            automaton_type="dfa",
            print_level=2,
            return_data=True,
        ),  # type: ignore
        eq_oracle.stats,
    )


def run_trad_learning(
    mc: SparseDtmc,
    alphabet: list[str],
    initial_observation: str,
    spec: str,
    threshold: float,
    horizon: int,
    walks_per_state: int,
    walk_len: int,
    use_risk: bool,
    use_horizon_in_filtering: bool,
) -> tuple[Dfa, dict]:
    sul = FilteringSUL(
        mc,
        initial_observation,
        alphabet,
        spec,
        threshold,
        horizon if use_horizon_in_filtering else None,
        use_risk,
    )
    eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state, walk_len)
    return run_Lstar(
        alphabet, sul, eq_oracle, automaton_type="dfa", print_level=2, return_data=True
    )  # type: ignore


def run_sampling_learning(
    mc: SparseDtmc,
    alphabet: list[str],
    initial_observation: str,
    spec: str,
    threshold: float,
    horizon: int,
    num_walks: int,
    walk_len: int,
    use_risk: bool,
    use_horizon_in_filtering: bool,
) -> tuple[Dfa, dict]:
    sul = FilteringSUL(
        mc,
        initial_observation,
        alphabet,
        spec,
        threshold,
        horizon if use_horizon_in_filtering else None,
        use_risk,
    )
    eq_oracle = SamplingEqOracle(alphabet, sul, mc, num_walks, walk_len)
    return run_Lstar(
        alphabet, sul, eq_oracle, automaton_type="dfa", print_level=2, return_data=True
    )  # type: ignore
