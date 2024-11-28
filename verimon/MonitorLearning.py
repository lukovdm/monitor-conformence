import logging

from aalpy import SUL, run_Lstar, Dfa, RandomWMethodEqOracle, Oracle
from stormpy import (
    SparseDtmc,
    SparsePomdp,
    SparseModelComponents,
    parse_properties,
    model_checking,
    ExpressionManager,
)
from stormpy.pomdp import (
    create_nondeterminstic_belief_tracker,
    NondeterministicBeliefTrackerDoubleSparse,
)

from verimon.loaders import aalpy_dfa_to_stormvogel
from verimon.logger import logger, setup_logging
from verimon.verify import false_positive, false_negative


class FilteringSUL(SUL):
    def __init__(
        self,
        mc: SparseDtmc,
        initial_observation: str,
        observation_classes: list[str],
        spec: str,
        threshold: float,
        horizon: int,
    ):
        super().__init__()
        self.observation_classes = observation_classes
        self.initial_observation = initial_observation
        self.threshold = threshold
        self.spec = spec
        self.mc = mc
        self.horizon = horizon
        self.observation_length = 0
        self.logging_level = logging.DEBUG - 1

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
            mc, observation_classes
        )

        self.pomdp = SparsePomdp(components)

        self.tracker: NondeterministicBeliefTrackerDoubleSparse = (
            create_nondeterminstic_belief_tracker(self.pomdp, 10000, 10000)
        )

        prop = parse_properties(spec)
        result = model_checking(mc, prop[0])
        logger.debug(
            f"Filtering SUL is using the following risk function: {result.get_truth_values()}"
        )
        self.tracker.set_risk(result.get_truth_values())

    def set_logging(self, log: bool):
        if log:
            self.logging_level = logging.DEBUG
        else:
            self.logging_level = logging.DEBUG - 1

    def pre(self):
        self.tracker.reset(self.observation_classes.index(self.initial_observation))
        logger.log(self.logging_level, "reset")
        self.observation_length = 0

    def post(self):
        pass

    def step(self, observation: str):
        if self.tracker.size() == 0:
            logger.log(
                self.logging_level,
                f"Risk collapsed to 0 after observing {observation} ({self.observation_length})",
            )
            return False

        if self.observation_length > self.horizon:
            logger.log(
                self.logging_level,
                f"Risk collapsed to 0 after observing past the horizon ({self.observation_length})",
            )
            return False

        if observation is not None:
            obs = self.observation_classes.index(observation)
            res = self.tracker.track(obs)
            self.observation_length += 1
            if not res:
                logger.log(
                    self.logging_level,
                    f"Observing {observation} resulted in collapse, {res}",
                )
                return False

        risk = self.tracker.obtain_current_risk(max=False)
        logger.log(
            self.logging_level,
            f"Risk after observing {observation} ({self.observation_length}): {risk} | {self.threshold} [{[str(b) for b in self.tracker.obtain_beliefs()]}]",
        )
        return risk >= self.threshold

    @staticmethod
    def _labels_to_observations(mc: SparseDtmc, observation_classes: list[str]):
        observations = []
        for state in mc.states:
            for label in mc.labeling.get_labels_of_state(state):
                if label in observation_classes:
                    observations.append(observation_classes.index(label))
                    break

        return observations


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

    def find_cex(self, hypothesis: Dfa):
        hypothesis.visualize(
            path=f"models/monitor{len(hypothesis.states)}", file_type="dot"
        )
        logger.debug("Finding false negative probability")
        mon_cycl = aalpy_dfa_to_stormvogel(hypothesis)
        res = false_negative(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            self.threshold + self.fp_slack,
            {
                "good_spec": self.spec,
                "good_label": self.good_label,
                "relative_error": self.relative_error,
                "use_risk": self.use_risk,
            },
        )
        if res is not None:
            result, trace, _, _ = res
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
            return trace

        logger.debug("Finding false positive probability")
        mon_cycl = aalpy_dfa_to_stormvogel(hypothesis)
        res = false_positive(
            self.mc,
            mon_cycl,
            self.horizon,
            self.expression_manager,
            1 - (self.threshold - self.fn_slack),
            {
                "good_spec": self.spec,
                "good_label": self.good_label,
                "relative_error": self.relative_error,
                "use_risk": self.use_risk,
            },
        )
        if res is not None:
            result, trace, _, _ = res
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
    spec: str,
    good_label: str,
    threshold: float,
    horizon: int,
    relative_error: float,
    use_risk: bool,
    fp_slack: float,
    fn_slack: float,
    expression_manager: ExpressionManager,
):
    sul = FilteringSUL(
        mc,
        alphabet[0],
        alphabet,
        spec,
        threshold,
        horizon,
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
    )

    return run_Lstar(alphabet, sul, eq_oracle, automaton_type="dfa", print_level=2)
