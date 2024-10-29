import logging

from aalpy import SUL, run_Lstar, Dfa, RandomWMethodEqOracle, Oracle
from stormpy import (
    SparseDtmc,
    SparsePomdp,
    SparseModelComponents,
    parse_properties,
    model_checking,
)
from stormpy.pomdp import (
    create_nondeterminstic_belief_tracker,
    NondeterministicBeliefTrackerDoubleSparse,
)
from stormvogel.model import new_mdp, Model

from verimon.logger import logger
from verimon.verify import false_positive, false_negative


class VerimonEqOracle(Oracle):

    def __init__(
        self,
        alphabet,
        sul: SUL,
        mc: Model,
        gb: Model,
        threshold: float,
        slack: float,
        horizon: int,
        spec: str,
        good_label: str,
        relative_error: float,
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
        self.alphabet = alphabet
        self.mc = mc
        self.gb = gb
        self.threshold = threshold
        self.slack = slack
        self.horizon = horizon
        self.spec = spec
        self.good_label = good_label
        self.relative_error = relative_error

    def find_cex(self, hypothesis: Dfa):
        hypothesis.visualize(path=f"model{len(hypothesis.states)}")
        logger.debug("Finding false negative probability")
        mon_cycl = aalpy_dfa_to_stormvogel(hypothesis)
        res = false_negative(
            self.mc,
            mon_cycl,
            self.gb,
            self.horizon,
            self.threshold + self.slack,
            {
                "good_spec": self.spec,
                "good_label": self.good_label,
                "relative_error": self.relative_error,
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
            return trace

        logger.debug("Finding false positive probability")
        mon_cycl = aalpy_dfa_to_stormvogel(hypothesis)
        res = false_positive(
            self.mc,
            mon_cycl,
            self.gb,
            self.horizon,
            1 - (self.threshold - self.slack),
            {
                "good_spec": self.spec,
                "good_label": self.good_label,
                "relative_error": self.relative_error,
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
            return trace

        return None

    def _check_sul_on_trace(self, trace):
        self.sul.pre()
        res = False
        for t in trace:
            res = self.sul.step(t)
        self.sul.post()
        return res

    @staticmethod
    def _check_hyp_on_trace(hypothesis: Dfa, trace):
        return hypothesis.compute_output_seq(hypothesis.initial_state, trace)[-1]


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

        components = SparseModelComponents(mc.transition_matrix, mc.labeling)
        components.choice_labeling = mc.choice_labeling
        components.state_valuations = mc.state_valuations
        components.observability_classes = FilteringSUL._labels_to_observations(
            mc, observation_classes
        )

        self.pomdp = SparsePomdp(components)

        self.tracker: NondeterministicBeliefTrackerDoubleSparse = (
            create_nondeterminstic_belief_tracker(self.pomdp, 10000, 10000)
        )

        prop = parse_properties(spec)
        result = model_checking(mc, prop[0])
        self.tracker.set_risk(result.get_truth_values())

    def pre(self):
        self.tracker.reset(self.observation_classes.index(self.initial_observation))
        self.observation_length = 0

    def post(self):
        pass

    def step(self, observation: str):
        if self.tracker.size() == 0:
            return False

        if self.observation_length > self.horizon:
            return False

        if observation is not None:
            obs = self.observation_classes.index(observation)
            res = self.tracker.track(obs)
            self.observation_length += 1
            if not res:
                return False

        return self.tracker.obtain_current_risk(max=False) >= self.threshold

    @staticmethod
    def _labels_to_observations(mc: SparseDtmc, observation_classes: list[str]):
        observations = []
        for state in mc.states:
            for label in mc.labeling.get_labels_of_state(state):
                if label in observation_classes:
                    observations.append(observation_classes.index(label))
                    break

        return observations


def learn_monitor(
    mc: SparseDtmc,
    initial_observation: str,
    observation_classes: list[str],
    spec: str,
    threshold: float,
    walks_per_state=100,
    walk_len=100,
):
    filtering_sul = FilteringSUL(
        mc, initial_observation, observation_classes, spec, threshold
    )
    # eq_oracle = StatePrefixEqOracle(
    #     observation_classes,
    #     filtering_sul,
    #     walks_per_state=walks_per_state,
    #     walk_len=walk_len,
    # )
    eq_oracle = RandomWMethodEqOracle(
        observation_classes, filtering_sul, walks_per_state, walk_len
    )
    learned_monitor = run_Lstar(
        observation_classes,
        filtering_sul,
        eq_oracle,
        automaton_type="dfa",
        print_level=2,
    )

    return learned_monitor


def aalpy_dfa_to_stormvogel(dfa_a: Dfa):
    dfa_sv = new_mdp()

    action_mapping = {}
    for act in dfa_a.get_input_alphabet():
        action_mapping[act] = dfa_sv.new_action(act, frozenset({act}))

    state_mapping = {dfa_a.initial_state: dfa_sv.get_initial_state()}
    dfa_sv.get_initial_state().add_label(dfa_a.initial_state.state_id)
    if dfa_a.initial_state.is_accepting:
        dfa_sv.get_initial_state().add_label("accepting")

    for s in sorted(dfa_a.states, key=lambda q: q.state_id):
        if s == dfa_a.initial_state:
            continue

        s_sv = dfa_sv.new_state(s.state_id)
        state_mapping[s] = s_sv
        if s.is_accepting:
            s_sv.add_label("accepting")

    for s in sorted(dfa_a.states, key=lambda q: q.state_id):
        for act, dest_s in s.transitions.items():
            state_mapping[s].add_transitions(
                [(action_mapping[act], state_mapping[dest_s])]
            )

    return dfa_sv
