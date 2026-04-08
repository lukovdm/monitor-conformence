from aalpy import SUL, run_Lstar, Dfa, RandomWMethodEqOracle
from stormpy import (
    SparseDtmc,
    SparseModelComponents,
    SparseExactModelComponents,
    SparsePomdp,
    SparseExactPomdp,
    ExpressionManager,
    Rational,
    parse_properties,
    model_checking,
)
from stormpy.pomdp import (
    create_nondeterminstic_belief_tracker,
    NondeterministicBeliefTrackerDoubleSparse,
    NondeterministicBeliefTrackerExactSparse,
)

from tover.utils.logger import logger
from tover.core.oracles import SamplingEqOracle, ToVerEqOracle

# Maximum belief states tracked by the nondeterministic belief tracker.
_MAX_BELIEF_STATES = 10000


class FilteringSUL(SUL):
    """System Under Learning that filters observations by risk threshold.

    Wraps a Markov chain and uses a nondeterministic belief tracker to compute
    the current risk. A step returns True only if the risk is at or above
    the configured threshold.
    """

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

        observations = self._labels_to_observations(mc, observation_classes)

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
            components.observability_classes = observations
            self.pomdp = SparseExactPomdp(components)
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
            components.observability_classes = observations
            self.pomdp = SparsePomdp(components)

        self.tracker = create_nondeterminstic_belief_tracker(
            self.pomdp, _MAX_BELIEF_STATES, _MAX_BELIEF_STATES
        )

        prop = parse_properties(spec)
        result = model_checking(mc, prop[0])
        if use_risk:
            logger.debug(f"FilteringSUL risk function: {result.get_values()}")
            self.tracker.set_risk(result.get_values())
        else:
            logger.debug(f"FilteringSUL risk function: {result.get_truth_values()}")
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

    def step(self, observation: str) -> bool:
        """Advance the belief tracker by one observation. Returns True if risk >= threshold."""
        if self.tracker.size() == 0:
            if self.do_logging:
                logger.debug(
                    f"Risk collapsed to 0 after observing {observation} ({self.observation_length})",
                )
            return False

        if self.horizon is not None and self.observation_length >= self.horizon:
            if self.do_logging:
                logger.debug(
                    f"Horizon reached after {self.observation_length} observations",
                )
            return False

        if observation is not None:
            obs = self.observation_classes.index(observation)
            res = self.tracker.track(obs)
            self.observation_length += 1
            if not res:
                if self.do_logging:
                    logger.debug(
                        f"Observing {observation} resulted in belief collapse",
                    )
                return False

        risk = self.tracker.obtain_current_risk(max=False)
        if self.do_logging:
            logger.debug(
                f"Risk after observing {observation} ({self.observation_length}): "
                f"{risk} | {self.threshold} "
                f"[{[str(b) for b in self.tracker.obtain_beliefs()]}]",
            )

        self.last_risk = risk
        return risk >= self.threshold

    def steps(self, trace: list[str]) -> float:
        """Run a complete trace and return the final risk value."""
        self.set_logging(True)
        self.pre()
        for t in trace:
            self.step(t)
        self.post()
        self.set_logging(False)
        return self.last_risk

    @staticmethod
    def _labels_to_observations(mc: SparseDtmc, observation_classes: list[str]) -> list[int]:
        observations = []
        for state in mc.states:
            for label in mc.labeling.get_labels_of_state(state):
                if label in observation_classes:
                    observations.append(observation_classes.index(label))
                    break
        return observations


def run_tover(
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
    export_benchmarks: bool = False,
    conditional_method: str = "rejection",
    base_dir: str | None = None,
) -> tuple[tuple[Dfa, dict], dict]:
    """Run the ToVer L*-based monitor learning algorithm."""
    sul = FilteringSUL(
        mc,
        initial_observation,
        alphabet,
        spec,
        threshold,
        horizon if use_horizon_in_filtering else None,
        use_risk,
    )

    eq_oracle = ToVerEqOracle(
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
        export_benchmarks,
        conditional_method,
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
    """Run L* with a random W-method equivalence oracle (no PAYNT)."""
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
    """Run L* with the MC-based sampling equivalence oracle (no PAYNT)."""
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
