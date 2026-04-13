from typing import Literal

from aalpy import SUL
from stormpy import (
    Rational,
    SparseDtmc,
    SparseExactModelComponents,
    SparseExactPomdp,
    SparseModelComponents,
    SparsePomdp,
    model_checking,
    parse_properties,
)
from stormpy.pomdp import (
    create_nondeterminstic_belief_tracker,
)

from tover.utils.logger import logger

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
        threshold: float | Rational | tuple[float, float] | tuple[Rational, Rational],
        horizon: int | None,
        use_risk: bool,
        use_dont_care: bool,
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
        self.use_dont_care = use_dont_care

        if self.use_dont_care != (type(self.threshold) is tuple):
            logger.warning(
                "When using don't cares it is reccomended to use an interval threshold"
            )

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
            risk_values = result.get_values()
            logger.debug(
                f"FilteringSUL risk function: {max(risk_values)} max, {min(risk_values)} min, {float(sum(risk_values) / len(risk_values)):.2f} avg, {risk_values[-3:]} tail",
            )
            self.tracker.set_risk(risk_values)
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

    def step(self, observation: str) -> bool | Literal["unknown"]:
        """Advance the belief tracker by one observation. Returns True if risk >= threshold."""
        if self.tracker.size() == 0:
            if self.do_logging:
                logger.debug(
                    f"No possible beliefs after observing {observation} ({self.observation_length})",
                )
            return self._box()

        if self.horizon is not None and self.observation_length >= self.horizon:
            if self.do_logging:
                logger.debug(
                    f"Horizon reached after {self.observation_length} observations",
                )
            return self._box()

        if observation is not None:
            obs = self.observation_classes.index(observation)
            res = self.tracker.track(obs)
            self.observation_length += 1
            if not res:
                if self.do_logging:
                    logger.debug(
                        f"Observing {observation} resulted in belief collapse",
                    )
                return self._box()

        risk = self.tracker.obtain_current_risk(max=False)
        if self.do_logging:
            logger.debug(
                f"Risk after observing {observation} ({self.observation_length}): "
                f"{risk} | {self.threshold} "
                f"[{[str(b) for b in self.tracker.obtain_beliefs()]}]",
            )

        self.last_risk = risk
        if type(self.threshold) is tuple:
            lower, upper = self.threshold
            if risk < lower:
                return False
            elif risk >= upper:
                return True
            else:
                return self._box()
        else:
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

    def _box(self) -> Literal[False] | Literal["unknown"]:
        if self.use_dont_care:
            return "unknown"

        return False

    @staticmethod
    def _labels_to_observations(
        mc: SparseDtmc, observation_classes: list[str]
    ) -> list[int]:
        observations = []
        for state in mc.states:
            for label in mc.labeling.get_labels_of_state(state):
                if label in observation_classes:
                    observations.append(observation_classes.index(label))
                    break
            else:
                raise ValueError(
                    f"State {state} has no label in the observation classes {observation_classes}"
                )
        return observations
