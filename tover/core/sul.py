from time import time
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


def storm_one(is_exact: bool):
    """One, in whichever arithmetic the model uses."""
    return Rational(1) if is_exact else 1.0


def storm_zero(is_exact: bool):
    """Zero, in whichever arithmetic the model uses."""
    return Rational(0) if is_exact else 0.0


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
        self.time_taken = 0.0

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


        start_time = time()
        prop = parse_properties(spec)
        result = model_checking(mc, prop[0])
        self.time_taken = time() - start_time
        if use_risk:
            risk_values = result.get_values()
            logger.debug(
                f"FilteringSUL risk function: {max(risk_values)} max, {min(risk_values)} min, {float(sum(risk_values) / len(risk_values)):.2f} avg, {risk_values[-3:]} tail",
            )
            self.tracker.set_risk(risk_values)
        else:
            # get_truth_values returns a BitVector, and iterating a BitVector yields the
            # *indices of the set bits* rather than one boolean per state. Passing it straight
            # through produced a risk vector of length popcount whose entries were state
            # indices -- a length mismatch on most models, and silently nonsense risks on a
            # model where every bit happens to be set. Expand it explicitly.
            truth_values = result.get_truth_values()
            one, zero = storm_one(mc.is_exact), storm_zero(mc.is_exact)
            risk_values = [zero] * mc.nr_states
            for state in truth_values:
                risk_values[state] = one
            logger.debug(f"FilteringSUL risk function: {sum(1 for v in risk_values if v)} of {mc.nr_states} states")
            self.tracker.set_risk(risk_values)
        # Kept so that seeding can run the belief search over exactly the same risk function
        # the tracker uses, rather than recomputing it and risking a divergence.
        self.risk_values = risk_values

    def set_logging(self, log: bool):
        self.do_logging = log

    def pre(self):
        start_time = time()
        self.tracker.reset(self.observation_classes.index(self.initial_observation))
        self.last_risk = self.tracker.obtain_current_risk(max=False)
        if self.do_logging:
            logger.debug(f"reset tracker, {self.last_risk}")
        self.observation_length = 0
        self.time_taken += time() - start_time

    def post(self):
        pass

    def step(self, observation: str) -> bool | Literal["unknown"]:
        """Advance the belief tracker by one observation. Returns True if risk >= threshold."""
        start_time = time()
        if self.tracker.size() == 0:
            if self.do_logging:
                logger.debug(
                    f"No possible beliefs after observing {observation} ({self.observation_length})",
                )
            self.time_taken += time() - start_time
            return self._box()

        if self.horizon is not None and self.observation_length >= self.horizon:
            if self.do_logging:
                logger.debug(
                    f"Horizon reached after {self.observation_length} observations",
                )
            self.time_taken += time() - start_time
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
                self.time_taken += time() - start_time
                return self._box()

        risk = self.tracker.obtain_current_risk(max=False)
        if self.do_logging:
            logger.debug(
                f"Risk after observing {observation} ({self.observation_length}): "
                f"{risk} | {self.threshold} "
                f"[{[str(b) for b in self.tracker.obtain_beliefs()]}]",
            )

        self.time_taken += time() - start_time
        self.last_risk = risk
        return self.output_for_risk(risk)

    def output_for_risk(self, risk) -> bool | Literal["unknown"]:
        """Map a raw risk value to the output this SUL would report.

        Factored out of :meth:`step` so that seeding (which obtains risks from the belief
        search rather than from the tracker) cannot drift from the SUL's own rule.
        """
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
