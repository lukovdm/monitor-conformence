from aalpy import SUL, StatePrefixEqOracle, run_Lstar
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


class FilteringSUL(SUL):
    def __init__(
        self,
        mc: SparseDtmc,
        initial_observation: str,
        observation_classes: list[str],
        spec: str,
        threshold: float,
    ):
        super().__init__()
        self.observation_classes = observation_classes
        self.initial_observation = initial_observation
        self.threshold = threshold

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

    def post(self):
        pass

    def step(self, observation: str):
        if self.tracker.size() == 0:
            return False

        if observation is not None:
            obs = self.observation_classes.index(observation)
            res = self.tracker.track(obs)
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
    eq_oracle = StatePrefixEqOracle(
        observation_classes,
        filtering_sul,
        walks_per_state=walks_per_state,
        walk_len=walk_len,
    )
    learned_monitor = run_Lstar(
        observation_classes,
        filtering_sul,
        eq_oracle,
        automaton_type="dfa",
        print_level=3,
    )

    return learned_monitor
