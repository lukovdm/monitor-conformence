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
from stormvogel.model import new_mdp


# class VerimonEqOracle(Oracle):
#
#     def __init__(self, alphabet, sul: SUL, threshold: int, mc: Model):
#         super().__init__(alphabet, sul)
#
#     def find_cex(self, hypothesis):
#         mon_cycl = aalpy_dfa_to_stormvogel(hypothesis)
#         mon = simulator_unroll(mon_cycl, 10)
#         prune_monitor(mon)
#         return None


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
        self.spec = spec
        self.mc = mc

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
