from json import loads
from paynt.family.family import Family
from stormpy import SparsePomdp, SparseMdp, SparseModelComponents
import re


def get_pos(json_val):
    return loads(str(json_val))["pos"]


def stormpy_pomdp_to_mdp(pomdp: SparsePomdp) -> SparseMdp:
    components = SparseModelComponents(pomdp.transition_matrix, pomdp.labeling)
    components.choice_labeling = pomdp.choice_labeling
    components.state_valuations = pomdp.state_valuations

    return SparseMdp(components)


def hole_to_observations(assignment: Family) -> dict[int, str]:
    action_map = {}
    for i in range(assignment.num_holes):
        obs = int(assignment.hole_name(i)[2:-3])
        act = assignment.hole_to_option_labels[i][assignment.hole_options(i)[0]]
        action_map[obs] = act
    return action_map
