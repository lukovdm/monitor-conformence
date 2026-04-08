from stormpy import (
    SparseExactMdp,
    SparseMdp,
    StateLabeling,
    SparseExactModelComponents,
    SparseModelComponents,
)
from stormvogel.model import Model

from tover.utils.logger import logger


def reachable_states(model: Model) -> set[int]:
    reachable = {model.get_initial_state().id}
    check_queue = [model.get_initial_state()]
    while check_queue:
        state = check_queue.pop(0)
        for b in model.transitions[state.id].transition.values():
            for p, s in b.branch:
                if p == 0 or s.id in reachable:
                    continue
                check_queue.append(s)
                reachable.add(s.id)
    return reachable


def remove_unreachable_states(model: Model):
    reachable = reachable_states(model)
    states = [s for i, s in model.states.items() if i not in reachable]
    for state in states:
        model.remove_state(state)
    reassign_ids(model)
    logger.info(f"Removed {len(states)} unreachable states")


def complement_monitor(mon: SparseMdp | SparseExactMdp, accepting_label: str):
    """Flip the accepting label on all states of a Stormpy MDP monitor."""
    new_labeling = StateLabeling(len(mon.states))
    for label in mon.labeling.get_labels():
        new_labeling.add_label(label)

    for s in mon.states:
        labels = s.labels
        if accepting_label in labels:
            labels.remove(accepting_label)
        else:
            labels.add(accepting_label)
        for label in labels:
            new_labeling.add_label_to_state(label, s.id)

    if mon.is_exact:
        components = SparseExactModelComponents(mon.transition_matrix, new_labeling)
        try:
            components.choice_labeling = mon.choice_labeling
        except RuntimeError:
            pass
        try:
            components.state_valuations = mon.state_valuations
        except RuntimeError:
            pass
        return SparseExactMdp(components)
    else:
        components = SparseModelComponents(mon.transition_matrix, new_labeling)
        try:
            components.choice_labeling = mon.choice_labeling
        except RuntimeError:
            pass
        try:
            components.state_valuations = mon.state_valuations
        except RuntimeError:
            pass
        return SparseMdp(components)


def complement_model(model: Model, accepting_label: str):
    """Flip the accepting label on all states of a Stormvogel model, adding a bottom sink for missing actions."""
    bottom = None
    for i, state in list(model.states.items()):
        if accepting_label in state.labels:
            state.labels.remove(accepting_label)
        else:
            state.add_label(accepting_label)

        for a in model.actions.values():
            if i not in model.transitions or list(a.labels)[0] not in [
                list(a_s.labels)[0] for a_s in model.transitions[i].transition
            ]:
                logger.debug(
                    f"In complement model, state {i} is missing action {a}, adding it"
                )
                if bottom is None:
                    bottom = model.new_state("bottom")
                    for a_prime in model.actions.values():
                        bottom.add_transitions([(a_prime, bottom)])
                state.add_transitions([(a, bottom)])


def reassign_ids(mon: Model):
    """Normalize state IDs to a contiguous range starting from 0."""
    id_map = {}
    new_states = {}
    for new_id, (old_id, value) in enumerate(sorted(mon.states.items())):
        id_map[old_id] = new_id
        new_states[new_id] = value
        value.id = new_id
    mon.states = new_states
    mon.transitions = {
        new_id: mon.transitions[old_id] for old_id, new_id in id_map.items()
    }
