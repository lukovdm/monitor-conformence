import logging

from stormvogel.model import Model


def reachable_states(model: Model):
    reachable = {model.get_initial_state().id}
    check_queue = [model.get_initial_state()]
    while check_queue:
        state = check_queue.pop(0)
        for b in model.transitions[state.id].transition.values():
            for p, s in b.branch:
                if p == 0 or s.id in reachable:
                    continue
                else:
                    check_queue.append(s)
                    reachable.add(s.id)

    return reachable


def remove_unreachable_states(model: Model):
    reachable = reachable_states(model)
    states = list([s for i, s in model.states.items() if i not in reachable])
    for state in states:
        model.remove_state(state)

    reassign_ids(model)
    logging.info(f"Remove {len(states)} unreachable states")


def complement_model(model: Model, accepting_label: str):
    bottom = None
    for id, state in model.states.items():
        # Invert the good states
        if accepting_label in state.labels:
            state.labels.remove(accepting_label)
        else:
            state.add_label(accepting_label)

        # Add missing transitions to bottom state
        for a in model.actions.values():
            if id not in model.transitions or a not in model.transitions[id].transition:
                if bottom is None:
                    bottom = model.new_state("bottom")
                state.add_transitions([(a, bottom)])


def reassign_ids(mon: Model):
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
