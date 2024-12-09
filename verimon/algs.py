from stormvogel.model import Model

from verimon.logger import logger


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
    logger.info(f"Removed {len(states)} unreachable states")


def complement_model(model: Model, accepting_label: str):
    bottom = None
    for i, state in list(model.states.items()):
        # Invert the good states
        if accepting_label in state.labels:
            state.labels.remove(accepting_label)
        else:
            state.add_label(accepting_label)

        # Add missing transitions to bottom state
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
