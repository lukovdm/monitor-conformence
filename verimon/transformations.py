from stormpy import (
    parse_properties,
    model_checking,
    SparseMdp,
    perform_sparse_bisimulation,
    BisimulationType,
)
from stormvogel.mapping import stormvogel_to_stormpy
from stormvogel.model import Model, new_mdp, State, Branch, Transition

from verimon.algs import reassign_ids


def product_unroll(mon: Model, horizon):
    ordering = _breath_first_ordering(mon)

    new_mon = new_mdp()
    new_mon.get_initial_state().labels = mon.get_initial_state().labels

    new_mon.actions = mon.actions.copy()

    states: dict[tuple[int, int], State] = {}
    for i in range(horizon):
        for id, s in mon.states.items():
            if s == mon.get_initial_state() and i == 0:
                new_s = new_mon.get_initial_state()
            else:
                new_s = new_mon.new_state([l for l in s.labels if l != "init"])
            states[(i, id)] = new_s
            new_s.add_label(f"i{i}")
            if i == horizon - 1:
                new_s.add_label("horizon")

    for i in range(horizon - 1):
        for s_id, trans_set in mon.transitions.items():
            new_s = states[(i, s_id)]
            new_trans = {}
            for a, b in trans_set.transition.items():
                new_branch = []
                for p, dest_s in b.branch:
                    if ordering[s_id] < ordering[dest_s.id]:
                        new_branch.append((p, states[(i, dest_s.id)]))
                    else:
                        new_branch.append((p, states[(i + 1, dest_s.id)]))
                new_trans[new_mon.actions[a.name]] = Branch(new_branch)
            new_s.set_transitions(Transition(new_trans))

    for s_id, trans_set in mon.transitions.items():
        new_s = states[(horizon - 1, s_id)]
        new_s.set_transitions([(a, new_s) for a in new_mon.actions.values()])

    return new_mon


def _breath_first_ordering(mon: Model) -> dict[int, int]:
    ordering: dict[int, int] = {0: 0}
    queue = [mon.get_initial_state()]
    while queue:
        s = queue.pop(0)
        for b in mon.transitions[s.id].transition.values():
            for p, s in b.branch:
                if p > 0 and s.id not in ordering:
                    ordering[s.id] = len(ordering)
                    queue.append(s)

    return ordering


def simulator_unroll(mon: Model, horizon):
    new_mon = new_mdp()
    new_mon.get_initial_state().labels = mon.get_initial_state().labels
    new_mon.get_initial_state().add_label("step=0")

    new_mon.actions = mon.actions.copy()
    states: dict[tuple[int, int], State] = {(0, 0): new_mon.get_initial_state()}
    queue = [(0, new_mon.get_initial_state(), mon.get_initial_state())]
    while queue:
        i, new_s, s = queue.pop(0)
        new_trans = {}
        for action, branch in mon.transitions[s.id].transition.items():
            new_branch = []
            for p, dest_s in branch.branch:
                if (i + 1, dest_s.id) in states:
                    new_dest_s = states[(i + 1, dest_s.id)]
                else:
                    new_dest_s = new_mon.new_state(
                        [
                            l
                            for l in dest_s.labels
                            if l != "init" and not l.startswith("step=")
                        ]
                    )
                    states[(i + 1, dest_s.id)] = new_dest_s
                    new_dest_s.add_label(f"step={i+1}")
                    if i + 1 == horizon - 1:
                        new_dest_s.add_label("horizon")
                    else:
                        queue.append((i + 1, new_dest_s, dest_s))

                new_branch.append((p, new_dest_s))
            new_trans[new_mon.actions[action.name]] = Branch(new_branch)
        new_s.set_transitions(Transition(new_trans))

    for s_id, trans_set in mon.transitions.items():
        if new_s := states.get((horizon - 1, s_id)):
            new_s.set_transitions([(a, new_s) for a in new_mon.actions.values()])

    return new_mon


def prune_monitor(mon: Model):
    mon_stormpy = stormvogel_to_stormpy(mon)
    prop = parse_properties('Pmax=? [F "accepting"]')
    result = model_checking(mon_stormpy, prop[0])
    to_delete = []
    for s_id, s in mon.states.items():
        if result.at(s_id) < 0.5:
            to_delete.append(s)

    for s in to_delete:
        mon.remove_state(s, False)

    reassign_ids(mon)


def bisim_minimise_monitor(mon: SparseMdp) -> SparseMdp:
    prop = parse_properties('Pmax=? [F "accepting"]')
    return perform_sparse_bisimulation(mon, prop, BisimulationType.STRONG)
