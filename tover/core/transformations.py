from collections import deque
from typing import Any

from stormpy import (
    parse_properties,
    model_checking,
    SparseMdp,
    SparseExactMdp,
    ExactSparseMatrixBuilder,
    SparseMatrixBuilder,
    perform_sparse_bisimulation,
    BisimulationType,
    StateLabeling,
    ChoiceLabeling,
    SparseExactModelComponents,
    SparseModelComponents,
    Rational,
)
from stormvogel.mapping import stormvogel_to_stormpy
from stormvogel.model import Model, new_mdp, State, Branch, Transition

from tover.models.algorithms import reassign_ids

# Maximum number of belief states allowed by the tracker.
# Increasing this allows more precise belief tracking at higher memory cost.
MAX_BELIEF_STATES = 10000


def stormpy_unroll(mon: SparseMdp | SparseExactMdp, horizon: int) -> SparseMdp | SparseExactMdp:
    """Unroll a cyclic Stormpy MDP monitor to a finite-horizon acyclic MDP.

    Uses a BFS matrix-builder approach. Horizon states loop back on themselves.
    """
    states: dict[tuple[int, int], int] = {(0, mon.initial_states[0]): 0}
    state_labels: dict[int, set[str]] = {
        new_s: {"step=0", "init"} for new_s in states.values()
    }
    labels = {"step=0", "init", "horizon"}.union(mon.labeling.get_labels())
    for h in range(horizon + 1):
        labels.add(f"step={h}")

    action_labels_map: dict[int, set[str]] = {}
    action_labels = set(mon.choice_labeling.get_labels())

    queue = deque([(i, new_s, s) for (i, new_s), s in states.items()])
    horizon_queue = deque()

    builder = (
        ExactSparseMatrixBuilder(0, 0, 0, False, True)
        if mon.is_exact
        else SparseMatrixBuilder(0, 0, 0, False, True)
    )

    current_row = 0
    while queue:
        i, new_s, s = queue.popleft()
        builder.new_row_group(current_row)
        old_state = mon.states[s]
        state_labels[new_s].update(old_state.labels.difference(["init"]))

        for action in old_state.actions:
            action_labels_map[current_row] = action.labels

            new_row_dict: dict[int, Any] = {}
            for transition in action.transitions:
                dest_s = transition.column
                if (i + 1, dest_s) in states:
                    new_dest_s = states[(i + 1, dest_s)]
                else:
                    new_dest_s = len(states)
                    states[(i + 1, dest_s)] = new_dest_s
                    state_labels[new_dest_s] = {f"step={i + 1}"}
                    if i + 1 == horizon - 1:
                        state_labels[new_dest_s].add("horizon")
                        horizon_queue.append((new_dest_s, dest_s))
                    else:
                        queue.append((i + 1, new_dest_s, dest_s))

                new_row_dict[new_dest_s] = transition.value()

            for new_dest_s, value in sorted(new_row_dict.items()):
                builder.add_next_value(current_row, new_dest_s, value)

            current_row += 1

    for new_s, s in horizon_queue:
        builder.new_row_group(current_row)
        old_state = mon.states[s]
        state_labels[new_s].update(old_state.labels.difference(["init"]))

        for action in old_state.actions:
            action_labels_map[current_row] = action.labels
            builder.add_next_value(
                current_row, new_s, Rational(1.0) if mon.is_exact else 1.0
            )
            current_row += 1

    matrix = builder.build(overridden_column_count=len(states))

    labeling = StateLabeling(len(states))
    for label in labels:
        labeling.add_label(label)
    for state, lbls in state_labels.items():
        for label in lbls:
            labeling.add_label_to_state(label, state)

    choice_labeling = ChoiceLabeling(len(action_labels_map))
    for label in action_labels:
        choice_labeling.add_label(label)
    for action, lbls in action_labels_map.items():
        for label in lbls:
            choice_labeling.add_label_to_choice(label, action)

    if mon.is_exact:
        components = SparseExactModelComponents(matrix, labeling)
        components.choice_labeling = choice_labeling
        return SparseExactMdp(components)
    else:
        components = SparseModelComponents(matrix, labeling)
        components.choice_labeling = choice_labeling
        return SparseMdp(components)


def stormvogel_unroll(mon: Model, horizon: int) -> Model:
    """Unroll a cyclic Stormvogel MDP monitor to a finite-horizon acyclic MDP.

    Uses a BFS queue approach. Horizon states loop back on themselves.
    """
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
                        [l for l in dest_s.labels if l != "init" and not l.startswith("step=")]
                    )
                    states[(i + 1, dest_s.id)] = new_dest_s
                    new_dest_s.add_label(f"step={i + 1}")
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


def stormvogel_product_unroll(mon: Model, horizon: int) -> Model:
    """Unroll using a breath-first ordering to handle cyclic monitors."""
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


def prune_monitor(mon: Model):
    """Remove states with zero probability of reaching the accepting label."""
    mon_stormpy = stormvogel_to_stormpy(mon)
    prop = parse_properties('Pmax=? [F "accepting"]')
    result = model_checking(mon_stormpy, prop[0])
    to_delete = [s for s_id, s in mon.states.items() if result.at(s_id) < 0.5]
    for s in to_delete:
        mon.remove_state(s, False)
    reassign_ids(mon)


def bisim_minimise_monitor(mon: SparseMdp) -> SparseMdp:
    """Reduce a monitor via strong bisimulation."""
    prop = parse_properties('Pmax=? [F "accepting"]')
    return perform_sparse_bisimulation(mon, prop, BisimulationType.STRONG)
