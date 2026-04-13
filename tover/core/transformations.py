from collections import deque
from typing import Any

from aalpy import Dfa, DfaState
from stormpy import (
    BisimulationType,
    ChoiceLabeling,
    ExactSparseMatrixBuilder,
    Rational,
    SparseDtmc,
    SparseExactMdp,
    SparseExactModelComponents,
    SparseMatrixBuilder,
    SparseMdp,
    SparseModelComponents,
    StateLabeling,
    parse_properties,
    perform_sparse_bisimulation,
)

# Maximum number of belief states allowed by the tracker.
# Increasing this allows more precise belief tracking at higher memory cost.
MAX_BELIEF_STATES = 10000


def stormpy_unroll(
    mon: SparseMdp | SparseExactMdp, horizon: int
) -> SparseMdp | SparseExactMdp:
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


def bisim_minimise_monitor(mon: SparseMdp) -> SparseMdp:
    """Reduce a monitor via strong bisimulation."""
    prop = parse_properties('Pmax=? [F "accepting"]')
    return perform_sparse_bisimulation(mon, prop, BisimulationType.STRONG)


def nfa_dict_to_dfa(
    nfa_transitions: dict[int, list[tuple[str, int]]],
    initial_state: int,
    accepting_states: set[int],
    alphabet: set[str],
) -> Dfa[str]:
    """Convert an NFA (given as a transition dict) to a DFA via powerset construction."""
    # Map from frozenset of NFA states → DfaState
    dfa_states: dict[frozenset[int], DfaState[str]] = {}
    state_id_counter = 0

    def get_or_create(nfa_state_set: frozenset[int]) -> DfaState[str]:
        nonlocal state_id_counter
        if nfa_state_set not in dfa_states:
            is_accepting = any(s in accepting_states for s in nfa_state_set)
            dfa_state = DfaState(str(set(nfa_state_set)), is_accepting)
            dfa_states[nfa_state_set] = dfa_state
            state_id_counter += 1
        return dfa_states[nfa_state_set]

    initial_set = frozenset({initial_state})
    initial_dfa_state = get_or_create(initial_set)

    queue = deque([initial_set])
    visited: set[frozenset[int]] = {initial_set}

    while queue:
        current_set = queue.popleft()
        current_dfa_state = dfa_states[current_set]
        for sym in alphabet:
            next_set = frozenset(
                dest
                for nfa_s in current_set
                for trans_sym, dest in nfa_transitions.get(nfa_s, [])
                if trans_sym == sym
            )
            if not next_set:
                # Dead state (empty set)
                next_set = frozenset()
            next_dfa_state = get_or_create(next_set)
            current_dfa_state.transitions[sym] = next_dfa_state
            if next_set not in visited:
                visited.add(next_set)
                queue.append(next_set)

    all_states = list(dfa_states.values())
    dfa = Dfa(initial_dfa_state, all_states)
    dfa.compute_prefixes()
    return dfa


def language_of_hmm(hmm: SparseDtmc, observation_classes: list[str]) -> Dfa[str]:
    """Construct a DFA accepting all traces possible in the hmm"""
    transitions: dict[int, list[tuple[str, int]]] = {}

    for s in hmm.states:
        transitions[s.id] = []

    rejecting_state = -1
    transitions[-1] = [(o, rejecting_state) for o in observation_classes]

    for s in hmm.states:
        observations_missed = set(observation_classes)
        for entry in s.actions[0].transitions:
            dest_s = hmm.states[entry.column]
            for obs in observation_classes:
                if obs in dest_s.labels:
                    transitions[s.id].append((obs, dest_s.id))
                    observations_missed.discard(obs)

        for obs in observations_missed:
            transitions[s.id].append((obs, rejecting_state))

    return nfa_dict_to_dfa(
        transitions,
        hmm.initial_states[0],
        set(range(len(hmm.states))),
        set(observation_classes),
    )


def accept_all_language(observation_classes: list[str]) -> Dfa[str]:
    """Construct a DFA accepting all traces over the given observation classes."""
    initial_state = DfaState(0, True)
    for obs in observation_classes:
        initial_state.transitions[obs] = initial_state
    return Dfa(initial_state, [initial_state])
