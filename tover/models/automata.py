from aalpy import Dfa
from stormpy import (
    SparseMdp,
    SparseExactMdp,
    DirectEncodingParserOptions,
    build_model_from_drn,
    BuilderOptions,
    build_sparse_model_with_options,
    build_sparse_exact_model_with_options,
    ExactSparseMatrixBuilder,
    SparseMatrixBuilder,
    StateLabeling,
    ChoiceLabeling,
    Rational,
    SparseExactModelComponents,
    SparseModelComponents,
    parse_prism_program,
)
from stormpy._core import _build_sparse_exact_model_from_drn
from stormvogel.model import Model, new_mdp

from tover.utils.logger import logger


def aalpy_dfa_to_stormvogel(dfa_a: Dfa) -> Model:
    """Convert an AALpy DFA to a Stormvogel MDP."""
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


def aalpy_dfa_to_stormpy(dfa_a: Dfa, use_exact: bool) -> SparseMdp | SparseExactMdp:
    """Convert an AALpy DFA to a Stormpy MDP."""
    state_labeling = StateLabeling(len(dfa_a.states))
    state_labeling.add_label("accepting")
    state_labeling.add_label("init")

    choice_labeling = ChoiceLabeling(sum([len(s.transitions) for s in dfa_a.states]))
    for label in dfa_a.get_input_alphabet():
        choice_labeling.add_label(label)

    state_mapping = {
        dfa_s.state_id: i
        for i, dfa_s in enumerate(sorted(dfa_a.states, key=lambda q: q.state_id))
    }

    state_labeling.add_label_to_state(
        "init", state_mapping[dfa_a.initial_state.state_id]
    )

    if use_exact:
        builder = ExactSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = SparseMatrixBuilder(0, 0, 0, False, True)

    current_row = 0
    for s in sorted(dfa_a.states, key=lambda q: q.state_id):
        if s.is_accepting:
            state_labeling.add_label_to_state("accepting", state_mapping[s.state_id])

        builder.new_row_group(current_row)
        for act, dest_s in s.transitions.items():
            builder.add_next_value(
                current_row,
                state_mapping[dest_s.state_id],
                Rational(1.0) if use_exact else 1.0,
            )
            choice_labeling.add_label_to_choice(act, current_row)
            current_row += 1

    matrix = builder.build(overridden_column_count=len(dfa_a.states))

    if use_exact:
        components = SparseExactModelComponents(matrix, state_labeling)
        components.choice_labeling = choice_labeling
        return SparseExactMdp(components)
    else:
        components = SparseModelComponents(matrix, state_labeling)
        components.choice_labeling = choice_labeling
        return SparseMdp(components)


def load_dfa_drn(path: str, use_exact: bool) -> SparseMdp:
    opts = DirectEncodingParserOptions()
    opts.build_choice_labels = True
    if use_exact:
        return _build_sparse_exact_model_from_drn(path, opts)
    else:
        return build_model_from_drn(path, opts)


def load_dfa_stormpy(path: str) -> SparseMdp:
    dfa_prism = parse_prism_program(path)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    return build_sparse_model_with_options(dfa_prism, options)


def load_dfa_stormpy_exact(path: str) -> SparseMdp:
    dfa_prism = parse_prism_program(path)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    return build_sparse_exact_model_with_options(dfa_prism, options)
